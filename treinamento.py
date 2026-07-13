from __future__ import annotations

import json
import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import (
    GroupKFold,
    RandomizedSearchCV,
    StratifiedGroupKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import make_scorer
from xgboost import XGBRegressor

from utils import ModeloSegmentadoFinal, preparar_dados

PASTA_SAIDAS = "saidas"
ARQUIVO_MODELO = "modelo_segmentado_final.joblib"
ARQUIVO_RELATORIO = os.path.join(PASTA_SAIDAS, "relatorio_modelo_final.json")
ARQUIVO_PREVISOES = os.path.join(PASTA_SAIDAS, "previsoes_teste_final.csv")

N_ITERACOES_POR_TIPO = {
    "apartamento": 25,
    "casa": 40,
}

CONFIGURACAO_POR_TIPO = {
    "apartamento": {
        "nome": "sem_target_encoding",
        "modo": None,
        "smoothing": None,
    },
    "casa": {
        "nome": "log_preco_m2_s15",
        "modo": "log_preco_m2",
        "smoothing": 15.0,
    },
}

COLUNAS_CATEGORICAS = [
    "cidade",
    "bairro",
    "cidade_bairro",
]

COLUNAS_NUMERICAS = [
    "area_m2",
    "quartos",
    "banheiros",
    "vagas",
    "latitude",
    "longitude",
    "coordenada_ausente",
    "area_por_quarto",
    "log_area_m2",
    "quantidade_amenidades",
]

COLUNAS_BOOLEANAS = [
    "tem_suite",
    "tem_piscina",
    "tem_varanda",
    "tem_lazer",
    "tem_mobiliado",
    "tem_elevador",
]

COLUNAS_NAO_PREDITIVAS = [
    "preco_venda",
    "id_anuncio",
    "fonte_dataset",
    "_grupo_preditores",
    "tipo_imovel",
]


class CrossFittedTargetEncodingBairro(BaseEstimator, TransformerMixin):
    """Codificacao-alvo segura por cidade e bairro.

    Durante o ajuste, as linhas de treino recebem valores cross-fitted.
    Durante transformacoes posteriores, sao usados apenas mapas aprendidos
    no conjunto fornecido ao ``fit``.
    """

    def __init__(
        self,
        modo: str = "log_preco_m2",
        smoothing: float = 15.0,
        n_splits: int = 5,
        coluna_saida: str = "target_encoding_bairro",
    ):
        self.modo = modo
        self.smoothing = smoothing
        self.n_splits = n_splits
        self.coluna_saida = coluna_saida

    @staticmethod
    def _como_dataframe(X: Any) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("O target encoder requer pandas.DataFrame.")
        return X.copy()

    @staticmethod
    def _assinatura_preditores(X: pd.DataFrame) -> pd.Series:
        colunas = [
            coluna
            for coluna in [
                "cidade",
                "bairro",
                "area_m2",
                "quartos",
                "banheiros",
                "vagas",
                *COLUNAS_BOOLEANAS,
            ]
            if coluna in X.columns
        ]
        dados = X[colunas].astype("string").fillna("<NA>")
        return pd.util.hash_pandas_object(dados, index=False).astype("uint64")

    def _alvo_codificacao(self, X: pd.DataFrame, y: Any) -> pd.Series:
        area = pd.to_numeric(X["area_m2"], errors="coerce")
        preco = pd.Series(np.asarray(y, dtype=float), index=X.index)
        preco_m2 = (preco / area).replace([np.inf, -np.inf], np.nan)

        if self.modo == "preco_m2":
            return preco_m2
        if self.modo == "log_preco_m2":
            return np.log1p(preco_m2)
        raise ValueError(f"Modo de target encoding invalido: {self.modo}")

    def _ajustar_mapas(self, X: pd.DataFrame, y: Any) -> None:
        dados = X[["cidade", "bairro", "area_m2"]].copy()
        dados["_alvo_te"] = self._alvo_codificacao(X, y)
        dados = dados.dropna(subset=["_alvo_te"])

        if dados.empty:
            self.media_global_ = 0.0
            self.mapa_cidade_ = {}
            self.mapa_bairro_ = {}
            return

        self.media_global_ = float(dados["_alvo_te"].mean())

        stats_cidade = dados.groupby("cidade")["_alvo_te"].agg(["sum", "count"])
        stats_cidade["valor"] = (
            stats_cidade["sum"] + self.smoothing * self.media_global_
        ) / (stats_cidade["count"] + self.smoothing)
        self.mapa_cidade_ = stats_cidade["valor"].to_dict()

        stats_bairro = dados.groupby(["cidade", "bairro"])["_alvo_te"].agg(
            ["sum", "count"]
        )
        self.mapa_bairro_ = {}
        for (cidade, bairro), linha in stats_bairro.iterrows():
            fallback = self.mapa_cidade_.get(cidade, self.media_global_)
            self.mapa_bairro_[(cidade, bairro)] = (
                linha["sum"] + self.smoothing * fallback
            ) / (linha["count"] + self.smoothing)

    def fit(self, X: pd.DataFrame, y: Any):
        X_df = self._como_dataframe(X)
        self._ajustar_mapas(X_df, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_df = self._como_dataframe(X)
        X_df[self.coluna_saida] = [
            self.mapa_bairro_.get(
                (cidade, bairro),
                self.mapa_cidade_.get(cidade, self.media_global_),
            )
            for cidade, bairro in zip(X_df["cidade"], X_df["bairro"])
        ]
        return X_df

    def fit_transform(self, X: pd.DataFrame, y: Any = None, **fit_params):
        if y is None:
            return self.fit(X, y).transform(X)

        X_df = self._como_dataframe(X).reset_index(drop=True)
        y_series = pd.Series(np.asarray(y, dtype=float), index=X_df.index)
        grupos = self._assinatura_preditores(X_df)
        quantidade_grupos = int(grupos.nunique())

        if quantidade_grupos < 2:
            self.fit(X_df, y_series)
            return self.transform(X_df)

        n_splits = min(self.n_splits, quantidade_grupos)
        divisor = GroupKFold(n_splits=n_splits)
        resultado = X_df.copy()
        resultado[self.coluna_saida] = np.nan

        for indice_ajuste, indice_validacao in divisor.split(
            X_df,
            y_series,
            grupos,
        ):
            codificador = CrossFittedTargetEncodingBairro(
                modo=self.modo,
                smoothing=self.smoothing,
                n_splits=self.n_splits,
                coluna_saida=self.coluna_saida,
            )
            codificador.fit(
                X_df.iloc[indice_ajuste],
                y_series.iloc[indice_ajuste],
            )
            transformado = codificador.transform(X_df.iloc[indice_validacao])
            resultado.loc[indice_validacao, self.coluna_saida] = transformado[
                self.coluna_saida
            ].to_numpy()

        self.fit(X_df, y_series)
        resultado[self.coluna_saida] = resultado[self.coluna_saida].fillna(
            self.media_global_
        )
        return resultado


def espaco_de_busca() -> dict[str, list[Any]]:
    """Busca final concentrada na regiao vencedora das rodadas anteriores."""
    return {
        "modelo__regressor__max_depth": [6, 7, 8, 9],
        "modelo__regressor__learning_rate": [0.03, 0.04, 0.05, 0.06],
        "modelo__regressor__n_estimators": [600, 750, 900, 1100, 1300],
        "modelo__regressor__subsample": [0.55, 0.60, 0.70, 0.80],
        "modelo__regressor__colsample_bytree": [0.55, 0.60, 0.70, 0.80],
        "modelo__regressor__reg_alpha": [0.0, 0.1, 0.3, 0.5],
        "modelo__regressor__reg_lambda": [0.5, 1.0, 2.0, 3.0, 5.0],
        "modelo__regressor__min_child_weight": [1, 2, 3, 5],
        "modelo__regressor__gamma": [0.0, 0.05, 0.1],
    }


def construir_pipeline(
    configuracao: dict[str, Any],
    parametros_xgb: dict[str, Any] | None = None,
) -> Pipeline:
    colunas_numericas = list(COLUNAS_NUMERICAS)
    etapas: list[tuple[str, Any]] = []

    if configuracao["modo"] is not None:
        etapas.append(
            (
                "target_encoder",
                CrossFittedTargetEncodingBairro(
                    modo=configuracao["modo"],
                    smoothing=configuracao["smoothing"],
                    n_splits=5,
                ),
            )
        )
        colunas_numericas.append("target_encoding_bairro")

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categoricas",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=True,
                            ),
                        ),
                    ]
                ),
                COLUNAS_CATEGORICAS,
            ),
            (
                "numericas",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="median", add_indicator=True),
                        ),
                        ("scaler", StandardScaler()),
                    ]
                ),
                colunas_numericas + COLUNAS_BOOLEANAS,
            ),
        ],
        remainder="drop",
    )

    parametros = {
        "random_state": 42,
        "n_jobs": 1,
        "objective": "reg:squarederror",
        "tree_method": "hist",
    }
    if parametros_xgb:
        parametros.update(parametros_xgb)

    regressor = XGBRegressor(**parametros)
    modelo = TransformedTargetRegressor(
        regressor=regressor,
        func=np.log1p,
        inverse_func=np.expm1,
    )

    etapas.extend(
        [
            ("preprocessor", preprocessor),
            ("modelo", modelo),
        ]
    )
    return Pipeline(etapas)


def _faixa_preco_por_grupo(resumo: pd.DataFrame) -> pd.Series:
    codigos = pd.qcut(
        resumo["preco_mediano"],
        q=4,
        labels=False,
        duplicates="drop",
    )
    return codigos.map(
        lambda valor: (
            f"Q{int(valor) + 1}"
            if pd.notna(valor)
            else "Q_nao_informado"
        )
    ).astype("string")


def criar_splits_cv(
    X: pd.DataFrame,
    y: pd.Series,
    grupos: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    resumo = pd.DataFrame(
        {
            "grupo": grupos.to_numpy(),
            "cidade": X["cidade"].astype("string").to_numpy(),
            "preco": y.to_numpy(),
        }
    ).groupby("grupo", as_index=False).agg(
        cidade=("cidade", "first"),
        preco_mediano=("preco", "median"),
    )

    if len(resumo) < n_splits:
        raise ValueError(
            f"Ha apenas {len(resumo)} grupos para {n_splits} folds."
        )

    resumo["faixa_preco"] = _faixa_preco_por_grupo(resumo)
    estratos = None

    for colunas in [["cidade", "faixa_preco"], ["faixa_preco"]]:
        candidato = resumo[colunas].astype("string").agg("__".join, axis=1)
        contagem = candidato.value_counts()
        if not contagem.empty and int(contagem.min()) >= n_splits:
            mapa = pd.Series(candidato.to_numpy(), index=resumo["grupo"])
            estratos = grupos.map(mapa)
            break

    if estratos is not None:
        divisor = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        splits = list(divisor.split(X, estratos, grupos))
    else:
        divisor = GroupKFold(n_splits=n_splits)
        splits = list(divisor.split(X, y, grupos))

    for indice_treino, indice_validacao in splits:
        if set(grupos.iloc[indice_treino]) & set(grupos.iloc[indice_validacao]):
            raise RuntimeError("Grupo presente simultaneamente em treino e validacao.")

    return splits


def calcular_metricas(y_true: Any, y_pred: Any) -> dict[str, float]:
    verdadeiro = np.asarray(y_true, dtype=float)
    previsto = np.maximum(np.asarray(y_pred, dtype=float), 0)
    erro_percentual = np.abs((verdadeiro - previsto) / verdadeiro)

    return {
        "r2": float(r2_score(verdadeiro, previsto)),
        "mape_percent": float(
            mean_absolute_percentage_error(verdadeiro, previsto) * 100
        ),
        "mdape_percent": float(np.median(erro_percentual) * 100),
        "mae": float(mean_absolute_error(verdadeiro, previsto)),
        "medae": float(median_absolute_error(verdadeiro, previsto)),
        "percentual_erro_ate_15": float(
            (erro_percentual <= 0.15).mean() * 100
        ),
        "bias_medio": float(np.mean(previsto - verdadeiro)),
    }


def executar_busca(
    X: pd.DataFrame,
    y: pd.Series,
    grupos: pd.Series,
    configuracao: dict[str, Any],
    n_iter: int,
) -> tuple[RandomizedSearchCV, list[tuple[np.ndarray, np.ndarray]]]:
    splits = criar_splits_cv(X, y, grupos)
    scorer = make_scorer(
        mean_absolute_percentage_error,
        greater_is_better=False,
    )

    busca = RandomizedSearchCV(
        estimator=construir_pipeline(configuracao),
        param_distributions=espaco_de_busca(),
        n_iter=n_iter,
        scoring=scorer,
        cv=splits,
        random_state=42,
        n_jobs=-1,
        verbose=0,
        refit=True,
        return_train_score=False,
    )
    busca.fit(X, y)
    return busca, splits


def prever_oof(
    estimador: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    previsoes = np.full(len(X), np.nan)

    for indice_treino, indice_validacao in splits:
        modelo = clone(estimador)
        modelo.fit(X.iloc[indice_treino], y.iloc[indice_treino])
        previsoes[indice_validacao] = modelo.predict(X.iloc[indice_validacao])

    if np.isnan(previsoes).any():
        raise RuntimeError("Existem registros sem previsao OOF.")

    return np.maximum(previsoes, 0)


def extrair_xy(
    df: pd.DataFrame,
    tipo: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    segmento = df.loc[df["tipo_imovel"] == tipo].copy().reset_index(drop=True)
    grupos = segmento["_grupo_preditores"].reset_index(drop=True)
    y = segmento["preco_venda"].reset_index(drop=True)
    metadados = segmento[
        ["fonte_dataset", "cidade", "bairro", "tipo_imovel"]
    ].reset_index(drop=True)
    X = segmento.drop(columns=COLUNAS_NAO_PREDITIVAS, errors="ignore")
    return X, y, grupos, metadados


def main() -> None:
    os.makedirs(PASTA_SAIDAS, exist_ok=True)
    dados = preparar_dados()

    modelos_avaliacao: dict[str, Pipeline] = {}
    modelos_finais: dict[str, Pipeline] = {}
    resultados_tipo: dict[str, Any] = {}
    previsoes_teste_partes: list[pd.DataFrame] = []
    previsoes_oof_geral: list[pd.DataFrame] = []

    for tipo in ["apartamento", "casa"]:
        configuracao = CONFIGURACAO_POR_TIPO[tipo]
        X_treino, y_treino, grupos_treino, meta_treino = extrair_xy(
            dados.treino,
            tipo,
        )
        X_teste, y_teste, _, meta_teste = extrair_xy(dados.teste, tipo)

        print(f"Buscando hiperparametros para {tipo}...")
        busca, splits = executar_busca(
            X_treino,
            y_treino,
            grupos_treino,
            configuracao,
            n_iter=N_ITERACOES_POR_TIPO[tipo],
        )

        melhor_modelo = busca.best_estimator_
        previsoes_oof = prever_oof(
            melhor_modelo,
            X_treino,
            y_treino,
            splits,
        )
        previsoes_teste = np.maximum(melhor_modelo.predict(X_teste), 0)

        metricas_oof = calcular_metricas(y_treino, previsoes_oof)
        metricas_teste = calcular_metricas(y_teste, previsoes_teste)

        modelos_avaliacao[tipo] = melhor_modelo

        previsoes_oof_geral.append(
            meta_treino.assign(
                preco_real=y_treino.to_numpy(),
                previsao=previsoes_oof,
            )
        )
        previsoes_teste_partes.append(
            meta_teste.assign(
                preco_real=y_teste.to_numpy(),
                previsao=previsoes_teste,
            )
        )

        X_completo, y_completo, _, _ = extrair_xy(dados.completo, tipo)
        parametros_finais = {
            chave.split("modelo__regressor__", 1)[1]: valor
            for chave, valor in busca.best_params_.items()
        }
        modelo_final = construir_pipeline(
            configuracao,
            parametros_xgb=parametros_finais,
        )
        modelo_final.fit(X_completo, y_completo)
        modelos_finais[tipo] = modelo_final

        resultados_tipo[tipo] = {
            "configuracao_entrada": configuracao,
            "n_treino": int(len(X_treino)),
            "n_teste": int(len(X_teste)),
            "n_completo": int(len(X_completo)),
            "n_iter_busca": N_ITERACOES_POR_TIPO[tipo],
            "melhores_hiperparametros": parametros_finais,
            "mape_medio_randomized_search_percent": float(
                -busca.best_score_ * 100
            ),
            "metricas_oof_pos_selecao": metricas_oof,
            "metricas_teste_externo": metricas_teste,
        }

        print(
            f"{tipo}: MAPE OOF={metricas_oof['mape_percent']:.2f}% | "
            f"MAPE teste={metricas_teste['mape_percent']:.2f}%"
        )

    previsoes_oof_df = pd.concat(previsoes_oof_geral, ignore_index=True)
    previsoes_teste_df = pd.concat(previsoes_teste_partes, ignore_index=True)

    metricas_oof_geral = calcular_metricas(
        previsoes_oof_df["preco_real"],
        previsoes_oof_df["previsao"],
    )
    metricas_teste_geral = calcular_metricas(
        previsoes_teste_df["preco_real"],
        previsoes_teste_df["previsao"],
    )

    previsoes_teste_df["erro_percentual"] = (
        np.abs(
            previsoes_teste_df["preco_real"]
            - previsoes_teste_df["previsao"]
        )
        / previsoes_teste_df["preco_real"]
        * 100
    )
    previsoes_teste_df.to_csv(ARQUIVO_PREVISOES, index=False)

    relatorio = {
        "linhas_aceitas": int(len(dados.completo)),
        "linhas_rejeitadas": int(len(dados.rejeitados)),
        "linhas_treino_externo": int(len(dados.treino)),
        "linhas_teste_externo": int(len(dados.teste)),
        "resultados_por_tipo": resultados_tipo,
        "metricas_oof_gerais_pos_selecao": metricas_oof_geral,
        "metricas_teste_externo_gerais": metricas_teste_geral,
        "modelo_final_retreinado_com_todos_os_dados": True,
    }

    with open(ARQUIVO_RELATORIO, "w", encoding="utf-8") as arquivo:
        json.dump(relatorio, arquivo, ensure_ascii=False, indent=2)

    bundle = ModeloSegmentadoFinal(
        modelos=modelos_finais,
        coordenadas=dados.coordenadas,
        metadados={
            "configuracoes_por_tipo": CONFIGURACAO_POR_TIPO,
            "resultados_avaliacao": resultados_tipo,
            "metricas_teste_externo_gerais": metricas_teste_geral,
        },
    )
    joblib.dump(bundle, ARQUIVO_MODELO)

    print(
        "Concluido: "
        f"MAPE OOF geral={metricas_oof_geral['mape_percent']:.2f}% | "
        f"MAPE teste geral={metricas_teste_geral['mape_percent']:.2f}%"
    )
    print(f"Modelo final: {ARQUIVO_MODELO}")
    print(f"Relatorio: {ARQUIVO_RELATORIO}")


if __name__ == "__main__":
    main()
