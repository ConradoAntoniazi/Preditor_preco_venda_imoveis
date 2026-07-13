from __future__ import annotations

import os
import time
import unicodedata
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from tqdm import tqdm

PASTA_SAIDAS = "saidas"
CAMINHO_DATASET_OUTRAS = os.path.join(PASTA_SAIDAS, "dataset_modelagem.csv")
CAMINHO_DATASET_OLX = "dataset_final_modelagem.csv"
CAMINHO_COORDENADAS = os.path.join(PASTA_SAIDAS, "long_lat.csv")

COLUNA_ALVO = "preco_venda"

COLUNAS_PREDITORAS = [
    "tipo_imovel",
    "cidade",
    "bairro",
    "area_m2",
    "quartos",
    "banheiros",
    "vagas",
    "tem_suite",
    "tem_piscina",
    "tem_varanda",
    "tem_lazer",
    "tem_mobiliado",
    "tem_elevador",
]

COLUNAS_MODELO = [COLUNA_ALVO, *COLUNAS_PREDITORAS]

COLUNAS_BOOLEANAS = [
    "tem_suite",
    "tem_piscina",
    "tem_varanda",
    "tem_lazer",
    "tem_mobiliado",
    "tem_elevador",
]

COLUNAS_NUMERICAS_BRUTAS = [
    "area_m2",
    "quartos",
    "banheiros",
    "vagas",
]

COLUNAS_ASSINATURA = list(COLUNAS_PREDITORAS)

MAPEAMENTO_TIPOS = {
    "casa de condominio": "casa",
    "sobrado": "casa",
    "terrea": "casa",
    "duplex": "apartamento",
    "studio": "apartamento",
    "flat": "apartamento",
    "cobertura": "apartamento",
}


@dataclass
class DadosPreparados:
    treino: pd.DataFrame
    teste: pd.DataFrame
    completo: pd.DataFrame
    coordenadas: pd.DataFrame
    rejeitados: pd.DataFrame


def normalizar_texto(valor: Any) -> str:
    if pd.isna(valor):
        return "nao informado"

    texto = str(valor).strip().lower()
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(
        caractere
        for caractere in texto
        if not unicodedata.combining(caractere)
    )
    return " ".join(texto.split()) or "nao informado"


def normalizar_booleano(valor: Any) -> float:
    if pd.isna(valor):
        return np.nan

    if isinstance(valor, (bool, np.bool_)):
        return float(valor)

    texto = normalizar_texto(valor)
    if texto in {"true", "1", "sim", "s", "yes"}:
        return 1.0
    if texto in {"false", "0", "nao", "n", "no"}:
        return 0.0

    raise ValueError(f"Valor booleano inesperado: {valor!r}")


def validar_colunas(
    df: pd.DataFrame,
    caminho: str,
    exige_alvo: bool = True,
) -> None:
    obrigatorias = COLUNAS_MODELO if exige_alvo else COLUNAS_PREDITORAS
    ausentes = [coluna for coluna in obrigatorias if coluna not in df.columns]
    if ausentes:
        raise ValueError(
            f"{caminho}: colunas obrigatorias ausentes: {ausentes}."
        )


def normalizar_dataframe(
    df: pd.DataFrame,
    exige_alvo: bool,
) -> pd.DataFrame:
    validar_colunas(df, "DataFrame", exige_alvo=exige_alvo)
    resultado = df.copy()

    for coluna in ["tipo_imovel", "cidade", "bairro"]:
        resultado[coluna] = resultado[coluna].map(normalizar_texto)

    resultado["tipo_imovel"] = resultado["tipo_imovel"].replace(
        MAPEAMENTO_TIPOS
    )

    colunas_numericas = list(COLUNAS_NUMERICAS_BRUTAS)
    if exige_alvo:
        colunas_numericas.insert(0, COLUNA_ALVO)

    for coluna in colunas_numericas:
        resultado[coluna] = pd.to_numeric(
            resultado[coluna],
            errors="coerce",
        )

    for coluna in COLUNAS_BOOLEANAS:
        resultado[coluna] = resultado[coluna].map(normalizar_booleano)

    return resultado


def carregar_datasets(
    caminho_outras: str = CAMINHO_DATASET_OUTRAS,
    caminho_olx: str = CAMINHO_DATASET_OLX,
) -> pd.DataFrame:
    df_outras = pd.read_csv(caminho_outras)
    df_olx = pd.read_csv(caminho_olx)

    validar_colunas(df_outras, caminho_outras, exige_alvo=True)
    validar_colunas(df_olx, caminho_olx, exige_alvo=True)

    df_outras = df_outras[COLUNAS_MODELO].copy()
    df_olx = df_olx[COLUNAS_MODELO].copy()

    df_outras["fonte_dataset"] = "outras_plataformas"
    df_olx["fonte_dataset"] = "olx"

    combinado = pd.concat([df_outras, df_olx], ignore_index=True)
    return normalizar_dataframe(combinado, exige_alvo=True)


def aplicar_regras_fixas(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    motivo = pd.Series("", index=df.index, dtype="string")

    def adicionar_motivo(mascara: pd.Series, texto: str) -> None:
        atual = motivo.loc[mascara].fillna("")
        motivo.loc[mascara] = atual.where(atual == "", atual + ";") + texto

    adicionar_motivo(df[COLUNA_ALVO].isna(), "preco_ausente")
    adicionar_motivo(df["area_m2"].isna(), "area_ausente")
    adicionar_motivo(
        ~df["tipo_imovel"].isin(["apartamento", "casa"]),
        "tipo_invalido",
    )
    adicionar_motivo(df[COLUNA_ALVO] < 80_000, "preco_abaixo_limite")
    adicionar_motivo(df[COLUNA_ALVO] > 10_000_000, "preco_acima_limite")
    adicionar_motivo(df["area_m2"] < 20, "area_abaixo_limite")
    adicionar_motivo(df["area_m2"] > 800, "area_acima_limite")

    mascara_rejeitada = motivo != ""
    rejeitados = df.loc[mascara_rejeitada].copy()
    rejeitados["motivo_rejeicao"] = motivo.loc[mascara_rejeitada]

    aceitos = df.loc[~mascara_rejeitada].copy()

    # Quartos, banheiros e vagas sao preservados como coletados.
    # Valores acima de 5 nao sao reduzidos nem removidos.
    return aceitos.reset_index(drop=True), rejeitados.reset_index(drop=True)


def gerar_coordenadas_bairros(
    df_imoveis: pd.DataFrame,
    caminho_saida: str = CAMINHO_COORDENADAS,
) -> pd.DataFrame:
    geolocator = Nominatim(user_agent="app_localizacao_imoveis_es")
    coordenadas = (
        df_imoveis[["bairro", "cidade"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    for indice, linha in tqdm(
        coordenadas.iterrows(),
        total=len(coordenadas),
        desc="Geocodificando bairros",
    ):
        endereco = f"{linha['bairro']}, {linha['cidade']}, ES, Brasil"
        try:
            localizacao = geolocator.geocode(endereco, timeout=10)
            if localizacao is not None:
                coordenadas.at[indice, "latitude"] = localizacao.latitude
                coordenadas.at[indice, "longitude"] = localizacao.longitude
            else:
                coordenadas.at[indice, "latitude"] = np.nan
                coordenadas.at[indice, "longitude"] = np.nan
        except Exception:
            coordenadas.at[indice, "latitude"] = np.nan
            coordenadas.at[indice, "longitude"] = np.nan

        time.sleep(1.5)

    os.makedirs(os.path.dirname(caminho_saida) or ".", exist_ok=True)
    coordenadas.to_csv(caminho_saida, index=False)
    return coordenadas


def carregar_coordenadas(
    df_imoveis: pd.DataFrame,
    caminho: str = CAMINHO_COORDENADAS,
) -> pd.DataFrame:
    if os.path.exists(caminho):
        coordenadas = pd.read_csv(caminho)
    else:
        coordenadas = gerar_coordenadas_bairros(df_imoveis, caminho)

    obrigatorias = {"cidade", "bairro", "latitude", "longitude"}
    ausentes = obrigatorias - set(coordenadas.columns)
    if ausentes:
        raise ValueError(
            f"Arquivo de coordenadas sem colunas: {sorted(ausentes)}"
        )

    coordenadas = coordenadas[
        ["cidade", "bairro", "latitude", "longitude"]
    ].copy()
    coordenadas["cidade"] = coordenadas["cidade"].map(normalizar_texto)
    coordenadas["bairro"] = coordenadas["bairro"].map(normalizar_texto)
    coordenadas["latitude"] = pd.to_numeric(
        coordenadas["latitude"], errors="coerce"
    )
    coordenadas["longitude"] = pd.to_numeric(
        coordenadas["longitude"], errors="coerce"
    )

    return (
        coordenadas.groupby(["cidade", "bairro"], as_index=False)
        .agg(
            latitude=("latitude", "median"),
            longitude=("longitude", "median"),
        )
    )


def adicionar_features_seguras(
    df: pd.DataFrame,
    coordenadas: pd.DataFrame,
) -> pd.DataFrame:
    resultado = df.merge(
        coordenadas,
        on=["cidade", "bairro"],
        how="left",
        validate="many_to_one",
    )

    resultado["cidade_bairro"] = (
        resultado["cidade"] + "__" + resultado["bairro"]
    )
    resultado["coordenada_ausente"] = (
        resultado["latitude"].isna()
        | resultado["longitude"].isna()
    ).astype(int)
    resultado["log_area_m2"] = np.log1p(resultado["area_m2"])
    resultado["area_por_quarto"] = (
        resultado["area_m2"] / (resultado["quartos"] + 1)
    )
    resultado["quantidade_amenidades"] = resultado[
        COLUNAS_BOOLEANAS
    ].sum(axis=1, min_count=1)

    return resultado


def assinatura_preditores(df: pd.DataFrame) -> pd.Series:
    dados = df[COLUNAS_ASSINATURA].copy()

    for coluna in ["tipo_imovel", "cidade", "bairro"]:
        dados[coluna] = dados[coluna].map(normalizar_texto)

    for coluna in COLUNAS_NUMERICAS_BRUTAS:
        dados[coluna] = (
            pd.to_numeric(dados[coluna], errors="coerce")
            .round(6)
            .astype("string")
            .fillna("<NA>")
        )

    for coluna in COLUNAS_BOOLEANAS:
        dados[coluna] = (
            pd.to_numeric(dados[coluna], errors="coerce")
            .astype("Int64")
            .astype("string")
            .fillna("<NA>")
        )

    return pd.util.hash_pandas_object(dados, index=False).astype("uint64")


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


def criar_estratos_grupos(
    df: pd.DataFrame,
    grupos: pd.Series,
    n_splits: int = 5,
) -> pd.Series | None:
    temporario = df.copy()
    temporario["_grupo"] = grupos.to_numpy()

    resumo = temporario.groupby("_grupo", as_index=False).agg(
        cidade=("cidade", "first"),
        tipo_imovel=("tipo_imovel", "first"),
        preco_mediano=(COLUNA_ALVO, "median"),
    )
    resumo["faixa_preco"] = _faixa_preco_por_grupo(resumo)

    candidatos = [
        ["cidade", "tipo_imovel", "faixa_preco"],
        ["tipo_imovel", "faixa_preco"],
        ["faixa_preco"],
    ]

    for colunas in candidatos:
        estrato = resumo[colunas].astype("string").agg("__".join, axis=1)
        contagem = estrato.value_counts()
        if not contagem.empty and int(contagem.min()) >= n_splits:
            mapa = pd.Series(estrato.to_numpy(), index=resumo["_grupo"])
            return grupos.map(mapa)

    return None


def particionar_dataset(
    df: pd.DataFrame,
    grupos: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    estratos = criar_estratos_grupos(df, grupos, n_splits=n_splits)

    if estratos is not None:
        divisor = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        splits = divisor.split(df, estratos, grupos)
    else:
        divisor = GroupKFold(n_splits=n_splits)
        splits = divisor.split(df, groups=grupos)

    indice_treino, indice_teste = next(splits)

    treino = df.iloc[indice_treino].copy()
    teste = df.iloc[indice_teste].copy()

    treino["_grupo_preditores"] = grupos.iloc[indice_treino].to_numpy()
    teste["_grupo_preditores"] = grupos.iloc[indice_teste].to_numpy()

    if set(treino["_grupo_preditores"]) & set(teste["_grupo_preditores"]):
        raise RuntimeError("Existem grupos simultaneamente no treino e teste.")

    return treino.reset_index(drop=True), teste.reset_index(drop=True)


def preparar_dados(
    caminho_outras: str = CAMINHO_DATASET_OUTRAS,
    caminho_olx: str = CAMINHO_DATASET_OLX,
    caminho_coordenadas: str = CAMINHO_COORDENADAS,
) -> DadosPreparados:
    combinado = carregar_datasets(caminho_outras, caminho_olx)
    aceitos, rejeitados = aplicar_regras_fixas(combinado)
    coordenadas = carregar_coordenadas(aceitos, caminho_coordenadas)
    completo = adicionar_features_seguras(aceitos, coordenadas)

    grupos = assinatura_preditores(completo)
    completo["_grupo_preditores"] = grupos.to_numpy()

    treino, teste = particionar_dataset(completo, grupos)

    return DadosPreparados(
        treino=treino,
        teste=teste,
        completo=completo.reset_index(drop=True),
        coordenadas=coordenadas,
        rejeitados=rejeitados,
    )


def preparar_entrada_inferencia(
    df: pd.DataFrame,
    coordenadas: pd.DataFrame,
) -> pd.DataFrame:
    normalizado = normalizar_dataframe(df, exige_alvo=False)
    tipos_invalidos = ~normalizado["tipo_imovel"].isin(["apartamento", "casa"])
    if tipos_invalidos.any():
        valores = sorted(normalizado.loc[tipos_invalidos, "tipo_imovel"].unique())
        raise ValueError(f"Tipos de imovel nao suportados: {valores}")

    return adicionar_features_seguras(normalizado, coordenadas)


class ModeloSegmentadoFinal:
    """Wrapper de inferencia para os modelos especializados por tipo."""

    def __init__(
        self,
        modelos: dict[str, Any],
        coordenadas: pd.DataFrame,
        metadados: dict[str, Any] | None = None,
    ):
        self.modelos = modelos
        self.coordenadas = coordenadas
        self.metadados = metadados or {}

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        preparado = preparar_entrada_inferencia(df, self.coordenadas)
        previsoes = np.full(len(preparado), np.nan, dtype=float)

        colunas_descartar = [
            COLUNA_ALVO,
            "id_anuncio",
            "fonte_dataset",
            "_grupo_preditores",
            "tipo_imovel",
        ]

        for tipo, modelo in self.modelos.items():
            mascara = preparado["tipo_imovel"] == tipo
            if not mascara.any():
                continue

            entrada = preparado.loc[mascara].drop(
                columns=colunas_descartar,
                errors="ignore",
            )
            previsoes[np.flatnonzero(mascara.to_numpy())] = modelo.predict(entrada)

        if np.isnan(previsoes).any():
            raise RuntimeError("Existem linhas sem modelo especializado aplicavel.")

        return np.maximum(previsoes, 0)
