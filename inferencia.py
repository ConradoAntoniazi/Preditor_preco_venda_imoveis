"""Executa inferencia com o modelo segmentado final.

Entrada esperada: arquivo CSV contendo as 13 variaveis preditoras originais:

- tipo_imovel
- cidade
- bairro
- area_m2
- quartos
- banheiros
- vagas
- tem_suite
- tem_piscina
- tem_varanda
- tem_lazer
- tem_mobiliado
- tem_elevador

O arquivo ``modelo_segmentado_final.joblib`` ja contem:
- os modelos especializados para casa e apartamento;
- os preprocessadores;
- o target encoding usado pelo modelo de casas;
- as coordenadas dos bairros;
- as regras de normalizacao e criacao de features.

Por isso, este script envia ao bundle apenas as variaveis brutas. Nenhuma
estatistica de preco e recalculada durante a inferencia.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Estes imports sao necessarios porque o arquivo joblib contem objetos cujas
# classes foram definidas nesses modulos. Mantenha utils.py e treinamento.py
# disponiveis no mesmo projeto ao carregar o modelo.
import utils
import treinamento


def registrar_compatibilidade_joblib() -> None:
    """Registra classes antigas salvas como pertencentes a ``__main__``.

    O modelo atual foi serializado durante a execucao direta de
    ``treinamento.py``. Por isso, o joblib gravou algumas classes com o
    caminho ``__main__.NomeDaClasse``. Durante a inferencia, ``__main__``
    passa a ser este arquivo, e a desserializacao nao encontra essas classes.

    Este registro e apenas uma camada de compatibilidade para carregar o
    modelo ja treinado. Em um novo treinamento, o ideal e manter essas
    classes em um modulo importavel estavel.
    """
    modulo_main = sys.modules["__main__"]

    classes_compatibilidade = {
        "CrossFittedTargetEncodingBairro": getattr(
            treinamento,
            "CrossFittedTargetEncodingBairro",
            None,
        ),
        "ModeloSegmentadoFinal": getattr(
            utils,
            "ModeloSegmentadoFinal",
            None,
        ),
    }

    for nome, classe in classes_compatibilidade.items():
        if classe is not None:
            setattr(modulo_main, nome, classe)


MODELO_PADRAO = "modelo_segmentado_final.joblib"
SAIDA_PADRAO = "previsoes_inferencia.csv"
COLUNA_PREVISAO_PADRAO = "preco_venda_estimado"


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Gera estimativas de preco de venda para imoveis residenciais "
            "usando o modelo segmentado final."
        )
    )
    parser.add_argument(
        "--entrada",
        required=True,
        help="Caminho do CSV com os imoveis a estimar.",
    )
    parser.add_argument(
        "--modelo",
        default=MODELO_PADRAO,
        help=f"Caminho do bundle joblib. Padrao: {MODELO_PADRAO}",
    )
    parser.add_argument(
        "--saida",
        default=SAIDA_PADRAO,
        help=f"CSV de saida. Padrao: {SAIDA_PADRAO}",
    )
    parser.add_argument(
        "--coluna-previsao",
        default=COLUNA_PREVISAO_PADRAO,
        help=(
            "Nome da coluna que recebera a estimativa. "
            f"Padrao: {COLUNA_PREVISAO_PADRAO}"
        ),
    )
    parser.add_argument(
        "--estrito",
        action="store_true",
        help=(
            "Interrompe a execucao quando houver area ausente ou fora da "
            "faixa observada no treinamento (20 a 800 m2). Sem esta opcao, "
            "o script apenas emite avisos."
        ),
    )
    return parser


def ler_csv(caminho: Path) -> pd.DataFrame:
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo de entrada nao encontrado: {caminho}")

    # sep=None permite detectar virgula, ponto e virgula ou tabulacao.
    # utf-8-sig tambem aceita arquivos UTF-8 com BOM.
    try:
        return pd.read_csv(
            caminho,
            sep=None,
            engine="python",
            encoding="utf-8-sig",
        )
    except UnicodeDecodeError:
        return pd.read_csv(
            caminho,
            sep=None,
            engine="python",
            encoding="latin-1",
        )


def validar_estrutura(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("O arquivo de entrada nao possui registros.")

    colunas_duplicadas = df.columns[df.columns.duplicated()].tolist()
    if colunas_duplicadas:
        raise ValueError(
            f"O CSV possui colunas duplicadas: {sorted(set(colunas_duplicadas))}"
        )

    ausentes = [
        coluna
        for coluna in utils.COLUNAS_PREDITORAS
        if coluna not in df.columns
    ]
    if ausentes:
        raise ValueError(
            "Colunas obrigatorias ausentes: " + ", ".join(ausentes)
        )


def diagnosticar_dominio(
    df: pd.DataFrame,
    estrito: bool,
) -> list[str]:
    """Valida a semantica da entrada sem duplicar o preprocessamento.

    A normalizacao real continuara sendo executada pelo bundle salvo. Aqui a
    funcao e usada apenas para produzir mensagens mais claras antes da
    predicao.
    """
    normalizado = utils.normalizar_dataframe(
        df[utils.COLUNAS_PREDITORAS],
        exige_alvo=False,
    )

    avisos: list[str] = []

    tipos_invalidos = ~normalizado["tipo_imovel"].isin(
        ["apartamento", "casa"]
    )
    if tipos_invalidos.any():
        valores = sorted(
            normalizado.loc[tipos_invalidos, "tipo_imovel"]
            .dropna()
            .unique()
            .tolist()
        )
        raise ValueError(
            "Tipos de imovel nao suportados apos normalizacao: "
            + ", ".join(map(str, valores))
        )

    area = pd.to_numeric(normalizado["area_m2"], errors="coerce")
    area_ausente = area.isna()
    area_fora = area.notna() & ~area.between(20, 800)

    if area_ausente.any():
        avisos.append(
            f"{int(area_ausente.sum())} registro(s) com area ausente ou invalida."
        )

    if area_fora.any():
        avisos.append(
            f"{int(area_fora.sum())} registro(s) com area fora da faixa "
            "observada no treinamento (20 a 800 m2)."
        )

    bairros_ausentes = normalizado["bairro"].eq("nao informado")
    if bairros_ausentes.any():
        avisos.append(
            f"{int(bairros_ausentes.sum())} registro(s) sem bairro informado; "
            "a localizacao tera menor poder preditivo."
        )

    cidades_ausentes = normalizado["cidade"].eq("nao informado")
    if cidades_ausentes.any():
        avisos.append(
            f"{int(cidades_ausentes.sum())} registro(s) sem cidade informada."
        )

    if estrito and (area_ausente.any() or area_fora.any()):
        raise ValueError("Entrada fora do dominio do treinamento: " + " ".join(avisos))

    return avisos


def carregar_modelo(caminho: Path):
    if not caminho.exists():
        raise FileNotFoundError(f"Modelo nao encontrado: {caminho}")

    registrar_compatibilidade_joblib()
    modelo = joblib.load(caminho)
    if not hasattr(modelo, "predict"):
        raise TypeError(
            "O arquivo carregado nao possui uma interface predict compativel."
        )

    return modelo


def gerar_previsoes(
    modelo,
    entrada_original: pd.DataFrame,
    coluna_previsao: str,
) -> pd.DataFrame:
    # O wrapper salvo espera somente as 13 variaveis preditoras brutas.
    entrada_modelo = entrada_original[utils.COLUNAS_PREDITORAS].copy()
    previsoes = np.asarray(modelo.predict(entrada_modelo), dtype=float)

    if len(previsoes) != len(entrada_original):
        raise RuntimeError(
            "O modelo retornou uma quantidade de previsoes diferente "
            "da quantidade de registros de entrada."
        )

    if not np.isfinite(previsoes).all():
        indices = np.flatnonzero(~np.isfinite(previsoes)).tolist()
        raise RuntimeError(
            f"O modelo produziu previsoes nao finitas nas linhas: {indices[:20]}"
        )

    resultado = entrada_original.copy()
    resultado[coluna_previsao] = np.maximum(previsoes, 0).round(2)
    return resultado


def salvar_resultado(df: pd.DataFrame, caminho: Path) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(caminho, index=False, encoding="utf-8-sig")


def main() -> int:
    parser = construir_parser()
    args = parser.parse_args()

    try:
        caminho_entrada = Path(args.entrada)
        caminho_modelo = Path(args.modelo)
        caminho_saida = Path(args.saida)

        entrada = ler_csv(caminho_entrada)
        validar_estrutura(entrada)
        avisos = diagnosticar_dominio(entrada, estrito=args.estrito)

        modelo = carregar_modelo(caminho_modelo)
        resultado = gerar_previsoes(
            modelo,
            entrada_original=entrada,
            coluna_previsao=args.coluna_previsao,
        )
        salvar_resultado(resultado, caminho_saida)

        previsoes = resultado[args.coluna_previsao]
        print(f"Registros processados: {len(resultado)}")
        print(f"Saida gerada: {caminho_saida}")
        print(
            "Resumo das estimativas: "
            f"min=R$ {previsoes.min():,.2f} | "
            f"mediana=R$ {previsoes.median():,.2f} | "
            f"max=R$ {previsoes.max():,.2f}"
        )

        for aviso in avisos:
            print(f"Aviso: {aviso}", file=sys.stderr)

        return 0

    except Exception as erro:
        print(f"Erro na inferencia: {erro}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
