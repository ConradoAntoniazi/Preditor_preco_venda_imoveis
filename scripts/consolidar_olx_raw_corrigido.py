#!/usr/bin/env python3
"""Consolida a coleta raw de listagens da OLX em uma base auditável.

Princípios desta etapa:
- preserva o CSV raw original;
- consolida anúncios repetidos por id_anuncio sem perder a trilha de auditoria;
- exige preço e área para a base candidata à modelagem;
- não remove outliers automaticamente: apenas os sinaliza para revisão;
- cria flags de características em três estados: sim, nao e nao_informado;
- não usa dados de páginas individuais, para evitar enriquecer apenas uma fração da base.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd


EXPECTED_CITIES = [
    "Vitória",
    "Vila Velha",
    "Serra",
    "Cariacica",
    "Viana",
    "Fundão",
    "Guarapari",
]

REQUIRED_RAW_COLUMNS = [
    "data_coleta_utc",
    "cidade_consultada",
    "tipo_consultado",
    "pagina",
    "id_anuncio",
    "url_anuncio",
    "titulo_raw",
    "preco_raw",
    "categoria_raw",
    "tipo_detalhado_raw",
    "cidade_raw",
    "bairro_raw",
    "area_raw",
    "quartos_raw",
    "banheiros_raw",
    "vagas_raw",
    "re_features_raw",
    "re_complex_features_raw",
]

MODEL_COLUMNS = [
    "preco_venda",
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

RAW_FIELDS_TO_MERGE = [
    "execucao_id",
    "data_coleta_utc",
    "cidade_consultada",
    "tipo_consultado",
    "pagina",
    "posicao_pagina",
    "url_listagem",
    "url_anuncio",
    "titulo_raw",
    "preco_raw",
    "categoria_raw",
    "tipo_detalhado_raw",
    "re_types_raw",
    "cidade_raw",
    "bairro_raw",
    "uf_raw",
    "area_raw",
    "quartos_raw",
    "banheiros_raw",
    "vagas_raw",
    "re_features_raw",
    "re_complex_features_raw",
    "anunciante_profissional_raw",
]

FEATURE_TERMS = {
    "tem_suite": {
        "positive": [r"\bsuite\b", r"\bsuites\b"],
        "negative": [
            r"\bsem\s+(?:uma\s+)?suite\b",
            r"\bnao\s+(?:possui|tem)\s+suite\b",
        ],
    },
    "tem_piscina": {
        "positive": [r"\bpiscina\b"],
        "negative": [
            r"\bsem\s+piscina\b",
            r"\bnao\s+(?:possui|tem)\s+piscina\b",
        ],
    },
    "tem_varanda": {
        "positive": [r"\bvaranda\b", r"\bsacada\b"],
        "negative": [
            r"\bsem\s+(?:varanda|sacada)\b",
            r"\bnao\s+(?:possui|tem)\s+(?:varanda|sacada)\b",
        ],
    },
    "tem_lazer": {
        "positive": [
            r"\blazer\b",
            r"\bacademia\b",
            r"\bfitness\b",
            r"\bplayground\b",
            r"\bsalao\s+de\s+festas\b",
            r"\bclube\b",
            r"\bbrinquedoteca\b",
            r"\bquadra\b",
            r"\bspa\b",
            r"\bsauna\b",
            r"\bespaco\s+gourmet\b",
        ],
        "negative": [
            r"\bsem\s+(?:area\s+de\s+)?lazer\b",
            r"\bnao\s+(?:possui|tem)\s+(?:area\s+de\s+)?lazer\b",
        ],
    },
    "tem_mobiliado": {
        "positive": [
            r"\bmobiliad[oa]s?\b",
            r"\bporteira\s+fechada\b",
        ],
        "negative": [
            r"\bnao\s+mobiliad[oa]\b",
            r"\bsem\s+mobilia\b",
        ],
    },
    "tem_elevador": {
        "positive": [r"\belevador(?:es)?\b"],
        "negative": [
            r"\bsem\s+elevador(?:es)?\b",
            r"\bnao\s+(?:possui|tem)\s+elevador(?:es)?\b",
        ],
    },
}


def normalize_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"\s+", " ", text)
    return text


def empty_to_na(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})


def parse_money(value: object) -> float | None:
    text = normalize_text(value)
    if not text:
        return None
    match = re.search(r"(?:r\$\s*)?([0-9][0-9.\s]*(?:,[0-9]{1,2})?)", text)
    if not match:
        return None
    token = match.group(1).replace(" ", "")
    if "," in token:
        token = token.replace(".", "").replace(",", ".")
    else:
        token = token.replace(".", "")
    try:
        number = float(token)
    except ValueError:
        return None
    return number if number > 0 else None


def parse_area(value: object) -> float | None:
    text = normalize_text(value)
    if not text:
        return None
    match = re.search(r"([0-9][0-9.\s]*(?:,[0-9]+)?)\s*(?:m2|m²|metros?\s+quadrados?)?", text)
    if not match:
        return None
    token = match.group(1).replace(" ", "")
    if "," in token:
        # Formato brasileiro: 1.234,56
        token = token.replace(".", "").replace(",", ".")
    elif "." in token:
        parts = token.split(".")
        # 1.000 ou 12.345: separador de milhar. 62.5: decimal.
        if len(parts) > 1 and all(len(part) == 3 for part in parts[1:]):
            token = "".join(parts)
    try:
        number = float(token)
    except ValueError:
        return None
    return number if number > 0 else None


def parse_count(value: object) -> tuple[int | None, bool]:
    text = normalize_text(value)
    if not text:
        return None, False
    censored = bool(re.search(r"\bou\s+mais\b|\+", text))
    match = re.search(r"\d+", text)
    if not match:
        return None, censored
    return int(match.group(0)), censored


def canonical_city(value: object) -> str | None:
    text = normalize_text(value)
    mapping = {normalize_text(city): city for city in EXPECTED_CITIES}
    mapping.update({
        "praia grande": "Fundão",
        "praia grande, fundao": "Fundão",
    })
    return mapping.get(text)


def infer_type(category_raw: object, tipo_consultado: object) -> tuple[str | None, str, bool]:
    category = normalize_text(category_raw)
    consulted = normalize_text(tipo_consultado)

    category_type = None
    has_apartment = "apartamento" in category
    has_house = "casa" in category
    if has_apartment and not has_house:
        category_type = "apartamento"
    elif has_house and not has_apartment:
        category_type = "casa"

    consulted_type = consulted if consulted in {"casa", "apartamento"} else None
    conflict = bool(category_type and consulted_type and category_type != consulted_type)

    if category_type:
        return category_type, "categoria_raw", conflict
    if category:
        # Categoria presente, mas diferente de Casas/Apartamentos: nao inventar tipo pela URL.
        return None, "categoria_raw_nao_reconhecida", conflict
    if consulted_type:
        return consulted_type, "tipo_consultado_fallback", conflict
    return None, "nao_identificado", conflict


def unique_nonempty(values: Iterable[object]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value is None or pd.isna(value):
            continue
        text = str(value).strip()
        if not text:
            continue
        if text not in result:
            result.append(text)
    return result


def consolidate_ids(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = raw.copy()
    raw["id_anuncio"] = raw["id_anuncio"].astype("string").str.strip()
    raw["data_coleta_dt"] = pd.to_datetime(raw["data_coleta_utc"], errors="coerce", utc=True)

    completeness_fields = [
        "preco_raw", "cidade_raw", "bairro_raw", "area_raw", "quartos_raw",
        "banheiros_raw", "vagas_raw", "re_features_raw", "re_complex_features_raw",
    ]
    raw["_completude"] = sum(
        raw[field].astype("string").str.strip().replace({"": pd.NA}).notna().astype(int)
        for field in completeness_fields
    )

    consolidated_rows = []
    duplicate_rows = []

    for ad_id, group in raw.groupby("id_anuncio", sort=False, dropna=False):
        ordered = group.sort_values(["_completude", "data_coleta_dt"], ascending=[False, False])
        base = ordered.iloc[0].copy()

        for field in RAW_FIELDS_TO_MERGE:
            if field not in group.columns:
                continue
            values = unique_nonempty(ordered[field].tolist())
            if values:
                base[field] = values[0]

        base["qtd_observacoes_id"] = len(group)
        base["paginas_id"] = "|".join(sorted({str(v) for v in group["pagina"].dropna().tolist()}))

        for field in ["preco_raw", "area_raw", "cidade_raw", "bairro_raw", "categoria_raw", "tipo_detalhado_raw"]:
            values = unique_nonempty(group[field].tolist()) if field in group.columns else []
            base[f"conflito_{field}"] = len(values) > 1
            base[f"valores_{field}"] = json.dumps(values, ensure_ascii=False)

        consolidated_rows.append(base)

        if len(group) > 1:
            duplicate_rows.append(group.drop(columns=["_completude", "data_coleta_dt"], errors="ignore"))

    consolidated = pd.DataFrame(consolidated_rows).drop(
        columns=["_completude", "data_coleta_dt"], errors="ignore"
    )
    duplicates = pd.concat(duplicate_rows, ignore_index=True) if duplicate_rows else raw.head(0).copy()
    duplicates = duplicates.drop(columns=["_completude", "data_coleta_dt"], errors="ignore")
    return consolidated, duplicates


def feature_state(structured_value: object, title_value: object, feature: str) -> tuple[str, str, bool]:
    structured = normalize_text(structured_value)
    title = normalize_text(title_value)
    terms = FEATURE_TERMS[feature]

    structured_positive = any(re.search(pattern, structured) for pattern in terms["positive"])
    title_negative = any(re.search(pattern, title) for pattern in terms["negative"])
    title_positive = any(re.search(pattern, title) for pattern in terms["positive"])

    conflict = structured_positive and title_negative
    if structured_positive:
        return "sim", "estruturado", conflict
    if title_negative:
        return "nao", "titulo_negacao", conflict
    if title_positive:
        return "sim", "titulo", conflict
    return "nao_informado", "ausente", conflict


def title_review_reasons(title: object, tipo_imovel: object) -> list[str]:
    text = normalize_text(title)
    if not text:
        return []

    reasons: list[str] = []

    strong_non_residential = [
        r"\bfigurinhas?\b",
        r"\bgalpao\b",
        r"\bsala\s+comercial\b",
        r"\bponto\s+comercial\b",
        r"\bpredio\s+comercial\b",
        r"\bloja\s+(?:a\s+venda|comercial|no|na)\b",
        r"\bterreno\s+(?:a\s+venda|vende-se|para\s+venda)\b",
        r"\blote\s+(?:a\s+venda|vende-se|para\s+venda)\b",
    ]
    if any(re.search(pattern, text) for pattern in strong_non_residential):
        reasons.append("titulo_sugere_fora_escopo_residencial")

    if re.search(r"\b(aluga-se|aluguel|locacao|para\s+alugar)\b", text):
        reasons.append("titulo_menciona_locacao")

    if tipo_imovel == "apartamento" and re.search(r"\b(casa|sobrado)\b", text):
        reasons.append("titulo_sugere_casa_em_categoria_apartamento")
    if tipo_imovel == "casa" and re.search(r"\b(apartamento|studio|kitnet|kitchenette)\b", text):
        reasons.append("titulo_sugere_apartamento_em_categoria_casa")

    return reasons


def build_report(df: pd.DataFrame, raw_count: int, duplicates: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("RELATORIO DE CONSOLIDACAO DA COLETA OLX")
    lines.append("=" * 88)
    lines.append(f"Linhas raw: {raw_count}")
    lines.append(f"Anuncios unicos consolidados: {len(df)}")
    lines.append(f"IDs com mais de uma observacao: {int((df['qtd_observacoes_id'] > 1).sum())}")
    lines.append(f"Linhas preservadas no arquivo de duplicatas: {len(duplicates)}")
    lines.append("")

    lines.append("DESTINO DOS REGISTROS")
    lines.append("-" * 88)
    lines.append(f"Base candidata a modelagem: {int(df['elegivel_modelagem_base'].sum())}")
    lines.append(f"Sem area valida: {int(df['sem_area'].sum())}")
    lines.append(f"Sem preco valido: {int(df['sem_preco'].sum())}")
    lines.append(f"Cidade fora do escopo: {int(df['cidade_fora_escopo'].sum())}")
    lines.append(f"Tipo nao reconhecido: {int(df['tipo_nao_reconhecido'].sum())}")
    lines.append(f"Transacao estruturada diferente de venda: {int(df['transacao_nao_venda'].sum())}")
    lines.append(f"Registros com algum motivo de revisao: {int(df['motivos_revisao'].astype('string').str.len().fillna(0).gt(0).sum())}")
    lines.append("")

    lines.append("COBERTURA POR CIDADE E TIPO")
    lines.append("-" * 88)
    coverage = (
        df.groupby(["cidade", "tipo_imovel"], dropna=False)
        .agg(
            registros=("id_anuncio", "size"),
            elegiveis=("elegivel_modelagem_base", "sum"),
            sem_area=("sem_area", "sum"),
            sem_preco=("sem_preco", "sum"),
        )
        .reset_index()
    )
    for _, row in coverage.iterrows():
        lines.append(
            f"{row['cidade']} | {row['tipo_imovel']}: "
            f"{int(row['registros'])} registros; {int(row['elegiveis'])} elegiveis; "
            f"{int(row['sem_area'])} sem area; {int(row['sem_preco'])} sem preco"
        )
    lines.append("")

    lines.append("FLAGS EM TRES ESTADOS NA BASE ELEGIVEL")
    lines.append("-" * 88)
    eligible = df[df["elegivel_modelagem_base"]]
    for feature in FEATURE_TERMS:
        counts = eligible[feature].value_counts(dropna=False).to_dict()
        lines.append(
            f"{feature}: sim={counts.get('sim', 0)}, nao={counts.get('nao', 0)}, "
            f"nao_informado={counts.get('nao_informado', 0)}"
        )
    lines.append("")

    lines.append("MOTIVOS DE REVISAO")
    lines.append("-" * 88)
    reason_counts: Counter[str] = Counter()
    for value in df["motivos_revisao"].dropna():
        for reason in str(value).split("|"):
            if reason:
                reason_counts[reason] += 1
    if reason_counts:
        for reason, count in reason_counts.most_common():
            lines.append(f"{reason}: {count}")
    else:
        lines.append("(nenhum)")
    lines.append("")

    lines.append("OBSERVACOES")
    lines.append("-" * 88)
    lines.append("- Nenhum outlier foi removido automaticamente nesta etapa.")
    lines.append("- Area e preco invalidos impedem entrada na base candidata.")
    lines.append("- Ausencia de mencao de caracteristica vira 'nao_informado', e nao 'nao'.")
    lines.append("- O arquivo dataset_revisao.csv deve ser inspecionado antes da limpeza final.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Consolida a coleta raw da OLX.")
    parser.add_argument("--raw", required=True, help="Caminho para olx_listagens_raw.csv")
    parser.add_argument("--output-dir", default="tratamento_olx", help="Diretorio de saida")
    args = parser.parse_args()

    raw_path = Path(args.raw)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        raise FileNotFoundError(f"Arquivo raw nao encontrado: {raw_path}")

    raw = pd.read_csv(raw_path, dtype="string", keep_default_na=False)
    missing_columns = [column for column in REQUIRED_RAW_COLUMNS if column not in raw.columns]
    if missing_columns:
        raise ValueError(
            "O CSV raw nao possui colunas obrigatorias: " + ", ".join(missing_columns)
        )

    for column in raw.columns:
        raw[column] = empty_to_na(raw[column])

    consolidated, duplicates = consolidate_ids(raw)

    # Normalizacao numerica.
    # Series.map pode converter os None retornados pelos parsers em float NaN.
    # NaN nao e None; portanto, round(NaN) gera ValueError. A conversao
    # vetorizada abaixo preserva os ausentes como <NA> nos dtypes anulaveis.
    preco_parsed = pd.to_numeric(
        consolidated["preco_raw"].map(parse_money),
        errors="coerce",
    ).replace([math.inf, -math.inf], pd.NA)
    consolidated["preco_venda"] = preco_parsed.round().astype("Int64")

    area_parsed = pd.to_numeric(
        consolidated["area_raw"].map(parse_area),
        errors="coerce",
    ).replace([math.inf, -math.inf], pd.NA)
    consolidated["area_m2"] = area_parsed.astype("Float64")

    for raw_column, output_column in [
        ("quartos_raw", "quartos"),
        ("banheiros_raw", "banheiros"),
        ("vagas_raw", "vagas"),
    ]:
        parsed = consolidated[raw_column].map(parse_count)
        consolidated[output_column] = pd.Series(
            [item[0] for item in parsed], index=consolidated.index, dtype="Int64"
        )
        consolidated[f"{output_column}_limite_inferior"] = [item[1] for item in parsed]

    # Cidade e tipo.
    city_raw_clean = empty_to_na(consolidated["cidade_raw"])
    consolidated["cidade"] = city_raw_clean.map(canonical_city)
    fallback_city = consolidated["cidade_consultada"].map(canonical_city)
    consolidated["cidade_origem"] = "cidade_raw"
    # Fallback somente quando a cidade raw realmente nao veio; cidade presente e desconhecida
    # deve continuar fora do escopo em vez de ser substituida pela cidade consultada.
    use_fallback_city = city_raw_clean.isna() & fallback_city.notna()
    consolidated.loc[use_fallback_city, "cidade"] = fallback_city[use_fallback_city]
    consolidated.loc[use_fallback_city, "cidade_origem"] = "cidade_consultada_fallback"

    type_results = consolidated.apply(
        lambda row: infer_type(row.get("categoria_raw"), row.get("tipo_consultado")), axis=1
    )
    consolidated["tipo_imovel"] = [item[0] for item in type_results]
    consolidated["tipo_origem"] = [item[1] for item in type_results]
    consolidated["conflito_tipo_consulta_categoria"] = [item[2] for item in type_results]

    consolidated["bairro"] = empty_to_na(consolidated["bairro_raw"])

    # Flags em tres estados.
    structured_text = (
        consolidated["re_features_raw"].fillna("").astype(str)
        + " | "
        + consolidated["re_complex_features_raw"].fillna("").astype(str)
    )
    for feature in FEATURE_TERMS:
        results = [
            feature_state(structured, title, feature)
            for structured, title in zip(structured_text, consolidated["titulo_raw"], strict=False)
        ]
        consolidated[feature] = [item[0] for item in results]
        consolidated[f"{feature}_origem"] = [item[1] for item in results]
        consolidated[f"{feature}_conflito"] = [item[2] for item in results]

    consolidated["sem_preco"] = consolidated["preco_venda"].isna() | (consolidated["preco_venda"] <= 0)
    consolidated["sem_area"] = consolidated["area_m2"].isna() | (consolidated["area_m2"] <= 0)
    consolidated["cidade_fora_escopo"] = ~consolidated["cidade"].isin(EXPECTED_CITIES)
    consolidated["tipo_nao_reconhecido"] = ~consolidated["tipo_imovel"].isin(["casa", "apartamento"])
    tipo_detalhado_norm = consolidated["tipo_detalhado_raw"].map(normalize_text)
    consolidated["transacao_nao_venda"] = tipo_detalhado_norm.str.contains(
        r"\b(?:aluguel|locacao)\b", regex=True, na=False
    )
    consolidated["cidade_divergente_consulta"] = (
        consolidated["cidade_raw"].map(canonical_city).notna()
        & consolidated["cidade_consultada"].map(canonical_city).notna()
        & (
            consolidated["cidade_raw"].map(canonical_city)
            != consolidated["cidade_consultada"].map(canonical_city)
        )
    )

    consolidated["preco_m2_auditoria"] = (
        consolidated["preco_venda"] / consolidated["area_m2"]
    ).round(2)
    consolidated["preco_extremo_auditoria"] = (
        consolidated["preco_venda"].notna()
        & ((consolidated["preco_venda"] < 40_000) | (consolidated["preco_venda"] > 20_000_000))
    )
    consolidated["area_extrema_auditoria"] = (
        consolidated["area_m2"].notna()
        & ((consolidated["area_m2"] < 10) | (consolidated["area_m2"] > 1000))
    )
    consolidated["preco_m2_extremo_auditoria"] = (
        consolidated["preco_m2_auditoria"].notna()
        & ((consolidated["preco_m2_auditoria"] < 300) | (consolidated["preco_m2_auditoria"] > 100_000))
    )

    reasons_all: list[str] = []
    for _, row in consolidated.iterrows():
        reasons: list[str] = []
        if row["cidade_fora_escopo"]:
            reasons.append("cidade_fora_escopo")
        if row["tipo_nao_reconhecido"]:
            reasons.append("tipo_nao_reconhecido")
        if row["transacao_nao_venda"]:
            reasons.append("transacao_nao_venda")
        if row["conflito_tipo_consulta_categoria"]:
            reasons.append("conflito_tipo_consulta_categoria")
        if row["cidade_divergente_consulta"]:
            reasons.append("cidade_divergente_consulta")
        reasons.extend(title_review_reasons(row.get("titulo_raw"), row.get("tipo_imovel")))
        if row["preco_extremo_auditoria"]:
            reasons.append("preco_extremo_auditoria")
        if row["area_extrema_auditoria"]:
            reasons.append("area_extrema_auditoria")
        if row["preco_m2_extremo_auditoria"]:
            reasons.append("preco_m2_extremo_auditoria")
        for feature in FEATURE_TERMS:
            if row[f"{feature}_conflito"]:
                reasons.append(f"conflito_{feature}")
        reasons_all.append("|".join(dict.fromkeys(reasons)))
    consolidated["motivos_revisao"] = reasons_all

    consolidated["elegivel_modelagem_base"] = ~(
        consolidated["sem_preco"]
        | consolidated["sem_area"]
        | consolidated["cidade_fora_escopo"]
        | consolidated["tipo_nao_reconhecido"]
        | consolidated["transacao_nao_venda"]
    )

    model = consolidated.loc[consolidated["elegivel_modelagem_base"], MODEL_COLUMNS].copy()
    model_with_id_columns = [
        "id_anuncio", "url_anuncio", "titulo_raw", "motivos_revisao",
        "quartos_limite_inferior", "banheiros_limite_inferior", "vagas_limite_inferior",
        *MODEL_COLUMNS,
    ]
    model_with_id = consolidated.loc[
        consolidated["elegivel_modelagem_base"], model_with_id_columns
    ].copy()
    sem_area = consolidated.loc[consolidated["sem_area"]].copy()
    sem_preco = consolidated.loc[consolidated["sem_preco"]].copy()
    fora_escopo = consolidated.loc[
        consolidated["cidade_fora_escopo"]
        | consolidated["tipo_nao_reconhecido"]
        | consolidated["transacao_nao_venda"]
    ].copy()
    revisao = consolidated.loc[consolidated["motivos_revisao"].astype("string").str.len().fillna(0) > 0].copy()

    # Ordenacao deterministica para facilitar comparacoes entre execucoes.
    sort_columns = ["cidade", "tipo_imovel", "bairro", "id_anuncio"]
    consolidated = consolidated.sort_values(sort_columns, na_position="last")
    model = model.sort_values(["cidade", "tipo_imovel", "bairro"], na_position="last")
    model_with_id = model_with_id.sort_values(
        ["cidade", "tipo_imovel", "bairro", "id_anuncio"], na_position="last"
    )
    sem_area = sem_area.sort_values(sort_columns, na_position="last")
    sem_preco = sem_preco.sort_values(sort_columns, na_position="last")
    fora_escopo = fora_escopo.sort_values(sort_columns, na_position="last")
    revisao = revisao.sort_values(sort_columns, na_position="last")

    encoding = "utf-8-sig"
    consolidated.to_csv(output_dir / "dataset_auditoria_consolidado.csv", index=False, encoding=encoding)
    model.to_csv(output_dir / "dataset_modelagem_base.csv", index=False, encoding=encoding)
    model_with_id.to_csv(
        output_dir / "dataset_modelagem_base_auditavel.csv", index=False, encoding=encoding
    )
    sem_area.to_csv(output_dir / "dataset_sem_area.csv", index=False, encoding=encoding)
    sem_preco.to_csv(output_dir / "dataset_sem_preco.csv", index=False, encoding=encoding)
    fora_escopo.to_csv(output_dir / "dataset_fora_escopo.csv", index=False, encoding=encoding)
    revisao.to_csv(output_dir / "dataset_revisao.csv", index=False, encoding=encoding)
    duplicates.to_csv(output_dir / "duplicatas_id_raw.csv", index=False, encoding=encoding)

    report = build_report(consolidated, len(raw), duplicates)
    (output_dir / "relatorio_consolidacao.txt").write_text(report, encoding="utf-8")

    print(report)
    print("\nArquivos gerados em:", output_dir.resolve())


if __name__ == "__main__":
    main()
