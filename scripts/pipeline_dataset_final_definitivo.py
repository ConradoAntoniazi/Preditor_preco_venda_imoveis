#!/usr/bin/env python3
"""Pipeline auditável para gerar o dataset final de imóveis.

Fluxo:
  1) preparar: parte do dataset_auditoria_flags_booleanas.csv, remove registros
     com motivos_revisao, normaliza ausências, remove duplicatas EXATAS nas
     colunas do modelo e cria uma fila obrigatória de revisão para casos
     suspeitos de escopo ou valores extremos.
  2) finalizar: valida a planilha preenchida, aplica manter/remover e exporta
     o dataset final, sem motivos_revisao e sem duplicatas exatas.

O pipeline é fail-closed: não gera o dataset final enquanto houver decisões
faltantes, valores inválidos ou casos suspeitos não cobertos pela revisão.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FLAGS = [
    "tem_suite",
    "tem_piscina",
    "tem_varanda",
    "tem_lazer",
    "tem_mobiliado",
    "tem_elevador",
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
    *FLAGS,
]

REQUIRED_INPUT_COLUMNS = {
    "id_anuncio",
    "url_anuncio",
    "titulo_raw",
    "motivos_revisao",
    "preco_venda",
    "tipo_imovel",
    "cidade",
    "bairro",
    "area_m2",
    "quartos",
    "banheiros",
    "vagas",
    "pagina_individual_status",
    *(f"{flag}_final" for flag in FLAGS),
    *(f"{flag}_conflito" for flag in FLAGS),
}

REVIEW_COLUMNS = [
    "id_anuncio",
    "chave_revisao",
    "decisao_final",
    "justificativa_manual",
    "sugestao",
    "motivos_para_revisao",
    "titulo_raw",
    "url_anuncio",
    "preco_venda",
    "area_m2",
    "preco_m2",
    "tipo_imovel",
    "cidade",
    "bairro",
    "quartos",
    "banheiros",
    "vagas",
]

# Padrões de alta precisão: normalmente indicam que o anúncio não representa
# uma única casa/apartamento residencial comparável.
STRICT_SCOPE_RULES: list[tuple[str, str]] = [
    (
        "terreno_lote_como_objeto_principal",
        r"^(?:vendo|vende-se|oportunidade|imperdivel)?\s*(?:loteamento|lotes?|terrenos?)\b"
        r"|\b(?:loteamento|lotes?|terrenos?)\s+(?:a|à)\s+venda\b",
    ),
    (
        "multiplas_unidades",
        r"\b(?:0?[2-9]|1[0-9]|20|duas|dois|tres|quatro|cinco|seis|sete|oito|nove|dez)\s*"
        r"(?:casas?|apartamentos?|aptos?|kitnets?|lojas?|unidades?)\b"
        r"|\bpredio\b.*\b(?:apartamentos?|casas?|kitnets?|lojas?|pontos? comerciais?)\b"
        r"|\b(?:duas|dois|tres|quatro|cinco|seis|sete|oito|nove|dez)\s+imoveis\b",
    ),
    (
        "comercial_hospedagem",
        r"\b(?:pousada|hotel|hostel|hospedagem|galpao|sala comercial|salas comerciais|"
        r"ponto comercial|pontos comerciais|predio comercial|imovel comercial|loja comercial)\b",
    ),
    ("rural", r"\b(?:chacara|sitio|fazenda)\b"),
    ("locacao", r"\b(?:aluga-se|para alugar|para aluga|locacao|aluguel)\b"),
    (
        "vaga_garagem_avulsa",
        r"^(?:vendo\s+)?(?:uma\s+)?vagas?\s+de\s+garagem\b|\bgaragem\s+avulsa\b",
    ),
    ("produto_modular", r"\b(?:casa|cabana)\s+modular\b|\bcontainer\b"),
]

# Padrões deliberadamente amplos: não removem automaticamente; criam revisão.
AMBIGUOUS_SCOPE_RULES: list[tuple[str, str]] = [
    ("menciona_predio", r"\bpredio\b"),
    ("menciona_lote_terreno", r"\b(?:loteamento|lotes?|terrenos?)\b"),
    ("menciona_comercial", r"\b(?:comercial|lojas?|ponto comercial|salas? comerciais?)\b"),
    (
        "obra_construcao",
        r"\b(?:em construcao|na planta|inacabado|inacabada|obra|fase de acabamento|"
        r"em acabamento|para terminar|para acabar)\b",
    ),
    (
        "preco_pode_ser_parcial",
        r"\b(?:agio|cessao|entrada\s*\+|entrada de|parcelas?|parcelado|financiamento direto)\b",
    ),
    ("troca_permuta", r"\b(?:troca|troco|permuta)\b"),
]


def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value).lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_bool_series(series: pd.Series, column: str) -> pd.Series:
    true_values = {"true", "1", "sim", "yes"}
    false_values = {"false", "0", "nao", "não", "no"}

    def parse(value: Any) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        text = str(value).strip().lower()
        if text in true_values:
            return True
        if text in false_values:
            return False
        raise ValueError(f"Valor booleano inválido em {column}: {value!r}")

    return series.map(parse).astype(bool)


def nonblank(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.strip().str.lower()
    return ~text.isin({"", "nan", "none", "null", "<na>"})


def ensure_columns(df: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_INPUT_COLUMNS - set(df.columns))
    if missing:
        raise ValueError("Colunas obrigatórias ausentes: " + ", ".join(missing))


def normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["id_anuncio"] = out["id_anuncio"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    if out["id_anuncio"].eq("").any():
        raise ValueError("Há id_anuncio vazio.")
    duplicated_ids = out.loc[out["id_anuncio"].duplicated(False), "id_anuncio"].unique()
    if len(duplicated_ids):
        raise ValueError(f"Há {len(duplicated_ids)} IDs duplicados na entrada.")

    for col in ["preco_venda", "area_m2", "quartos", "banheiros", "vagas"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Preço e área são indispensáveis para o objetivo do modelo.
    invalid_core = out["preco_venda"].isna() | out["area_m2"].isna()
    if invalid_core.any():
        ids = out.loc[invalid_core, "id_anuncio"].head(10).tolist()
        raise ValueError(f"Preço/área ausentes na entrada. Exemplos de IDs: {ids}")

    if (out["preco_venda"] <= 0).any() or (out["area_m2"] <= 0).any():
        raise ValueError("Há preço ou área não positivos.")

    # Ausentes numéricos opcionais permanecem NaN. Zero não é convertido
    # genericamente em ausência: vagas=0 é válido; quartos=0 pode ser studio.
    out.loc[out["banheiros"] <= 0, "banheiros"] = np.nan
    out.loc[out["vagas"] < 0, "vagas"] = np.nan

    title_norm = out["titulo_raw"].map(normalize_text)
    studio_like = title_norm.str.contains(r"\b(?:studio|kitnet|kit net|flat|loft)\b", regex=True)
    out.loc[(out["quartos"] <= 0) & ~studio_like, "quartos"] = np.nan
    out.loc[out["quartos"] < 0, "quartos"] = np.nan

    out["bairro"] = out["bairro"].fillna("").astype(str).str.strip()
    out.loc[out["bairro"].eq(""), "bairro"] = "Não informado"
    for col in ["tipo_imovel", "cidade"]:
        out[col] = out[col].fillna("").astype(str).str.strip()
        if out[col].eq("").any():
            raise ValueError(f"Há valores vazios em {col}.")

    for flag in FLAGS:
        out[flag] = parse_bool_series(out[f"{flag}_final"], f"{flag}_final")
        conflict = parse_bool_series(out[f"{flag}_conflito"], f"{flag}_conflito")
        if conflict.any():
            ids = out.loc[conflict, "id_anuncio"].head(10).tolist()
            raise ValueError(
                f"Ainda existem conflitos em {flag}. Exemplos de IDs: {ids}. "
                "Use o enriquecimento revisado antes deste pipeline."
            )

    # Tipos numéricos estáveis para comparação/deduplicação.
    out["preco_venda"] = out["preco_venda"].round().astype("Int64")
    out["area_m2"] = out["area_m2"].round(4).astype("Float64")
    for col in ["quartos", "banheiros", "vagas"]:
        out[col] = out[col].round().astype("Int64")

    return out


def model_signature(row: pd.Series) -> str:
    values: list[str] = []
    for col in MODEL_COLUMNS:
        value = row[col]
        if pd.isna(value):
            values.append("<NA>")
        elif isinstance(value, (bool, np.bool_)):
            values.append("true" if bool(value) else "false")
        else:
            values.append(str(value).strip())
    return hashlib.sha256("|".join(values).encode("utf-8")).hexdigest()


def review_reasons(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    titles = df["titulo_raw"].map(normalize_text)
    reasons: list[list[str]] = [[] for _ in range(len(df))]
    suggestions: list[list[str]] = [[] for _ in range(len(df))]

    for reason, pattern in STRICT_SCOPE_RULES:
        mask = titles.str.contains(pattern, regex=True, na=False)
        for pos in np.flatnonzero(mask.to_numpy()):
            reasons[pos].append(reason)
            suggestions[pos].append("remover")

    for reason, pattern in AMBIGUOUS_SCOPE_RULES:
        mask = titles.str.contains(pattern, regex=True, na=False)
        for pos in np.flatnonzero(mask.to_numpy()):
            reasons[pos].append(reason)
            suggestions[pos].append("revisar")

    price = df["preco_venda"].astype("Float64")
    area = df["area_m2"].astype("Float64")
    price_m2 = price / area

    numeric_rules = [
        ("preco_abaixo_80000", price < 80_000),
        ("preco_acima_10000000", price > 10_000_000),
        ("area_abaixo_20m2", area < 20),
        ("area_acima_800m2", area > 800),
        ("preco_m2_abaixo_200", price_m2 < 200),
        ("preco_m2_acima_50000", price_m2 > 50_000),
    ]
    for reason, mask in numeric_rules:
        for pos in np.flatnonzero(mask.fillna(False).to_numpy()):
            reasons[pos].append(reason)
            suggestions[pos].append("revisar")

    reason_series = pd.Series(["|".join(sorted(set(x))) for x in reasons], index=df.index)

    def consolidate_suggestion(values: list[str]) -> str:
        return "remover" if "remover" in values else ("revisar" if values else "")

    suggestion_series = pd.Series([consolidate_suggestion(x) for x in suggestions], index=df.index)
    return reason_series, suggestion_series


def audit_quality_score(df: pd.DataFrame, suspicious: pd.Series) -> pd.Series:
    score = pd.Series(0, index=df.index, dtype="int64")
    score += df["pagina_individual_status"].fillna("").astype(str).str.lower().eq("ok").astype(int) * 100
    score -= suspicious.astype(int) * 50
    evidence_cols = [c for c in df.columns if c.endswith("_evidencia")]
    if evidence_cols:
        evidence_count = sum(nonblank(df[c]).astype(int) for c in evidence_cols)
        score += evidence_count
    return score


def make_review_key(row: pd.Series) -> str:
    raw = json.dumps(
        {
            "id_anuncio": str(row["id_anuncio"]),
            "motivos": str(row["motivos_para_revisao"]),
            "preco": None if pd.isna(row["preco_venda"]) else int(row["preco_venda"]),
            "area": None if pd.isna(row["area_m2"]) else float(row["area_m2"]),
            "titulo": str(row["titulo_raw"]),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig", na_rep="")


def prepare(args: argparse.Namespace) -> int:
    input_path = Path(args.audit)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(input_path, low_memory=False)
    ensure_columns(raw)
    normalized = normalize_dataset(raw)

    review_mask = nonblank(normalized["motivos_revisao"])
    removed_review = normalized.loc[review_mask].copy()
    removed_review["motivo_remocao_final"] = (
        "motivos_revisao:" + removed_review["motivos_revisao"].astype(str)
    )
    candidate = normalized.loc[~review_mask].copy()

    reasons, suggestions = review_reasons(candidate)
    candidate["motivos_para_revisao"] = reasons
    candidate["sugestao"] = suggestions
    suspicious = candidate["motivos_para_revisao"].ne("")

    # Duplicata exata = igualdade em TODAS as colunas usadas pelo modelo após
    # normalização. Mantemos a linha auditavelmente mais completa; empate por ID.
    candidate["assinatura_modelo"] = candidate.apply(model_signature, axis=1)
    candidate["qualidade_auditoria"] = audit_quality_score(candidate, suspicious)
    candidate["id_ordem"] = pd.to_numeric(candidate["id_anuncio"], errors="coerce").fillna(10**30)
    ordered = candidate.sort_values(
        ["assinatura_modelo", "qualidade_auditoria", "id_ordem"],
        ascending=[True, False, True],
        kind="stable",
    )
    keep_mask = ~ordered["assinatura_modelo"].duplicated(keep="first")
    kept = ordered.loc[keep_mask].copy()
    duplicates = ordered.loc[~keep_mask].copy()

    representative = kept.set_index("assinatura_modelo")["id_anuncio"].to_dict()
    duplicates["id_anuncio_mantido"] = duplicates["assinatura_modelo"].map(representative)
    duplicates["motivo_remocao_final"] = "duplicata_exata_todas_colunas_modelo"

    kept = kept.sort_index().copy()
    kept.drop(columns=["qualidade_auditoria", "id_ordem"], inplace=True)

    review = kept.loc[kept["motivos_para_revisao"].ne("")].copy()
    review["preco_m2"] = (
        review["preco_venda"].astype("Float64") / review["area_m2"].astype("Float64")
    ).round(2)
    review["chave_revisao"] = review.apply(make_review_key, axis=1)
    review["decisao_final"] = ""
    review["justificativa_manual"] = ""
    review = review[REVIEW_COLUMNS].sort_values(
        ["sugestao", "motivos_para_revisao", "cidade", "tipo_imovel", "id_anuncio"],
        kind="stable",
    )

    # A base preparada contém todos os candidatos únicos; o finalizador só usa
    # decisões vinculadas por id+hash, evitando aplicação em arquivo alterado.
    write_csv(kept, output_dir / "base_preparada_sem_duplicatas.csv")
    write_csv(removed_review, output_dir / "removidos_motivos_revisao.csv")
    write_csv(duplicates, output_dir / "removidos_duplicatas_exatas.csv")
    write_csv(review, output_dir / "revisao_final_obrigatoria.csv")

    report = [
        "RELATORIO DE PREPARACAO DO DATASET FINAL DEFINITIVO",
        "=" * 78,
        f"Registros de entrada: {len(normalized)}",
        f"Removidos por motivos_revisao: {len(removed_review)}",
        f"Removidos como duplicatas exatas: {len(duplicates)}",
        f"Registros únicos após essas remoções: {len(kept)}",
        f"Registros que exigem decisão manual: {len(review)}",
        "",
        "DUPLICATA EXATA",
        "-" * 78,
        "Duas linhas são duplicatas somente quando as 14 colunas do modelo são iguais",
        "após normalização de tipos e ausências. Uma linha é mantida e as demais",
        "são registradas com id_anuncio_mantido.",
        "",
        "PRÓXIMO PASSO",
        "-" * 78,
        "Preencha decisao_final com manter ou remover em revisao_final_obrigatoria.csv.",
        "Preencha justificativa_manual em todas as linhas. Depois execute finalizar.",
    ]
    (output_dir / "relatorio_preparacao.txt").write_text("\n".join(report) + "\n", encoding="utf-8")

    print("Preparação concluída.")
    print(f"Fila de revisão: {output_dir / 'revisao_final_obrigatoria.csv'}")
    print(f"Duplicatas exatas removidas: {len(duplicates)}")
    print(f"Decisões manuais necessárias: {len(review)}")
    return 0


def validate_review(base: pd.DataFrame, review: pd.DataFrame) -> pd.DataFrame:
    required = {"id_anuncio", "chave_revisao", "decisao_final", "justificativa_manual"}
    missing = sorted(required - set(review.columns))
    if missing:
        raise ValueError("Colunas ausentes na revisão: " + ", ".join(missing))

    review = review.copy()
    review["id_anuncio"] = review["id_anuncio"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    if review["id_anuncio"].duplicated().any():
        dupes = review.loc[review["id_anuncio"].duplicated(False), "id_anuncio"].unique()
        raise ValueError(f"IDs repetidos na planilha de revisão: {dupes[:10].tolist()}")

    expected = base.loc[base["motivos_para_revisao"].fillna("").ne("")].copy()
    expected["chave_revisao"] = expected.apply(make_review_key, axis=1)
    expected_keys = expected.set_index("id_anuncio")["chave_revisao"].to_dict()
    received_keys = review.set_index("id_anuncio")["chave_revisao"].astype(str).to_dict()

    if set(expected_keys) != set(received_keys):
        missing_ids = sorted(set(expected_keys) - set(received_keys))
        extra_ids = sorted(set(received_keys) - set(expected_keys))
        raise ValueError(
            f"A revisão não corresponde à fila original. Faltantes={missing_ids[:10]}, "
            f"extras={extra_ids[:10]}"
        )
    changed = [ad_id for ad_id, key in expected_keys.items() if received_keys.get(ad_id) != key]
    if changed:
        raise ValueError(
            "Dados estruturais da revisão foram alterados. IDs com hash incompatível: "
            + ", ".join(changed[:10])
        )

    review["decisao_final"] = review["decisao_final"].fillna("").astype(str).str.strip().str.lower()
    invalid = ~review["decisao_final"].isin({"manter", "remover"})
    if invalid.any():
        ids = review.loc[invalid, "id_anuncio"].head(10).tolist()
        raise ValueError(f"Decisões faltantes/inválidas. Use manter ou remover. IDs: {ids}")
    if (~nonblank(review["justificativa_manual"])).any():
        ids = review.loc[~nonblank(review["justificativa_manual"]), "id_anuncio"].head(10).tolist()
        raise ValueError(f"Justificativas faltantes. IDs: {ids}")

    return review[["id_anuncio", "decisao_final", "justificativa_manual"]]


def finalize(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir)
    review_path = Path(args.review)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_path = workdir / "base_preparada_sem_duplicatas.csv"
    if not base_path.exists():
        raise ValueError(f"Base preparada não encontrada: {base_path}")

    base = pd.read_csv(base_path, low_memory=False)
    review = pd.read_csv(review_path, low_memory=False)
    base["id_anuncio"] = base["id_anuncio"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    decisions = validate_review(base, review)

    merged = base.merge(decisions, on="id_anuncio", how="left", validate="one_to_one")
    reviewed_mask = merged["motivos_para_revisao"].fillna("").ne("")
    # Linhas não suspeitas entram automaticamente. Linhas suspeitas dependem da decisão.
    remove_mask = reviewed_mask & merged["decisao_final"].eq("remover")
    keep_mask = ~remove_mask

    removed_manual = merged.loc[remove_mask].copy()
    removed_manual["motivo_remocao_final"] = (
        "revisao_final_manual:" + removed_manual["justificativa_manual"].astype(str)
    )
    final_audit = merged.loc[keep_mask].copy()

    # Validações finais fortes.
    if nonblank(final_audit["motivos_revisao"]).any():
        raise ValueError("Falha interna: motivos_revisao presente no dataset final.")
    if final_audit["id_anuncio"].duplicated().any():
        raise ValueError("Falha interna: ID duplicado no dataset final.")

    # Não pode restar duplicata exata nas colunas de modelagem.
    duplicate_final = final_audit.duplicated(MODEL_COLUMNS, keep=False)
    if duplicate_final.any():
        ids = final_audit.loc[duplicate_final, "id_anuncio"].head(10).tolist()
        raise ValueError(f"Falha interna: duplicatas exatas permaneceram. IDs: {ids}")

    if final_audit["preco_venda"].isna().any() or final_audit["area_m2"].isna().any():
        raise ValueError("Preço ou área ausente no dataset final.")
    if (pd.to_numeric(final_audit["preco_venda"]) <= 0).any() or (
        pd.to_numeric(final_audit["area_m2"]) <= 0
    ).any():
        raise ValueError("Preço ou área não positivo no dataset final.")

    for flag in FLAGS:
        final_audit[flag] = parse_bool_series(final_audit[flag], flag)

    final_model = final_audit[MODEL_COLUMNS].copy()

    # Colunas de preparação não pertencem ao auditável final.
    prep_columns = {
        "assinatura_modelo",
        "sugestao",
        "decisao_final",
        "justificativa_manual",
    }
    audit_columns = [c for c in final_audit.columns if c not in prep_columns]
    final_audit = final_audit[audit_columns]

    write_csv(final_model, output_dir / "dataset_final_modelagem.csv")
    write_csv(final_audit, output_dir / "dataset_final_auditavel.csv")
    write_csv(removed_manual, output_dir / "removidos_revisao_final_manual.csv")

    # Copia as remoções anteriores para que a pasta final seja autocontida.
    for filename in ["removidos_motivos_revisao.csv", "removidos_duplicatas_exatas.csv"]:
        src = workdir / filename
        if src.exists():
            dst = output_dir / filename
            dst.write_bytes(src.read_bytes())

    missing_counts = final_model.isna().sum().to_dict()
    report = [
        "RELATORIO DO DATASET FINAL DEFINITIVO",
        "=" * 78,
        f"Registros na base preparada: {len(base)}",
        f"Removidos na revisão final manual: {len(removed_manual)}",
        f"Registros no dataset final: {len(final_model)}",
        f"IDs únicos no dataset final: {final_audit['id_anuncio'].nunique()}",
        f"Duplicatas exatas restantes: {int(final_model.duplicated(MODEL_COLUMNS).sum())}",
        "",
        "AUSÊNCIAS",
        "-" * 78,
        *(f"{col}: {int(missing_counts[col])}" for col in MODEL_COLUMNS),
        "",
        "SEMÂNTICA DAS FLAGS",
        "-" * 78,
        "True = característica declarada nas fontes consultadas.",
        "False = nenhuma evidência positiva encontrada após página validada ou negação.",
        "False não garante ausência física da característica.",
        "",
        "GARANTIAS DESTA EXECUÇÃO",
        "-" * 78,
        "- nenhum registro com motivos_revisao;",
        "- nenhum ID duplicado;",
        "- nenhuma linha exatamente duplicada nas 14 colunas do modelo;",
        "- preço e área presentes e positivos;",
        "- todos os casos suspeitos detectados receberam decisão explícita.",
    ]
    (output_dir / "relatorio_dataset_final.txt").write_text("\n".join(report) + "\n", encoding="utf-8")

    print("Finalização concluída.")
    print(f"Dataset de modelagem: {output_dir / 'dataset_final_modelagem.csv'}")
    print(f"Dataset auditável: {output_dir / 'dataset_final_auditavel.csv'}")
    print(f"Relatório: {output_dir / 'relatorio_dataset_final.txt'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pipeline definitivo do dataset OLX.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("preparar", help="Prepara base e fila de revisão.")
    p_prepare.add_argument("--audit", required=True)
    p_prepare.add_argument("--output-dir", default="saneamento_final_preparado")
    p_prepare.set_defaults(func=prepare)

    p_finalize = sub.add_parser("finalizar", help="Aplica revisão e gera dataset final.")
    p_finalize.add_argument("--workdir", required=True)
    p_finalize.add_argument("--review", required=True)
    p_finalize.add_argument("--output-dir", default="dataset_final_definitivo")
    p_finalize.set_defaults(func=finalize)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except (ValueError, FileNotFoundError, pd.errors.ParserError) as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
