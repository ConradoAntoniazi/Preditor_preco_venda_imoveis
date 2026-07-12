#!/usr/bin/env python3
"""Reintegra anúncios removidos apenas por igualdade das 14 colunas do modelo.

Política:
- mantém removidos os anúncios cujo título normalizado é igual ao representante;
- reintegra anúncios cujo título normalizado é diferente;
- não reintegra registros que ainda possuem motivos_para_revisao, para não
  desfazer as decisões conservadoras de saneamento tomadas anteriormente;
- deriva o CSV de modelagem diretamente do CSV auditável resultante.

A normalização do título remove acentos, diferença entre maiúsculas/minúsculas,
pontuação e espaços repetidos. Não remove palavras, números ou códigos.
"""

from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from pathlib import Path

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


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV sem cabeçalho: {path}")
        return list(reader.fieldnames), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalize_title(value: str | None) -> str:
    text = unicodedata.normalize("NFKD", value or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def normalize_id(value: str | None) -> str:
    text = (value or "").strip()
    return re.sub(r"\.0$", "", text)


def numeric_text(value: str | None, integer: bool = False) -> str:
    text = (value or "").strip()
    if not text or text.lower() in {"nan", "none", "null", "<na>"}:
        return ""
    number = float(text.replace(",", "."))
    if integer:
        return str(int(round(number)))
    return f"{number:.1f}"


def bool_text(value: str | None) -> str:
    text = (value or "").strip().lower()
    if text in {"true", "1", "sim", "yes"}:
        return "True"
    if text in {"false", "0", "nao", "não", "no"}:
        return "False"
    raise ValueError(f"Valor booleano inválido: {value!r}")


def normalize_model_values(row: dict[str, str]) -> dict[str, str]:
    out = dict(row)
    out["preco_venda"] = numeric_text(row.get("preco_venda"), integer=True)
    out["area_m2"] = numeric_text(row.get("area_m2"))
    for column in ["quartos", "banheiros", "vagas"]:
        out[column] = numeric_text(row.get(column))
    for column in [
        "tem_suite",
        "tem_piscina",
        "tem_varanda",
        "tem_lazer",
        "tem_mobiliado",
        "tem_elevador",
    ]:
        out[column] = bool_text(row.get(column))
    for column in ["tipo_imovel", "cidade", "bairro"]:
        out[column] = (row.get(column) or "").strip()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-audit", required=True, type=Path)
    parser.add_argument("--removed-duplicates", required=True, type=Path)
    parser.add_argument("--removed-manual", required=True, type=Path)
    parser.add_argument("--removed-not-ready", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    audit_header, final_rows = read_csv(args.final_audit)
    duplicate_header, duplicate_rows = read_csv(args.removed_duplicates)
    _, manual_rows = read_csv(args.removed_manual)
    _, not_ready_rows = read_csv(args.removed_not_ready)

    required_audit = {"id_anuncio", "titulo_raw", *MODEL_COLUMNS}
    missing = sorted(required_audit - set(audit_header))
    if missing:
        raise ValueError(f"Colunas ausentes no auditável final: {missing}")

    required_duplicates = {
        "id_anuncio",
        "id_anuncio_mantido",
        "titulo_raw",
        "motivos_para_revisao",
        *MODEL_COLUMNS,
    }
    missing = sorted(required_duplicates - set(duplicate_header))
    if missing:
        raise ValueError(f"Colunas ausentes no arquivo de duplicatas: {missing}")

    final_by_id = {normalize_id(row.get("id_anuncio")): row for row in final_rows}
    manual_by_id = {normalize_id(row.get("id_anuncio")): row for row in manual_rows}
    not_ready_by_id = {normalize_id(row.get("id_anuncio")): row for row in not_ready_rows}
    representatives = {**final_by_id, **manual_by_id, **not_ready_by_id}

    restored: list[dict[str, str]] = []
    same_title: list[dict[str, str]] = []
    rejected_quality: list[dict[str, str]] = []

    for raw_row in duplicate_rows:
        row = normalize_model_values(raw_row)
        ad_id = normalize_id(row.get("id_anuncio"))
        representative_id = normalize_id(row.get("id_anuncio_mantido"))
        row["id_anuncio"] = ad_id
        row["id_anuncio_mantido"] = representative_id

        representative = representatives.get(representative_id)
        if representative is None:
            raise ValueError(
                f"Representante {representative_id} do anúncio {ad_id} não encontrado."
            )

        normalized_title = normalize_title(row.get("titulo_raw"))
        normalized_representative_title = normalize_title(representative.get("titulo_raw"))
        metadata = {
            "titulo_representante": representative.get("titulo_raw", ""),
            "titulo_normalizado": normalized_title,
            "titulo_representante_normalizado": normalized_representative_title,
        }

        if normalized_title == normalized_representative_title:
            enriched = {**row, **metadata}
            enriched["motivo_decisao_reintegracao"] = "titulo_normalizado_igual_ao_representante"
            same_title.append(enriched)
            continue

        review_reason = (row.get("motivos_para_revisao") or "").strip()
        if review_reason:
            enriched = {**row, **metadata}
            enriched["motivo_decisao_reintegracao"] = (
                "nao_reintegrado_por_motivos_para_revisao:" + review_reason
            )
            rejected_quality.append(enriched)
            continue

        audit_row = {column: row.get(column, "") for column in audit_header}
        restored.append(audit_row)

    original_ids = {normalize_id(row.get("id_anuncio")) for row in final_rows}
    restored_ids = {normalize_id(row.get("id_anuncio")) for row in restored}
    if len(restored_ids) != len(restored):
        raise RuntimeError("IDs duplicados entre os registros a reintegrar.")
    overlap = original_ids & restored_ids
    if overlap:
        raise RuntimeError(f"IDs já presentes no dataset final: {sorted(overlap)[:10]}")

    combined_audit = [normalize_model_values(row) for row in final_rows] + restored
    combined_ids = [normalize_id(row.get("id_anuncio")) for row in combined_audit]
    if len(combined_ids) != len(set(combined_ids)):
        raise RuntimeError("IDs duplicados no auditável resultante.")

    model_rows = [
        {column: row.get(column, "") for column in MODEL_COLUMNS}
        for row in combined_audit
    ]

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "dataset_final_auditavel.csv", audit_header, combined_audit)
    write_csv(output / "dataset_final_modelagem.csv", MODEL_COLUMNS, model_rows)

    decision_extra = [
        "titulo_representante",
        "titulo_normalizado",
        "titulo_representante_normalizado",
        "motivo_decisao_reintegracao",
    ]
    decision_header = duplicate_header + [c for c in decision_extra if c not in duplicate_header]
    write_csv(output / "reintegrados_titulos_distintos.csv", audit_header, restored)
    write_csv(output / "mantidos_removidos_mesmo_titulo.csv", decision_header, same_title)
    write_csv(output / "nao_reintegrados_por_qualidade.csv", decision_header, rejected_quality)

    report = [
        "RELATORIO DE REINTEGRACAO POR TITULO DISTINTO",
        "=" * 78,
        f"Registros no dataset final anterior: {len(final_rows)}",
        f"Registros originalmente removidos por igualdade das 14 colunas: {len(duplicate_rows)}",
        f"Reintegrados por título normalizado diferente: {len(restored)}",
        f"Mantidos removidos por título normalizado igual: {len(same_title)}",
        f"Não reintegrados por outras ressalvas de qualidade: {len(rejected_quality)}",
        f"Registros no novo dataset final: {len(combined_audit)}",
        f"IDs únicos no novo dataset: {len(set(combined_ids))}",
        "",
        "CRITÉRIO DE TÍTULO",
        "-" * 78,
        "A normalização remove apenas acentos, caixa, pontuação e espaços repetidos.",
        "Palavras e números são preservados; portanto, títulos semanticamente diferentes",
        "não são fundidos apenas por possuírem as mesmas 14 variáveis de modelagem.",
        "",
        "RESSALVA",
        "-" * 78,
        "As linhas reintegradas podem repetir exatamente as 14 colunas do modelo.",
        "Isso agora é intencional: títulos diferentes são tratados como anúncios distintos.",
        "Registros ainda marcados com motivos_para_revisao não foram reintegrados.",
    ]
    (output / "relatorio_reintegracao.txt").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    print("Reintegração concluída.")
    print(f"Reintegrados: {len(restored)}")
    print(f"Novo total: {len(combined_audit)}")
    print(f"Saída: {output}")


if __name__ == "__main__":
    main()
