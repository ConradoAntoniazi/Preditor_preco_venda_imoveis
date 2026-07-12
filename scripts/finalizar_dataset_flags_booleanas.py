#!/usr/bin/env python3
"""Aplica o enriquecimento das paginas individuais ao dataset auditavel.

Somente registros com todas as flags ja resolvidas na base ou com pagina individual
processada com status=ok entram no dataset booleano final. Falhas, bloqueios e
anuncios indisponiveis permanecem em CSV separado; nunca viram False silenciosamente.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

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


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_rows(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"true", "1", "sim", "yes"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Gera dataset final sem nao_informado.")
    parser.add_argument("--base", required=True)
    parser.add_argument("--enrichment", required=True)
    parser.add_argument("--output-dir", default="dataset_flags_booleanas")
    args = parser.parse_args()

    base_rows = read_rows(Path(args.base))
    enrichment_rows = read_rows(Path(args.enrichment))

    latest_ok: dict[str, dict[str, str]] = {}
    for row in enrichment_rows:
        ad_id = str(row.get("id_anuncio") or "").strip()
        if ad_id and row.get("status") == "ok":
            latest_ok[ad_id] = row

    output_dir = Path(args.output_dir)
    model_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    pending_rows: list[dict[str, Any]] = []

    audit_columns = list(base_rows[0].keys()) if base_rows else []
    for extra in ["pagina_individual_status"]:
        if extra not in audit_columns:
            audit_columns.append(extra)
    for flag in FLAGS:
        for suffix in ["_final", "_fonte", "_evidencia", "_conflito"]:
            column = flag + suffix
            if column not in audit_columns:
                audit_columns.append(column)

    for row in base_rows:
        ad_id = str(row.get("id_anuncio") or "").strip()
        unresolved = any(str(row.get(flag) or "").strip() == "nao_informado" for flag in FLAGS)
        enrichment = latest_ok.get(ad_id)

        if unresolved and enrichment is None:
            pending = dict(row)
            pending["motivo_pendencia_flags"] = "pagina_individual_nao_processada_com_sucesso"
            pending_rows.append(pending)
            continue

        audit = dict(row)
        audit["pagina_individual_status"] = "ok" if enrichment else "nao_necessaria"
        model = {column: row.get(column, "") for column in MODEL_COLUMNS}

        for flag in FLAGS:
            if enrichment is not None:
                final_value = parse_bool(enrichment.get(flag))
                source = enrichment.get(f"{flag}_fonte", "")
                evidence = enrichment.get(f"{flag}_evidencia", "")
                conflict = parse_bool(enrichment.get(f"{flag}_conflito"))
            else:
                final_value = str(row.get(flag) or "").strip() == "sim"
                source = "base_resolvida"
                evidence = ""
                conflict = False
            model[flag] = final_value
            audit[f"{flag}_final"] = final_value
            audit[f"{flag}_fonte"] = source
            audit[f"{flag}_evidencia"] = evidence
            audit[f"{flag}_conflito"] = conflict

        model_rows.append(model)
        audit_rows.append(audit)

    pending_columns = list(base_rows[0].keys()) + ["motivo_pendencia_flags"] if base_rows else []
    write_rows(output_dir / "dataset_modelagem_flags_booleanas.csv", MODEL_COLUMNS, model_rows)
    write_rows(output_dir / "dataset_auditoria_flags_booleanas.csv", audit_columns, audit_rows)
    write_rows(output_dir / "dataset_flags_pendentes.csv", pending_columns, pending_rows)

    report = [
        "RELATORIO DE FINALIZACAO DAS FLAGS BOOLEANAS",
        "=" * 72,
        f"Registros na base: {len(base_rows)}",
        f"Paginas individuais OK disponiveis: {len(latest_ok)}",
        f"Registros no dataset booleano final: {len(model_rows)}",
        f"Registros pendentes: {len(pending_rows)}",
        "",
        "Semantica:",
        "True = caracteristica declarada na listagem, em propriedade estruturada ou na descricao.",
        "False = pagina individual processada com sucesso, mas sem evidencia positiva; ou negacao explicita.",
        "False nao garante ausencia fisica da caracteristica no imovel.",
    ]
    (output_dir / "relatorio_flags_booleanas.txt").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"Dataset final: {output_dir / 'dataset_modelagem_flags_booleanas.csv'}")
    print(f"Auditoria: {output_dir / 'dataset_auditoria_flags_booleanas.csv'}")
    print(f"Pendentes: {output_dir / 'dataset_flags_pendentes.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
