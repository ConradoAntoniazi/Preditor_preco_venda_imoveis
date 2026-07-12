#!/usr/bin/env python3
"""Exporta conflitos de flags para revisao manual e aplica as decisoes.

Fluxo:
  1) exportar -> gera CSV com uma linha por flag conflitante.
  2) preencher decisao_manual com true ou false e justificativa_manual.
  3) aplicar -> gera um novo resultado_flags_olx revisado, sem alterar o original.

Somente a coluna da flag em conflito e seus campos de auditoria sao alterados.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
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

TRUE_VALUES = {"true", "1", "sim", "yes", "verdadeiro"}
FALSE_VALUES = {"false", "0", "nao", "não", "no", "falso"}

REVIEW_COLUMNS = [
    "id_anuncio",
    "url_anuncio",
    "titulo",
    "cidade",
    "bairro",
    "tipo_imovel",
    "flag_em_conflito",
    "valor_automatico",
    "fonte_automatica",
    "evidencia_automatica",
    "valor_base",
    "decisao_manual",
    "justificativa_manual",
    "motivos_revisao",
    "re_features_json",
    "re_complex_features_json",
]


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader.fieldnames or []), list(reader)


def write_rows(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_bool(value: Any) -> bool:
    normalized = str(value or "").strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise ValueError(f"valor booleano invalido: {value!r}")


def is_true(value: Any) -> bool:
    return str(value or "").strip().lower() in TRUE_VALUES


def key(ad_id: str, flag: str) -> tuple[str, str]:
    return ad_id.strip(), flag.strip()


def exportar(args: argparse.Namespace) -> int:
    _, base_rows = read_rows(Path(args.base))
    _, enrichment_rows = read_rows(Path(args.enrichment))

    base_by_id = {
        str(row.get("id_anuncio") or "").strip(): row
        for row in base_rows
        if str(row.get("id_anuncio") or "").strip()
    }

    review_rows: list[dict[str, Any]] = []
    ads: set[str] = set()
    counts = Counter()

    for row in enrichment_rows:
        if str(row.get("status") or "").strip().lower() != "ok":
            continue

        ad_id = str(row.get("id_anuncio") or "").strip()
        if not ad_id:
            continue
        base = base_by_id.get(ad_id, {})

        for flag in FLAGS:
            if not is_true(row.get(f"{flag}_conflito")):
                continue

            ads.add(ad_id)
            counts[flag] += 1
            review_rows.append(
                {
                    "id_anuncio": ad_id,
                    "url_anuncio": row.get("url_anuncio", "") or base.get("url_anuncio", ""),
                    "titulo": row.get("titulo_individual", "") or base.get("titulo_raw", ""),
                    "cidade": base.get("cidade", ""),
                    "bairro": base.get("bairro", ""),
                    "tipo_imovel": base.get("tipo_imovel", ""),
                    "flag_em_conflito": flag,
                    "valor_automatico": str(parse_bool(row.get(flag))),
                    "fonte_automatica": row.get(f"{flag}_fonte", ""),
                    "evidencia_automatica": row.get(f"{flag}_evidencia", ""),
                    "valor_base": base.get(flag, ""),
                    "decisao_manual": "",
                    "justificativa_manual": "",
                    "motivos_revisao": base.get("motivos_revisao", ""),
                    "re_features_json": row.get("re_features_json", ""),
                    "re_complex_features_json": row.get("re_complex_features_json", ""),
                }
            )

    review_rows.sort(key=lambda r: (r["id_anuncio"], r["flag_em_conflito"]))
    output = Path(args.output)
    write_rows(output, REVIEW_COLUMNS, review_rows)

    report_path = output.with_name(output.stem + "_relatorio.txt")
    report = [
        "RELATORIO DE EXPORTACAO DE CONFLITOS",
        "=" * 72,
        f"Anuncios com conflito: {len(ads)}",
        f"Flags conflitantes: {len(review_rows)}",
        "",
        "CONFLITOS POR FLAG",
        "-" * 72,
    ]
    report.extend(f"{flag}: {counts[flag]}" for flag in FLAGS)
    report.extend(
        [
            "",
            "INSTRUCOES",
            "-" * 72,
            "Preencha decisao_manual com true ou false em todas as linhas.",
            "Use justificativa_manual para registrar o motivo da decisao.",
            "Nao altere id_anuncio nem flag_em_conflito.",
        ]
    )
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"Planilha CSV: {output}")
    print(f"Relatorio: {report_path}")
    print(f"Anuncios com conflito: {len(ads)}")
    print(f"Flags conflitantes: {len(review_rows)}")
    return 0


def aplicar(args: argparse.Namespace) -> int:
    enrichment_columns, enrichment_rows = read_rows(Path(args.enrichment))
    review_columns, review_rows = read_rows(Path(args.review))

    missing_columns = [column for column in REVIEW_COLUMNS if column not in review_columns]
    if missing_columns:
        raise SystemExit(
            "A planilha de revisao nao possui as colunas obrigatorias: "
            + ", ".join(missing_columns)
        )

    current_conflicts: set[tuple[str, str]] = set()
    for row in enrichment_rows:
        if str(row.get("status") or "").strip().lower() != "ok":
            continue
        ad_id = str(row.get("id_anuncio") or "").strip()
        for flag in FLAGS:
            if ad_id and is_true(row.get(f"{flag}_conflito")):
                current_conflicts.add(key(ad_id, flag))

    decisions: dict[tuple[str, str], tuple[bool, str]] = {}
    errors: list[str] = []

    for line_number, row in enumerate(review_rows, start=2):
        ad_id = str(row.get("id_anuncio") or "").strip()
        flag = str(row.get("flag_em_conflito") or "").strip()
        decision_raw = str(row.get("decisao_manual") or "").strip()
        justification = str(row.get("justificativa_manual") or "").strip()

        if flag not in FLAGS:
            errors.append(f"linha {line_number}: flag invalida {flag!r}")
            continue
        review_key = key(ad_id, flag)
        if review_key in decisions:
            errors.append(f"linha {line_number}: conflito duplicado {ad_id}/{flag}")
            continue
        if not decision_raw:
            errors.append(f"linha {line_number}: decisao_manual vazia para {ad_id}/{flag}")
            continue
        try:
            decision = parse_bool(decision_raw)
        except ValueError:
            errors.append(
                f"linha {line_number}: decisao_manual invalida para {ad_id}/{flag}: {decision_raw!r}"
            )
            continue
        decisions[review_key] = (decision, justification)

    missing_reviews = sorted(current_conflicts - set(decisions))
    extra_reviews = sorted(set(decisions) - current_conflicts)
    if missing_reviews:
        errors.append(
            f"faltam decisoes para {len(missing_reviews)} conflito(s), por exemplo: "
            + ", ".join(f"{ad}/{flag}" for ad, flag in missing_reviews[:5])
        )
    if extra_reviews:
        errors.append(
            f"a planilha contem {len(extra_reviews)} decisao(oes) que nao correspondem a conflitos atuais, por exemplo: "
            + ", ".join(f"{ad}/{flag}" for ad, flag in extra_reviews[:5])
        )

    if errors:
        print("A aplicacao foi cancelada. Corrija a planilha:")
        for error in errors[:30]:
            print(f"- {error}")
        if len(errors) > 30:
            print(f"- ... e mais {len(errors) - 30} erro(s)")
        return 2

    applied = 0
    affected_ads: set[str] = set()

    for row in enrichment_rows:
        ad_id = str(row.get("id_anuncio") or "").strip()
        for flag in FLAGS:
            review_key = key(ad_id, flag)
            if review_key not in decisions:
                continue

            decision, justification = decisions[review_key]
            old_value = row.get(flag, "")
            old_source = row.get(f"{flag}_fonte", "")
            old_evidence = row.get(f"{flag}_evidencia", "")

            row[flag] = str(decision)
            row[f"{flag}_fonte"] = "revisao_manual"
            row[f"{flag}_conflito"] = "False"
            row[f"{flag}_evidencia"] = (
                f"decisao_manual={decision}; justificativa={justification or '[nao informada]'}; "
                f"valor_automatico={old_value}; fonte_automatica={old_source}; "
                f"evidencia_automatica={old_evidence}"
            )
            affected_ads.add(ad_id)
            applied += 1

    output = Path(args.output)
    write_rows(output, enrichment_columns, enrichment_rows)

    report_path = output.with_name(output.stem + "_relatorio.txt")
    report = [
        "RELATORIO DE APLICACAO DA REVISAO MANUAL",
        "=" * 72,
        f"Anuncios afetados: {len(affected_ads)}",
        f"Flags revisadas: {applied}",
        f"Conflitos restantes no arquivo de saida: 0",
        "",
        "O arquivo original de enriquecimento nao foi alterado.",
        "Use o arquivo revisado como --enrichment no finalizar_dataset_flags_booleanas.py.",
    ]
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"Enriquecimento revisado: {output}")
    print(f"Relatorio: {report_path}")
    print(f"Flags revisadas: {applied}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exporta e aplica revisoes manuais dos conflitos das flags OLX."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser(
        "exportar", help="Gera CSV com uma linha por flag conflitante."
    )
    export_parser.add_argument("--base", required=True)
    export_parser.add_argument("--enrichment", required=True)
    export_parser.add_argument("--output", default="revisao_conflitos_flags.csv")
    export_parser.set_defaults(func=exportar)

    apply_parser = subparsers.add_parser(
        "aplicar", help="Aplica as decisoes manuais e gera novo enriquecimento."
    )
    apply_parser.add_argument("--enrichment", required=True)
    apply_parser.add_argument("--review", required=True)
    apply_parser.add_argument("--output", default="resultado_flags_olx_revisado.csv")
    apply_parser.set_defaults(func=aplicar)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
