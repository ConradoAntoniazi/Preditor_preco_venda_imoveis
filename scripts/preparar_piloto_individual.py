from __future__ import annotations

import argparse
import csv
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urljoin


DEFAULT_RAW = "coleta_olx_completa/olx_listagens_raw.csv"


def is_blank(value: Any) -> bool:
    return value is None or str(value).strip() == ""


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "").strip().lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", text)


def normalize_url(value: Any) -> str:
    url = str(value or "").strip()
    if not url:
        return ""
    if url.startswith("//"):
        return "https:" + url
    return urljoin("https://www.olx.com.br", url)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def completeness_score(row: dict[str, str]) -> tuple[int, str]:
    important = [
        "preco_raw",
        "cidade_raw",
        "bairro_raw",
        "area_raw",
        "quartos_raw",
        "banheiros_raw",
        "vagas_raw",
        "re_features_raw",
        "re_complex_features_raw",
    ]
    score = sum(not is_blank(row.get(column)) for column in important)
    return score, row.get("data_coleta_utc", "")


def deduplicate_by_id(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    best: dict[str, dict[str, str]] = {}
    without_id: list[dict[str, str]] = []

    for row in rows:
        ad_id = str(row.get("id_anuncio") or "").strip()
        row = dict(row)
        row["url_anuncio"] = normalize_url(row.get("url_anuncio"))

        if not ad_id:
            without_id.append(row)
            continue

        current = best.get(ad_id)
        if current is None or completeness_score(row) > completeness_score(current):
            best[ad_id] = row

    result = list(best.values()) + without_id
    result.sort(
        key=lambda row: (
            row.get("cidade_consultada", ""),
            row.get("tipo_consultado", ""),
            int(row.get("pagina") or 0),
            int(row.get("posicao_pagina") or 0),
        )
    )
    return result


def city_mismatch(row: dict[str, str]) -> bool:
    expected = normalize_text(row.get("cidade_consultada"))
    returned = normalize_text(row.get("cidade_raw"))
    return bool(expected and returned and expected != returned)


def add_reason(
    selected: dict[str, dict[str, Any]],
    row: dict[str, str],
    reason: str,
    priority: int,
) -> None:
    ad_id = str(row.get("id_anuncio") or "").strip()
    if not ad_id or is_blank(row.get("url_anuncio")):
        return

    if ad_id not in selected:
        selected[ad_id] = {
            "row": dict(row),
            "reasons": set(),
            "priority": priority,
        }
    selected[ad_id]["reasons"].add(reason)
    selected[ad_id]["priority"] = min(selected[ad_id]["priority"], priority)


def grouped(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    result: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        result[(row.get("cidade_consultada", ""), row.get("tipo_consultado", ""))].append(row)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepara a triagem da coleta raw e uma amostra estratificada para "
            "explorar paginas individuais da OLX."
        )
    )
    parser.add_argument("--raw", default=DEFAULT_RAW)
    parser.add_argument("--output-dir", default="piloto_paginas_individuais")
    parser.add_argument(
        "--sem-area-base",
        type=int,
        default=2,
        help="Amostra minima sem area por cidade/tipo.",
    )
    parser.add_argument(
        "--sem-area-alta",
        type=int,
        default=6,
        help="Amostra sem area quando ao menos metade do grupo esta incompleta.",
    )
    parser.add_argument("--controles-por-grupo", type=int, default=1)
    parser.add_argument("--divergentes", type=int, default=6)
    parser.add_argument("--max-total", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260710)
    args = parser.parse_args()

    raw_path = Path(args.raw)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fieldnames, raw_rows = read_csv(raw_path)
    unique_rows = deduplicate_by_id(raw_rows)

    sem_preco = [row for row in unique_rows if is_blank(row.get("preco_raw"))]
    pendentes_area = [
        row
        for row in unique_rows
        if not is_blank(row.get("preco_raw")) and is_blank(row.get("area_raw"))
    ]
    divergentes = [row for row in unique_rows if city_mismatch(row)]

    write_csv(output_dir / "raw_unico_por_id.csv", fieldnames, unique_rows)
    write_csv(output_dir / "sem_preco_raw.csv", fieldnames, sem_preco)
    write_csv(output_dir / "pendentes_area_raw.csv", fieldnames, pendentes_area)
    write_csv(output_dir / "cidade_divergente_raw.csv", fieldnames, divergentes)

    rng = random.Random(args.seed)
    selected: dict[str, dict[str, Any]] = {}

    # Todos os casos sem preco entram no piloto: a pagina individual pode mostrar
    # se o valor estava apenas ausente no cartao da listagem.
    for row in sem_preco:
        add_reason(selected, row, "sem_preco_na_listagem", priority=0)

    # Os raros casos sem bairro tambem sao uteis para verificar a pagina individual.
    for row in unique_rows:
        if is_blank(row.get("bairro_raw")):
            add_reason(selected, row, "sem_bairro_na_listagem", priority=1)

    # Amostra sem area por cidade/tipo. Grupos com >=50% de ausencia recebem
    # amostra maior, pois sao os mais importantes para a decisao de enriquecimento.
    by_group = grouped(unique_rows)
    for (city, property_type), group_rows in sorted(by_group.items()):
        missing = [row for row in group_rows if is_blank(row.get("area_raw"))]
        if not missing:
            continue
        rate = len(missing) / len(group_rows)
        sample_size = args.sem_area_alta if rate >= 0.50 else args.sem_area_base
        rng.shuffle(missing)
        for row in missing[:sample_size]:
            add_reason(
                selected,
                row,
                f"sem_area:{city}/{property_type}",
                priority=2,
            )

    # Pequena amostra de anuncios completos para confirmar que o parser individual
    # encontra os mesmos campos quando eles ja existiam na listagem.
    for (city, property_type), group_rows in sorted(by_group.items()):
        controls = [
            row
            for row in group_rows
            if not is_blank(row.get("preco_raw")) and not is_blank(row.get("area_raw"))
        ]
        rng.shuffle(controls)
        for row in controls[: args.controles_por_grupo]:
            add_reason(
                selected,
                row,
                f"controle_completo:{city}/{property_type}",
                priority=4,
            )

    # Alguns conflitos de cidade ajudam a verificar se a pagina individual confirma
    # a localizacao retornada no dado estruturado da listagem.
    rng.shuffle(divergentes)
    for row in divergentes[: args.divergentes]:
        add_reason(selected, row, "cidade_divergente", priority=3)

    queue_entries = sorted(
        selected.values(),
        key=lambda item: (
            item["priority"],
            item["row"].get("cidade_consultada", ""),
            item["row"].get("tipo_consultado", ""),
            item["row"].get("id_anuncio", ""),
        ),
    )

    if args.max_total > 0:
        queue_entries = queue_entries[: args.max_total]

    queue_rows: list[dict[str, Any]] = []
    for item in queue_entries:
        row = dict(item["row"])
        row["motivo_amostra"] = "|".join(sorted(item["reasons"]))
        queue_rows.append(row)

    queue_fields = list(fieldnames)
    if "motivo_amostra" not in queue_fields:
        queue_fields.append("motivo_amostra")
    write_csv(output_dir / "fila_piloto_individual.csv", queue_fields, queue_rows)

    report_lines = [
        "TRIAGEM E FILA DO PILOTO INDIVIDUAL",
        "=" * 72,
        f"Arquivo raw: {raw_path.resolve()}",
        f"Observacoes raw: {len(raw_rows)}",
        f"IDs/linhas unicas apos consolidacao: {len(unique_rows)}",
        f"Sem preco na listagem: {len(sem_preco)}",
        f"Sem area e com preco: {len(pendentes_area)}",
        f"Cidade divergente: {len(divergentes)}",
        f"Paginas individuais selecionadas: {len(queue_rows)}",
        "",
        "AMOSTRA POR MOTIVO",
        "-" * 72,
    ]
    reason_counts: dict[str, int] = defaultdict(int)
    for row in queue_rows:
        for reason in str(row.get("motivo_amostra") or "").split("|"):
            if reason:
                reason_counts[reason] += 1
    for reason, count in sorted(reason_counts.items()):
        report_lines.append(f"{reason}: {count}")

    (output_dir / "resumo_fila_piloto.txt").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    print("\n".join(report_lines))
    print(f"\nFila: {(output_dir / 'fila_piloto_individual.csv').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
