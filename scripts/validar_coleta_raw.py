from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


RAW_FILE = "olx_listagens_raw.csv"
LOG_FILE = "olx_log_paginas.csv"
CHECKPOINT_FILE = "checkpoint.json"

REQUIRED_RAW_COLUMNS = {
    "execucao_id",
    "data_coleta_utc",
    "cidade_consultada",
    "tipo_consultado",
    "pagina",
    "posicao_pagina",
    "url_listagem",
    "id_anuncio",
    "url_anuncio",
    "titulo_raw",
    "preco_raw",
    "categoria_raw",
    "cidade_raw",
    "area_raw",
    "quartos_raw",
    "banheiros_raw",
    "vagas_raw",
}

CRITICAL_VALUE_COLUMNS = [
    "execucao_id",
    "cidade_consultada",
    "tipo_consultado",
    "pagina",
    "id_anuncio",
    "url_anuncio",
    "titulo_raw",
    "preco_raw",
    "categoria_raw",
    "cidade_raw",
]

NORMAL_LOG_STATUSES = {
    "ok",
    "ok_ja_estava_salva",
    "fim_sem_anuncios",
    "fim_pagina_repetida",
    "fim_sem_ids_novos",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def is_blank(value: Any) -> bool:
    return value is None or str(value).strip() == ""


def pct(part: int, total: int) -> float:
    return 0.0 if total == 0 else part / total * 100


def int_or_none(value: Any) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def target_key(city: str, property_type: str) -> str:
    return f"{city}|{property_type}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Valida a estrutura e a cobertura de uma coleta raw da OLX."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Pasta contendo olx_listagens_raw.csv, olx_log_paginas.csv e checkpoint.json.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Arquivo TXT de saída. Padrão: <input-dir>/relatorio_validacao_raw.txt",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output = Path(args.output) if args.output else input_dir / "relatorio_validacao_raw.txt"

    raw = read_csv(input_dir / RAW_FILE)
    logs = read_csv(input_dir / LOG_FILE)
    checkpoint = read_json(input_dir / CHECKPOINT_FILE)

    errors: list[str] = []
    warnings: list[str] = []
    lines: list[str] = []

    lines.append("RELATÓRIO DE VALIDAÇÃO DA COLETA RAW")
    lines.append("=" * 72)
    lines.append(f"Pasta: {input_dir.resolve()}")
    lines.append(f"Execução: {checkpoint.get('execution_id', '(não informada)')}")
    lines.append(f"Linhas raw: {len(raw)}")
    lines.append(f"Linhas de log: {len(logs)}")

    if not raw:
        errors.append("O CSV raw está vazio.")
        raw_columns: set[str] = set()
    else:
        raw_columns = set(raw[0].keys())
        missing_columns = sorted(REQUIRED_RAW_COLUMNS - raw_columns)
        if missing_columns:
            errors.append("Colunas obrigatórias ausentes: " + ", ".join(missing_columns))

    # Campos críticos
    lines.append("")
    lines.append("AUSÊNCIAS")
    lines.append("-" * 72)
    for column in CRITICAL_VALUE_COLUMNS + ["bairro_raw", "area_raw", "banheiros_raw", "vagas_raw"]:
        if column not in raw_columns:
            continue
        missing = sum(is_blank(row.get(column)) for row in raw)
        lines.append(f"{column}: {missing} ({pct(missing, len(raw)):.2f}%)")
        if column in CRITICAL_VALUE_COLUMNS and missing:
            errors.append(f"{column} possui {missing} valor(es) ausente(s).")

    area_missing = sum(is_blank(row.get("area_raw")) for row in raw)
    if area_missing:
        warnings.append(
            f"{area_missing} anúncio(s) ({pct(area_missing, len(raw)):.2f}%) estão sem área; "
            "eles deverão ir para a fila de enriquecimento posterior."
        )

    # Duplicações técnicas
    observation_keys = [
        (
            row.get("cidade_consultada", ""),
            row.get("tipo_consultado", ""),
            row.get("pagina", ""),
            row.get("id_anuncio", ""),
        )
        for row in raw
    ]
    duplicate_observations = sum(count - 1 for count in Counter(observation_keys).values() if count > 1)
    if duplicate_observations:
        errors.append(f"Há {duplicate_observations} observação(ões) repetida(s) na mesma página.")

    ids = [row.get("id_anuncio", "") for row in raw if not is_blank(row.get("id_anuncio"))]
    repeated_ids = {ad_id: count for ad_id, count in Counter(ids).items() if count > 1}

    lines.append("")
    lines.append("DUPLICAÇÕES E POSIÇÕES")
    lines.append("-" * 72)
    lines.append(f"IDs únicos: {len(set(ids))}")
    lines.append(f"IDs que aparecem mais de uma vez no raw: {len(repeated_ids)}")
    lines.append(f"Observações duplicadas na mesma página: {duplicate_observations}")

    rows_by_page: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in raw:
        rows_by_page[(row.get("cidade_consultada", ""), row.get("tipo_consultado", ""), row.get("pagina", ""))].append(row)

    position_problems = 0
    for page_key, page_rows in sorted(rows_by_page.items()):
        positions = [int_or_none(row.get("posicao_pagina")) for row in page_rows]
        valid_positions = [position for position in positions if position is not None]
        if len(valid_positions) != len(page_rows) or len(valid_positions) != len(set(valid_positions)):
            position_problems += 1
            warnings.append(
                f"Posições ausentes ou repetidas em {page_key[0]} | {page_key[1]} | página {page_key[2]}."
            )
    lines.append(f"Páginas com problema de posição: {position_problems}")

    # Cobertura por alvo
    lines.append("")
    lines.append("COBERTURA POR CIDADE E TIPO")
    lines.append("-" * 72)

    completed_pages = set(checkpoint.get("completed_pages", []))
    target_meta = checkpoint.get("targets", {}) if isinstance(checkpoint.get("targets"), dict) else {}

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in raw:
        grouped[(row.get("cidade_consultada", ""), row.get("tipo_consultado", ""))].append(row)

    for (city, property_type), target_rows in sorted(grouped.items()):
        key = target_key(city, property_type)
        pages = sorted({int_or_none(row.get("pagina")) for row in target_rows if int_or_none(row.get("pagina")) is not None})
        planned = int_or_none(target_meta.get(key, {}).get("paginas_planejadas_nesta_execucao"))
        stop_reason = target_meta.get(key, {}).get("motivo_parada")
        missing_area_target = sum(is_blank(row.get("area_raw")) for row in target_rows)
        city_mismatch = sum(
            not is_blank(row.get("cidade_raw"))
            and row.get("cidade_raw", "").strip() != city
            for row in target_rows
        )
        lines.append(
            f"{city} | {property_type}: {len(target_rows)} linhas; "
            f"{len(pages)} página(s); área ausente {missing_area_target} "
            f"({pct(missing_area_target, len(target_rows)):.2f}%); "
            f"cidade divergente {city_mismatch}; planejadas {planned if planned is not None else '?'}"
            + (f"; parada={stop_reason}" if stop_reason else "")
        )

        completed_for_target = {
            value for value in completed_pages if value.startswith(f"{city}|{property_type}|")
        }
        if planned is not None and len(completed_for_target) < planned and not stop_reason:
            errors.append(
                f"{city} | {property_type}: somente {len(completed_for_target)} de {planned} páginas "
                "foram concluídas e não há motivo de parada registrado."
            )

    # Logs
    abnormal_logs = [row for row in logs if row.get("status") not in NORMAL_LOG_STATUSES]
    lines.append("")
    lines.append("LOG")
    lines.append("-" * 72)
    status_counts = Counter(row.get("status", "") for row in logs)
    for status, count in sorted(status_counts.items()):
        lines.append(f"{status or '(vazio)'}: {count}")
    if abnormal_logs:
        errors.append(
            "O log contém status que exigem revisão: "
            + ", ".join(f"{status}={count}" for status, count in sorted(Counter(row.get('status', '') for row in abnormal_logs).items()))
        )

    # Resultado
    lines.append("")
    lines.append("RESULTADO")
    lines.append("=" * 72)
    if errors:
        lines.append("REPROVADA PARA CONSOLIDAÇÃO: há erro(s) técnico(s) a corrigir.")
    else:
        lines.append("APROVADA TECNICAMENTE: a coleta pode seguir para triagem e enriquecimento.")

    if errors:
        lines.append("")
        lines.append("ERROS")
        for item in errors:
            lines.append(f"- {item}")

    if warnings:
        lines.append("")
        lines.append("AVISOS")
        for item in warnings:
            lines.append(f"- {item}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nRelatório salvo em: {output.resolve()}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
