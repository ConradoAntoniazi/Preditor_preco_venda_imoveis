from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from bs4 import BeautifulSoup


RESULT_COLUMNS = [
    "data_utc",
    "id_anuncio",
    "url_anuncio",
    "cidade_consultada",
    "tipo_consultado",
    "motivo_amostra",
    "status",
    "tentativas",
    "preco_raw_listagem",
    "area_raw_listagem",
    "bairro_raw_listagem",
    "preco_individual",
    "area_individual",
    "bairro_individual",
    "cidade_individual",
    "quartos_individual",
    "banheiros_individual",
    "vagas_individual",
    "condominio_individual",
    "iptu_individual",
    "nomes_propriedades_json",
    "termos_interesse_json",
    "fontes_encontradas_json",
    "observacao",
]

LONG_COLUMNS = ["id_anuncio", "source", "path", "key", "value"]
LOG_COLUMNS = [
    "data_utc",
    "id_anuncio",
    "url_anuncio",
    "status",
    "tentativas",
    "campos_estruturados",
    "area_recuperada",
    "observacao",
]

BLOCK_MARKERS = (
    "sorry, you have been blocked",
    "you have been blocked",
    "access denied",
    "request unsuccessful. incapsula incident id",
)

INTERESTING_TERMS = {
    "suite": ["suite", "suíte"],
    "piscina": ["piscina"],
    "varanda": ["varanda", "sacada"],
    "lazer": ["lazer", "academia", "playground", "salao de festas", "salão de festas", "clube"],
    "mobiliado": ["mobiliado", "mobiliada", "mobilia", "mobília"],
    "elevador": ["elevador"],
}

KEY_ALIASES = {
    "preco": ["pricevalue", "price", "preco", "valorvenda", "offersprice"],
    "area": ["size", "area", "areautil", "areaconstruida", "areatotal", "areaprivativa", "floorsize"],
    "bairro": ["neighbourhood", "neighborhood", "bairro", "addressdistrict"],
    "cidade": ["municipality", "cidade", "addresslocality", "city"],
    "quartos": ["rooms", "quartos", "bedrooms", "numberofrooms"],
    "banheiros": ["bathrooms", "banheiros", "numberofbathrooms"],
    "vagas": ["garagespaces", "garage", "vagas", "parking"],
    "condominio": ["condominio", "condominium", "hoa"],
    "iptu": ["iptu", "propertytax"],
}


class BlockedError(RuntimeError):
    pass


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def is_blank(value: Any) -> bool:
    return value is None or str(value).strip() == ""


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "").lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", text).strip()


def normalize_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize_text(value))


def compact_value(value: Any, limit: int = 600) -> str:
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    else:
        text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader.fieldnames or []), list(reader)


def append_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def save_debug_html(debug_dir: Path, ad_id: str, reason: str, html: str) -> Path:
    debug_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = debug_dir / f"{timestamp}_{ad_id}_{reason}.html"
    path.write_text(html, encoding="utf-8")
    return path


def make_driver(headless: bool, page_timeout: int, block_resources: bool = True):
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
    except ImportError as error:
        raise RuntimeError("Selenium nao instalado.") from error

    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.page_load_strategy = "eager"
    options.add_argument("--window-size=1440,1000")
    options.add_argument("--lang=pt-BR")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-notifications")
    options.add_argument("--mute-audio")
    options.add_argument("--disable-background-networking")

    if block_resources:
        options.add_experimental_option(
            "prefs",
            {
                "profile.managed_default_content_settings.images": 2,
                "profile.default_content_setting_values.notifications": 2,
                "profile.default_content_setting_values.geolocation": 2,
            },
        )

    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(page_timeout)

    if block_resources:
        try:
            driver.execute_cdp_cmd("Network.enable", {})
            driver.execute_cdp_cmd(
                "Network.setBlockedURLs",
                {
                    "urls": [
                        "*.png", "*.jpg", "*.jpeg", "*.gif", "*.webp", "*.ico",
                        "*.woff", "*.woff2", "*.ttf", "*.mp4", "*.webm",
                        "*doubleclick.net/*", "*googletagmanager.com/*",
                        "*google-analytics.com/*", "*taboola.com/*", "*criteo.com/*",
                        "*rubiconproject.com/*", "*scorecardresearch.com/*",
                        "*mouseflow.com/*", "*navdmp.com/*", "*datadoghq.com/*",
                    ]
                },
            )
        except Exception as error:
            print(f"Aviso: bloqueio de recursos nao foi aplicado: {error}")
    return driver


def page_ready(driver: Any) -> bool:
    try:
        return bool(
            driver.execute_script(
                """
                const scripts = document.scripts || [];
                let structured = false;
                for (let i = 0; i < scripts.length; i++) {
                    const s = scripts[i];
                    const text = s.textContent || '';
                    if (
                        s.id === '__NEXT_DATA__' ||
                        s.type === 'application/ld+json' ||
                        text.includes('self.__next_f.push') ||
                        text.includes('real_estate_type') ||
                        text.includes('garage_spaces')
                    ) {
                        structured = true;
                        break;
                    }
                }
                const body = ((document.body && document.body.innerText) || '');
                return structured && body.length > 200;
                """
            )
        )
    except Exception:
        return False


def is_blocked(html: str) -> bool:
    lower = html.lower()
    return any(marker in lower for marker in BLOCK_MARKERS)


def decode_next_f_push(script_text: str) -> Optional[list[Any]]:
    text = script_text.strip()
    prefix = "self.__next_f.push("
    if not text.startswith(prefix):
        return None
    body = text[len(prefix):].strip()
    if body.endswith(");"):
        body = body[:-2]
    elif body.endswith(")"):
        body = body[:-1]
    try:
        value = json.loads(body)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, list) else None


def add_record(
    records: list[dict[str, str]], source: str, path: str, key: str, value: Any
) -> None:
    if value is None or isinstance(value, (dict, list)) and not value:
        return
    records.append(
        {
            "source": source,
            "path": path,
            "key": str(key),
            "value": compact_value(value),
        }
    )


def walk_json(
    value: Any,
    source: str,
    records: list[dict[str, str]],
    path: str = "$",
    depth: int = 0,
) -> None:
    if depth > 30:
        return
    if isinstance(value, dict):
        # Pares name/value sao comuns nas propriedades estruturadas da OLX.
        if "name" in value and "value" in value and not isinstance(value.get("name"), (dict, list)):
            add_record(records, source, path, value.get("name"), value.get("value"))
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if isinstance(child, (dict, list)):
                walk_json(child, source, records, child_path, depth + 1)
            elif child is not None:
                add_record(records, source, child_path, key, child)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            walk_json(child, source, records, f"{path}[{index}]", depth + 1)


def parse_json_scripts(soup: BeautifulSoup, records: list[dict[str, str]]) -> set[str]:
    sources: set[str] = set()
    for index, script in enumerate(soup.find_all("script")):
        text = script.string or script.get_text() or ""
        script_type = (script.get("type") or "").lower()

        if script_type == "application/ld+json":
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                continue
            sources.add("json_ld")
            walk_json(data, "json_ld", records, f"$jsonld[{index}]")

        if script.get("id") == "__NEXT_DATA__":
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                continue
            sources.add("next_data")
            walk_json(data, "next_data", records, "$next_data")

        pushed = decode_next_f_push(text)
        if pushed and len(pushed) >= 2 and isinstance(pushed[1], str):
            sources.add("react_flight")
            parse_react_flight_chunk(pushed[1], records, index)
    return sources


def parse_react_flight_chunk(
    chunk: str, records: list[dict[str, str]], script_index: int
) -> None:
    decoder = json.JSONDecoder()
    starts = set()
    for pattern in (r'\{"name"\s*:', r'\{"properties"\s*:', r'\{"locationDetails"\s*:'):
        starts.update(match.start() for match in re.finditer(pattern, chunk))

    for start in sorted(starts):
        try:
            value, _ = decoder.raw_decode(chunk[start:])
        except json.JSONDecodeError:
            continue
        walk_json(value, "react_flight", records, f"$flight[{script_index}]@{start}")

    # Captura direta de chaves relevantes mesmo quando o trecho Flight nao forma
    # um objeto JSON completo no ponto encontrado.
    aliases = sorted({alias for values in KEY_ALIASES.values() for alias in values})
    key_pattern = re.compile(r'"([^"\\]+)"\s*:\s*')
    for match in key_pattern.finditer(chunk):
        raw_key = match.group(1)
        norm = normalize_key(raw_key)
        if not any(alias in norm for alias in aliases):
            continue
        try:
            value, _ = decoder.raw_decode(chunk[match.end():])
        except json.JSONDecodeError:
            continue
        add_record(
            records,
            "react_flight",
            f"$flight[{script_index}]@{match.start()}.{raw_key}",
            raw_key,
            value,
        )


def parse_dom_text(soup: BeautifulSoup, records: list[dict[str, str]]) -> str:
    text = soup.get_text(" ", strip=True)
    patterns = {
        "area_dom": [
            r"(?:area\s+(?:util|construida|total|privativa)|área\s+(?:útil|construída|total|privativa))\s*[:\-]?\s*([\d.,]+)\s*m²",
            r"\b([\d.,]+)\s*m²\b",
        ],
        "quartos_dom": [r"\bquartos?\s*[:\-]?\s*(\d+\+?)", r"\b(\d+\+?)\s+quartos?\b"],
        "banheiros_dom": [r"\bbanheiros?\s*[:\-]?\s*(\d+\+?)", r"\b(\d+\+?)\s+banheiros?\b"],
        "vagas_dom": [r"\bvagas?(?:\s+na\s+garagem)?\s*[:\-]?\s*(\d+\+?)", r"\b(\d+\+?)\s+vagas?\b"],
        "condominio_dom": [r"condom[ií]nio\s*[:\-]?\s*(R\$\s*[\d.,]+(?:/m[eê]s)?)"],
        "iptu_dom": [r"\bIPTU\s*[:\-]?\s*(R\$\s*[\d.,]+)"],
        "preco_dom": [r"\bR\$\s*[\d.]+(?:,\d{2})?\b"],
    }
    for key, regexes in patterns.items():
        for regex in regexes:
            match = re.search(regex, text, flags=re.IGNORECASE)
            if match:
                add_record(records, "dom_text", f"$dom.{key}", key, match.group(1) if match.groups() else match.group(0))
                break
    return text


def deduplicate_records(records: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for record in records:
        key = (record["source"], record["path"], record["key"], record["value"])
        if key not in seen:
            seen.add(key)
            result.append(record)
    return result


def candidate_value(records: list[dict[str, str]], field: str) -> str:
    aliases = KEY_ALIASES[field]
    ranked: list[tuple[int, str]] = []
    for record in records:
        key_norm = normalize_key(record["key"])
        path_norm = normalize_key(record["path"])
        score = 0
        for alias in aliases:
            if key_norm == alias:
                score = max(score, 4)
            elif alias in key_norm:
                score = max(score, 3)
            elif alias in path_norm:
                score = max(score, 2)
        if record["source"] == "dom_text" and field in key_norm:
            score = max(score, 1)
        if score:
            ranked.append((score, record["value"]))
    if not ranked:
        return ""
    ranked.sort(key=lambda item: (-item[0], len(item[1])))
    return ranked[0][1]


def feature_mentions(text: str, records: list[dict[str, str]]) -> dict[str, dict[str, list[str]]]:
    combined = text + " " + " ".join(record["value"] for record in records)
    normalized = normalize_text(combined)
    result: dict[str, dict[str, list[str]]] = {}
    for feature, terms in INTERESTING_TERMS.items():
        mentions: set[str] = set()
        negations: set[str] = set()
        for term in terms:
            normalized_term = normalize_text(term)
            if normalized_term not in normalized:
                continue
            mentions.add(normalized_term)
            for pattern in (
                f"sem {normalized_term}",
                f"nao possui {normalized_term}",
                f"nao tem {normalized_term}",
                f"nao conta com {normalized_term}",
            ):
                if pattern in normalized:
                    negations.add(pattern)
        if mentions:
            result[feature] = {
                "mencoes": sorted(mentions),
                "negacoes": sorted(negations),
            }
    return result


def property_names(records: list[dict[str, str]]) -> list[str]:
    names: set[str] = set()
    ignored = {"name", "value", "id", "type"}
    for record in records:
        key = record["key"].strip()
        if (
            key
            and normalize_key(key) not in ignored
            and len(key) <= 80
            and record["source"] in {"react_flight", "next_data"}
        ):
            names.add(key)
    return sorted(names, key=lambda value: normalize_text(value))


def fetch_html(driver: Any, url: str, marker_timeout: int, retries: int) -> tuple[str, int]:
    from selenium.common.exceptions import TimeoutException, WebDriverException
    from selenium.webdriver.support.ui import WebDriverWait

    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        print(f"Tentativa {attempt}/{retries}: {url}")
        try:
            driver.get(url)
        except TimeoutException:
            print("Timeout de carregamento; analisando o HTML disponivel.")
        except WebDriverException as error:
            last_error = error
            if attempt < retries:
                time.sleep(4 * attempt + random.uniform(0.5, 1.5))
                continue
            raise RuntimeError(f"Falha do navegador: {error}") from error

        try:
            WebDriverWait(driver, marker_timeout, poll_frequency=0.25).until(page_ready)
        except TimeoutException:
            pass
        try:
            driver.execute_script("window.stop();")
        except WebDriverException:
            pass
        html = driver.page_source
        if is_blocked(html):
            raise BlockedError("Bloqueio detectado pela pagina.")
        if len(html) > 500:
            return html, attempt
        last_error = RuntimeError("HTML vazio ou muito curto")
    raise RuntimeError(f"Pagina nao carregada: {last_error}")


def existing_ids(result_path: Path) -> set[str]:
    if not result_path.exists() or result_path.stat().st_size == 0:
        return set()
    with result_path.open("r", encoding="utf-8-sig", newline="") as file:
        return {
            str(row.get("id_anuncio") or "").strip()
            for row in csv.DictReader(file)
            if str(row.get("id_anuncio") or "").strip()
        }


def write_report(result_path: Path, long_path: Path, output_path: Path) -> None:
    _, results = read_csv(result_path) if result_path.exists() else ([], [])
    _, long_rows = read_csv(long_path) if long_path.exists() else ([], [])

    ok = [row for row in results if row.get("status") == "ok"]
    missing_area = [row for row in ok if is_blank(row.get("area_raw_listagem"))]
    recovered_area = [row for row in missing_area if not is_blank(row.get("area_individual"))]
    missing_price = [row for row in ok if is_blank(row.get("preco_raw_listagem"))]
    recovered_price = [row for row in missing_price if not is_blank(row.get("preco_individual"))]

    lines = [
        "RELATORIO DO PILOTO DE PAGINAS INDIVIDUAIS",
        "=" * 72,
        f"Paginas processadas: {len(results)}",
        f"Status OK: {len(ok)}",
        f"Falhas/bloqueios: {len(results) - len(ok)}",
        f"Sem area na listagem dentro do piloto: {len(missing_area)}",
        f"Area recuperada na pagina individual: {len(recovered_area)}",
        (
            f"Taxa de recuperacao de area: {len(recovered_area) / len(missing_area) * 100:.2f}%"
            if missing_area
            else "Taxa de recuperacao de area: n/a"
        ),
        f"Sem preco na listagem dentro do piloto: {len(missing_price)}",
        f"Preco recuperado na pagina individual: {len(recovered_price)}",
        "",
        "RECUPERACAO DE AREA POR CIDADE/TIPO",
        "-" * 72,
    ]

    by_group: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in missing_area:
        by_group[(row.get("cidade_consultada", ""), row.get("tipo_consultado", ""))].append(row)
    for (city, property_type), rows in sorted(by_group.items()):
        recovered = sum(not is_blank(row.get("area_individual")) for row in rows)
        lines.append(f"{city} | {property_type}: {recovered}/{len(rows)}")

    lines += ["", "CHAVES MAIS FREQUENTES", "-" * 72]
    key_counts = Counter(row.get("key", "") for row in long_rows if row.get("key"))
    for key, count in key_counts.most_common(40):
        lines.append(f"{key}: {count}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Explora uma amostra de paginas individuais da OLX para medir a "
            "recuperacao de area e inventariar campos estruturados."
        )
    )
    parser.add_argument("--queue", required=True)
    parser.add_argument("--output-dir", default="piloto_paginas_individuais/resultado")
    parser.add_argument("--max-items", type=int, default=60)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--min-sleep", type=float, default=6.0)
    parser.add_argument("--max-sleep", type=float, default=9.0)
    parser.add_argument("--page-timeout", type=int, default=20)
    parser.add_argument("--marker-timeout", type=int, default=10)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--restart-driver-every", type=int, default=25)
    parser.add_argument("--no-block-resources", action="store_true")
    args = parser.parse_args()

    if args.min_sleep < 0 or args.max_sleep < args.min_sleep:
        parser.error("Intervalo de sleep invalido.")

    _, queue = read_csv(Path(args.queue))
    if args.max_items > 0:
        queue = queue[: args.max_items]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "resultado_piloto_individual.csv"
    long_path = output_dir / "campos_individuais_long.csv"
    log_path = output_dir / "log_piloto_individual.csv"
    report_path = output_dir / "relatorio_piloto_individual.txt"
    debug_dir = output_dir / "debug_html"

    done = existing_ids(result_path)
    driver = make_driver(
        headless=args.headless,
        page_timeout=args.page_timeout,
        block_resources=not args.no_block_resources,
    )
    processed_since_restart = 0

    try:
        for index, row in enumerate(queue, start=1):
            ad_id = str(row.get("id_anuncio") or "").strip()
            url = str(row.get("url_anuncio") or "").strip()
            if not ad_id or not url or ad_id in done:
                continue

            if args.restart_driver_every > 0 and processed_since_restart >= args.restart_driver_every:
                driver.quit()
                driver = make_driver(
                    headless=args.headless,
                    page_timeout=args.page_timeout,
                    block_resources=not args.no_block_resources,
                )
                processed_since_restart = 0

            print(f"\n[{index}/{len(queue)}] anuncio {ad_id}")
            status = "ok"
            attempts = 0
            observation = ""
            records: list[dict[str, str]] = []
            sources: set[str] = set()
            visible_text = ""

            try:
                html, attempts = fetch_html(
                    driver, url, args.marker_timeout, args.retries
                )
                soup = BeautifulSoup(html, "html.parser")
                sources = parse_json_scripts(soup, records)
                visible_text = parse_dom_text(soup, records)
                records = deduplicate_records(records)
            except BlockedError as error:
                status = "bloqueado"
                observation = str(error)
                html = driver.page_source
                debug_path = save_debug_html(debug_dir, ad_id, "bloqueio", html)
                observation += f" HTML: {debug_path}"
            except Exception as error:
                status = "erro"
                observation = str(error)
                try:
                    html = driver.page_source
                    debug_path = save_debug_html(debug_dir, ad_id, "erro", html)
                    observation += f" HTML: {debug_path}"
                except Exception:
                    pass

            names = property_names(records)
            mentions = feature_mentions(visible_text, records) if status == "ok" else {}
            result = {
                "data_utc": utc_now_iso(),
                "id_anuncio": ad_id,
                "url_anuncio": url,
                "cidade_consultada": row.get("cidade_consultada"),
                "tipo_consultado": row.get("tipo_consultado"),
                "motivo_amostra": row.get("motivo_amostra"),
                "status": status,
                "tentativas": attempts,
                "preco_raw_listagem": row.get("preco_raw"),
                "area_raw_listagem": row.get("area_raw"),
                "bairro_raw_listagem": row.get("bairro_raw"),
                "preco_individual": candidate_value(records, "preco"),
                "area_individual": candidate_value(records, "area"),
                "bairro_individual": candidate_value(records, "bairro"),
                "cidade_individual": candidate_value(records, "cidade"),
                "quartos_individual": candidate_value(records, "quartos"),
                "banheiros_individual": candidate_value(records, "banheiros"),
                "vagas_individual": candidate_value(records, "vagas"),
                "condominio_individual": candidate_value(records, "condominio"),
                "iptu_individual": candidate_value(records, "iptu"),
                "nomes_propriedades_json": json.dumps(names, ensure_ascii=False),
                "termos_interesse_json": json.dumps(mentions, ensure_ascii=False),
                "fontes_encontradas_json": json.dumps(sorted(sources), ensure_ascii=False),
                "observacao": observation,
            }
            append_csv(result_path, RESULT_COLUMNS, [result])

            long_rows = [
                {
                    "id_anuncio": ad_id,
                    "source": record["source"],
                    "path": record["path"],
                    "key": record["key"],
                    "value": record["value"],
                }
                for record in records
            ]
            append_csv(long_path, LONG_COLUMNS, long_rows)

            area_recovered = (
                is_blank(row.get("area_raw"))
                and not is_blank(result.get("area_individual"))
            )
            append_csv(
                log_path,
                LOG_COLUMNS,
                [
                    {
                        "data_utc": result["data_utc"],
                        "id_anuncio": ad_id,
                        "url_anuncio": url,
                        "status": status,
                        "tentativas": attempts,
                        "campos_estruturados": len(records),
                        "area_recuperada": area_recovered,
                        "observacao": observation,
                    }
                ],
            )
            done.add(ad_id)
            processed_since_restart += 1

            if status == "bloqueado":
                print("Bloqueio detectado. Interrompendo o piloto.")
                break

            if index < len(queue):
                delay = random.uniform(args.min_sleep, args.max_sleep)
                print(f"Aguardando {delay:.1f}s...")
                time.sleep(delay)
    except KeyboardInterrupt:
        print("\nPiloto interrompido pelo usuario. Resultados ja gravados foram preservados.")
    finally:
        try:
            driver.quit()
        except Exception:
            pass

    write_report(result_path, long_path, report_path)
    print(f"\nResultado: {result_path.resolve()}")
    print(f"Campos: {long_path.resolve()}")
    print(f"Relatorio: {report_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
