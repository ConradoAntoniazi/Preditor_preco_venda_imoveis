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

BLOCK_MARKERS_STRONG = (
    "sorry, you have been blocked",
    "you have been blocked",
    "request unsuccessful. incapsula incident id",
    "verifique que voce e humano",
    "verifique que você é humano",
    "confirme que voce e humano",
    "confirme que você é humano",
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
                    ]
                },
            )
        except Exception as error:
            print(f"Aviso: bloqueio de recursos nao foi aplicado: {error}")
    return driver


def visible_body_text(driver: Any) -> str:
    try:
        return str(
            driver.execute_script(
                "return ((document.body && document.body.innerText) || '');"
            )
            or ""
        )
    except Exception:
        return ""


def page_ready(driver: Any) -> bool:
    """Espera conteudo do anuncio, sem considerar JSON-LD generico suficiente."""
    try:
        return bool(
            driver.execute_script(
                r"""
                const body = ((document.body && document.body.innerText) || '');
                const lowerBody = body.toLowerCase();
                const scripts = Array.from(document.scripts || [])
                    .map(s => s.textContent || '')
                    .join('\n');

                const hasStructuredPropertyData =
                    scripts.includes('real_estate_type') ||
                    scripts.includes('garage_spaces') ||
                    scripts.includes('re_features') ||
                    scripts.includes('re_complex_features') ||
                    scripts.includes('"properties"') ||
                    scripts.includes('"bathrooms"') ||
                    scripts.includes('"rooms"') ||
                    scripts.includes('"size"');

                const hasAdSections =
                    lowerBody.includes('descrição') ||
                    lowerBody.includes('descricao') ||
                    lowerBody.includes('características') ||
                    lowerBody.includes('caracteristicas') ||
                    lowerBody.includes('detalhes do imóvel') ||
                    lowerBody.includes('detalhes do imovel') ||
                    lowerBody.includes('localização') ||
                    lowerBody.includes('localizacao');

                const hasPriceOrPropertyWords =
                    /r\$\s*[\d.]+/.test(lowerBody) ||
                    lowerBody.includes('quarto') ||
                    lowerBody.includes('banheiro') ||
                    lowerBody.includes('m²');

                return body.length > 300 &&
                       (hasStructuredPropertyData || (hasAdSections && hasPriceOrPropertyWords));
                """
            )
        )
    except Exception:
        return False


def block_reason(driver: Any, html: str = "") -> str:
    """Detecta bloqueio no titulo/texto visivel, evitando falso positivo em bundles JS."""
    try:
        title = normalize_text(driver.title)
    except Exception:
        title = ""
    try:
        current_url = normalize_text(driver.current_url)
    except Exception:
        current_url = ""

    body = normalize_text(visible_body_text(driver))[:8000]
    body_start = body[:1200]

    for marker in BLOCK_MARKERS_STRONG:
        normalized_marker = normalize_text(marker)
        if normalized_marker in title or normalized_marker in body_start:
            return f"marcador visivel: {marker}"

    # "Access denied" e termos de desafio sao comuns em codigo-fonte; so contam
    # quando aparecem no titulo ou no inicio do texto que o usuario realmente ve.
    if "access denied" in title or body_start.startswith("access denied"):
        return "access denied visivel"
    if "captcha" in title or "captcha" in current_url:
        return "pagina de captcha"
    if "incapsula" in body_start and "incident" in body_start:
        return "desafio/incidente Incapsula visivel"
    return ""


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


def fetch_page_once(
    driver: Any,
    url: str,
    marker_timeout: int,
    settle_seconds: float,
) -> tuple[str, bool]:
    from selenium.common.exceptions import TimeoutException, WebDriverException
    from selenium.webdriver.support.ui import WebDriverWait

    try:
        driver.get(url)
    except TimeoutException:
        print("Timeout de carregamento; aguardando os marcadores disponiveis.")

    ready_timed_out = False
    try:
        WebDriverWait(driver, marker_timeout, poll_frequency=0.25).until(
            lambda current: bool(block_reason(current)) or page_ready(current)
        )
    except TimeoutException:
        ready_timed_out = True
        print("Marcadores completos nao apareceram dentro do tempo; analisando o HTML disponivel.")

    reason = block_reason(driver)
    if reason:
        raise BlockedError(reason)

    if settle_seconds > 0:
        time.sleep(settle_seconds)

    try:
        driver.execute_script("window.stop();")
    except WebDriverException:
        pass

    html = driver.page_source
    reason = block_reason(driver, html)
    if reason:
        raise BlockedError(reason)
    if len(html) <= 500:
        raise RuntimeError("HTML vazio ou muito curto")
    return html, ready_timed_out


def successful_ids(result_path: Path) -> set[str]:
    """Somente sucesso e considerado concluido; erros/bloqueios serao tentados de novo."""
    if not result_path.exists() or result_path.stat().st_size == 0:
        return set()
    with result_path.open("r", encoding="utf-8-sig", newline="") as file:
        return {
            str(row.get("id_anuncio") or "").strip()
            for row in csv.DictReader(file)
            if row.get("status") == "ok" and str(row.get("id_anuncio") or "").strip()
        }


def consolidate_latest_results(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Uma linha por anuncio; sucesso substitui falha anterior, senao vale a mais recente."""
    best: dict[str, dict[str, str]] = {}
    order: list[str] = []
    for index, row in enumerate(rows):
        ad_id = str(row.get("id_anuncio") or f"__sem_id_{index}").strip()
        current = best.get(ad_id)
        if current is None:
            best[ad_id] = row
            order.append(ad_id)
            continue
        if row.get("status") == "ok" or current.get("status") != "ok":
            best[ad_id] = row
    return [best[ad_id] for ad_id in order]


def write_report(result_path: Path, long_path: Path, output_path: Path) -> None:
    _, all_results = read_csv(result_path) if result_path.exists() else ([], [])
    results = consolidate_latest_results(all_results)
    _, long_rows = read_csv(long_path) if long_path.exists() else ([], [])

    status_counts = Counter(row.get("status") or "sem_status" for row in results)
    ok = [row for row in results if row.get("status") == "ok"]
    failures = [row for row in results if row.get("status") != "ok"]
    missing_area = [row for row in ok if is_blank(row.get("area_raw_listagem"))]
    recovered_area = [row for row in missing_area if not is_blank(row.get("area_individual"))]
    missing_price = [row for row in ok if is_blank(row.get("preco_raw_listagem"))]
    recovered_price = [row for row in missing_price if not is_blank(row.get("preco_individual"))]

    lines = [
        "RELATORIO DO PILOTO DE PAGINAS INDIVIDUAIS",
        "=" * 72,
        f"Anuncios unicos processados: {len(results)}",
        f"Status OK: {len(ok)}",
        f"Falhas/bloqueios atuais: {len(failures)}",
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
        "STATUS",
        "-" * 72,
    ]
    for status, count in sorted(status_counts.items()):
        lines.append(f"{status}: {count}")

    lines += ["", "RECUPERACAO DE AREA POR CIDADE/TIPO", "-" * 72]
    by_group: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in missing_area:
        by_group[(row.get("cidade_consultada", ""), row.get("tipo_consultado", ""))].append(row)
    if not by_group:
        lines.append("(ainda sem amostra suficiente)")
    for (city, property_type), rows in sorted(by_group.items()):
        recovered = sum(not is_blank(row.get("area_individual")) for row in rows)
        lines.append(f"{city} | {property_type}: {recovered}/{len(rows)}")

    lines += ["", "FALHAS/BLOQUEIOS", "-" * 72]
    if not failures:
        lines.append("(nenhum)")
    else:
        for row in failures[:30]:
            observation = re.sub(r"\s+", " ", str(row.get("observacao") or "")).strip()
            lines.append(
                f"{row.get('id_anuncio', '')} | {row.get('status', '')} | "
                f"tentativas={row.get('tentativas', '')} | {observation[:500]}"
            )

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
    parser.add_argument("--min-sleep", type=float, default=12.0)
    parser.add_argument("--max-sleep", type=float, default=18.0)
    parser.add_argument("--page-timeout", type=int, default=25)
    parser.add_argument("--marker-timeout", type=int, default=15)
    parser.add_argument("--settle-seconds", type=float, default=3.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--restart-driver-every", type=int, default=15)
    parser.add_argument("--block-retry-min", type=float, default=25.0)
    parser.add_argument("--block-retry-max", type=float, default=40.0)
    parser.add_argument("--no-block-resources", action="store_true")
    args = parser.parse_args()

    if args.min_sleep < 0 or args.max_sleep < args.min_sleep:
        parser.error("Intervalo de sleep invalido.")
    if args.block_retry_min < 0 or args.block_retry_max < args.block_retry_min:
        parser.error("Intervalo de espera apos suspeita de bloqueio invalido.")
    if args.settle_seconds < 0:
        parser.error("--settle-seconds deve ser nao negativo.")

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

    done = successful_ids(result_path)
    driver = make_driver(
        headless=args.headless,
        page_timeout=args.page_timeout,
        block_resources=not args.no_block_resources,
    )
    processed_since_restart = 0
    confirmed_block = False

    def restart_driver(current_driver: Any):
        try:
            current_driver.quit()
        except Exception:
            pass
        return make_driver(
            headless=args.headless,
            page_timeout=args.page_timeout,
            block_resources=not args.no_block_resources,
        )

    try:
        for index, row in enumerate(queue, start=1):
            ad_id = str(row.get("id_anuncio") or "").strip()
            url = str(row.get("url_anuncio") or "").strip()
            if not ad_id or not url or ad_id in done:
                continue

            if args.restart_driver_every > 0 and processed_since_restart >= args.restart_driver_every:
                driver = restart_driver(driver)
                processed_since_restart = 0

            print(f"\n[{index}/{len(queue)}] anuncio {ad_id}")
            status = "erro"
            attempts = 0
            observation = ""
            records: list[dict[str, str]] = []
            sources: set[str] = set()
            visible_text = ""
            html = ""

            for attempt in range(1, args.retries + 1):
                attempts = attempt
                print(f"Tentativa {attempt}/{args.retries}: {url}")
                try:
                    html, ready_timed_out = fetch_page_once(
                        driver,
                        url,
                        marker_timeout=args.marker_timeout,
                        settle_seconds=args.settle_seconds,
                    )
                    soup = BeautifulSoup(html, "html.parser")
                    attempt_records: list[dict[str, str]] = []
                    attempt_sources = parse_json_scripts(soup, attempt_records)
                    attempt_visible_text = parse_dom_text(soup, attempt_records)
                    attempt_records = deduplicate_records(attempt_records)

                    # Uma pagina individual valida deve trazer ao menos algum dado
                    # estruturado ou marcadores DOM de um anuncio.
                    if not attempt_records and len(attempt_visible_text) < 300:
                        raise RuntimeError("Pagina sem conteudo de anuncio reconhecivel")

                    records = attempt_records
                    sources = attempt_sources
                    visible_text = attempt_visible_text
                    status = "ok"
                    if ready_timed_out:
                        observation = "Marcadores completos excederam o timeout; HTML parcial analisado."
                    break

                except BlockedError as error:
                    observation = f"Suspeita de bloqueio: {error}"
                    try:
                        current_html = driver.page_source
                        debug_path = save_debug_html(
                            debug_dir, ad_id, f"bloqueio_tentativa_{attempt}", current_html
                        )
                        observation += f" HTML: {debug_path}"
                    except Exception:
                        pass

                    if attempt < args.retries:
                        delay = random.uniform(args.block_retry_min, args.block_retry_max)
                        print(
                            "Suspeita de bloqueio. Reiniciando o navegador e "
                            f"aguardando {delay:.1f}s antes de confirmar."
                        )
                        driver = restart_driver(driver)
                        processed_since_restart = 0
                        time.sleep(delay)
                        continue

                    status = "bloqueado_confirmado"
                    confirmed_block = True

                except Exception as error:
                    observation = str(error)
                    try:
                        current_html = driver.page_source
                        debug_path = save_debug_html(
                            debug_dir, ad_id, f"erro_tentativa_{attempt}", current_html
                        )
                        observation += f" HTML: {debug_path}"
                    except Exception:
                        pass

                    if attempt < args.retries:
                        delay = 5 * attempt + random.uniform(1.0, 3.0)
                        print(
                            f"Falha na tentativa {attempt}: {error}. "
                            f"Reiniciando o navegador e aguardando {delay:.1f}s."
                        )
                        driver = restart_driver(driver)
                        processed_since_restart = 0
                        time.sleep(delay)
                        continue
                    status = "erro"

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
                status == "ok"
                and is_blank(row.get("area_raw"))
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

            if status == "ok":
                done.add(ad_id)
            processed_since_restart += 1

            if confirmed_block:
                print(
                    "Bloqueio confirmado em duas tentativas com navegador novo. "
                    "Interrompendo o piloto para nao insistir na plataforma."
                )
                break

            if status != "ok":
                driver = restart_driver(driver)
                processed_since_restart = 0

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
