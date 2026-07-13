from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
import unicodedata
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import urlsplit, urlunsplit

from bs4 import BeautifulSoup


RESULT_COLUMNS = [
    "data_utc",
    "id_anuncio",
    "url_anuncio",
    "url_requisitada",
    "cidade_raw_listagem",
    "tipo_consultado",
    "area_raw_listagem",
    "status",
    "list_id_extraido",
    "id_confere",
    "preco_individual",
    "cidade_individual",
    "bairro_individual",
    "area_estruturada_raw",
    "area_estruturada_m2",
    "areas_descricao_json",
    "area_descricao_m2",
    "area_final_m2",
    "area_fonte",
    "quartos_individual",
    "banheiros_individual",
    "vagas_individual",
    "categoria_individual",
    "tipo_detalhado_individual",
    "re_features_json",
    "re_complex_features_json",
    "re_types_raw",
    "descricao_tamanho",
    "descricao_trecho",
    "evidencias_flags_json",
    "observacao",
]

LOG_COLUMNS = [
    "data_utc",
    "id_anuncio",
    "url_requisitada",
    "status",
    "area_recuperada",
    "tempo_segundos",
    "observacao",
]

BLOCK_MARKERS = (
    "sorry, you have been blocked",
    "you are unable to access",
    "attention required! | cloudflare",
    "cf-error-details",
    "cloudflare ray id",
    "verifique que voce e humano",
    "verifique que você é humano",
    "confirme que voce e humano",
    "confirme que você é humano",
)

FEATURE_TERMS = {
    "tem_suite": ("suite", "suíte"),
    "tem_piscina": ("piscina",),
    "tem_varanda": ("varanda", "sacada"),
    "tem_lazer": (
        "lazer",
        "academia",
        "fitness",
        "playground",
        "salao de festas",
        "salão de festas",
        "brinquedoteca",
        "espaco brincar",
        "espaço brincar",
        "clube",
    ),
    "tem_mobiliado": ("mobiliado", "mobiliada", "porteira fechada"),
    "tem_elevador": ("elevador",),
}

NEGATION_PATTERNS = (
    "sem {term}",
    "nao tem {term}",
    "não tem {term}",
    "nao possui {term}",
    "não possui {term}",
    "nao conta com {term}",
    "não conta com {term}",
)


class BlockedError(RuntimeError):
    pass


class ParseError(RuntimeError):
    pass


@dataclass
class ParsedAd:
    list_id: str
    price: str
    city: str
    neighbourhood: str
    area_raw: str
    area_m2: Optional[float]
    area_description_candidates: list[float]
    area_description_m2: Optional[float]
    rooms: str
    bathrooms: str
    garage_spaces: str
    category: str
    real_estate_type: str
    re_features: list[str]
    re_complex_features: list[str]
    re_types: str
    description: str
    feature_evidence: dict[str, Any]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def is_blank(value: Any) -> bool:
    return value is None or str(value).strip() == ""


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "").lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", text).strip()



def redact_personal_contacts(text: str) -> str:
    text = re.sub(
        r"(?<!\d)(?:\+?55\s*)?(?:\(?\d{2}\)?[\s.-]*)?\d{4,5}[\s.-]*\d{4}(?!\d)",
        "[telefone removido]",
        text,
    )
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[email removido]", text)
    return text

def canonicalize_url(url: str) -> str:
    parts = urlsplit(str(url).strip())
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def append_csv(path: Path, columns: list[str], rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def successful_ids(result_path: Path) -> set[str]:
    if not result_path.exists() or result_path.stat().st_size == 0:
        return set()
    return {
        str(row.get("id_anuncio") or "").strip()
        for row in read_csv(result_path)
        if row.get("status") in {
            "area_recuperada",
            "sem_area_confirmado",
            "controle_ok",
            "ok_sem_area_na_listagem",
        }
        and str(row.get("id_anuncio") or "").strip()
    }


def save_debug_html(debug_dir: Path, ad_id: str, reason: str, html: str) -> Path:
    debug_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = debug_dir / f"{stamp}_{ad_id}_{reason}.html"
    path.write_text(html, encoding="utf-8")
    return path


def detect_block(html: str, title: str = "") -> str:
    sample = normalize_text(f"{title}\n{html[:10000]}")
    for marker in BLOCK_MARKERS:
        if normalize_text(marker) in sample:
            return marker
    return ""


def parse_number_pt(value: Any) -> Optional[float]:
    if is_blank(value):
        return None
    text = str(value).strip().replace("\xa0", " ")
    match = re.search(r"\d[\d.]*(?:,\d+)?", text)
    if not match:
        return None
    token = match.group(0)
    if "," in token:
        token = token.replace(".", "").replace(",", ".")
    elif re.fullmatch(r"\d{1,3}(?:\.\d{3})+", token):
        # Pontos como separadores de milhar: 1.000 -> 1000.
        token = token.replace(".", "")
    elif token.count(".") > 1:
        token = token.replace(".", "")
    try:
        return float(token)
    except ValueError:
        return None


def normalize_area(value: Any) -> Optional[float]:
    number = parse_number_pt(value)
    if number is None or number <= 0 or number > 100000:
        return None
    return number


def property_map(properties: Any) -> tuple[dict[str, str], dict[str, list[str]]]:
    values: dict[str, str] = {}
    labels: dict[str, list[str]] = {}
    if not isinstance(properties, list):
        return values, labels
    for item in properties:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        raw_value = item.get("value")
        values[name] = "" if raw_value is None else str(raw_value).strip()
        raw_labels = item.get("values") or []
        labels[name] = [
            str(entry.get("label")).strip()
            for entry in raw_labels
            if isinstance(entry, dict) and not is_blank(entry.get("label"))
        ]
    return values, labels


def html_description_to_text(value: Any) -> str:
    if is_blank(value):
        return ""
    return BeautifulSoup(unescape(str(value)), "html.parser").get_text(" ", strip=True)


def extract_area_candidates(description: str) -> list[float]:
    if not description:
        return []
    patterns = (
        r"(?<!\d)(\d{1,5}(?:[.,]\d{1,2})?)\s*m\s*(?:²|2)\b",
        r"(?<!\d)(\d{1,5}(?:[.,]\d{1,2})?)\s*metros?\s+quadrados?\b",
    )
    values: list[float] = []
    for pattern in patterns:
        for match in re.finditer(pattern, description, flags=re.IGNORECASE):
            value = normalize_area(match.group(1))
            if value is not None:
                values.append(value)
    result: list[float] = []
    for value in values:
        if not any(abs(value - existing) < 1e-9 for existing in result):
            result.append(value)
    return result


def feature_evidence(
    title: str,
    description: str,
    structured_features: list[str],
) -> dict[str, Any]:
    structured_text = normalize_text(" | ".join(structured_features))
    free_text = normalize_text(f"{title} | {description}")
    result: dict[str, Any] = {}

    for column, terms in FEATURE_TERMS.items():
        structured_hits: list[str] = []
        text_hits: list[str] = []
        negative_hits: list[str] = []

        seen_terms: set[str] = set()
        for term in terms:
            norm_term = normalize_text(term)
            if not norm_term or norm_term in seen_terms:
                continue
            seen_terms.add(norm_term)
            if norm_term in structured_text:
                structured_hits.append(term)
            if norm_term in free_text:
                text_hits.append(term)
            for template in NEGATION_PATTERNS:
                phrase = normalize_text(template.format(term=term))
                if phrase and phrase in free_text:
                    negative_hits.append(template.format(term=term))

        positive = bool(structured_hits or text_hits)
        if negative_hits and not structured_hits:
            status = "negado"
        elif positive:
            status = "positivo"
        else:
            status = "nao_informado"

        result[column] = {
            "status": status,
            "estruturado": sorted(set(structured_hits)),
            "texto": sorted(set(text_hits)),
            "negacoes": sorted(set(negative_hits)),
        }
    return result


def parse_initial_data(html: str, expected_id: str = "") -> ParsedAd:
    soup = BeautifulSoup(html, "html.parser")
    tag = soup.find("script", id="initial-data")
    if tag is None:
        raise ParseError("elemento script#initial-data nao encontrado")

    raw_json = tag.get("data-json")
    if is_blank(raw_json):
        raise ParseError("atributo data-json de script#initial-data vazio")

    try:
        data = json.loads(str(raw_json))
    except json.JSONDecodeError as error:
        raise ParseError(f"data-json invalido: {error}") from error

    ad = data.get("ad")
    if not isinstance(ad, dict):
        raise ParseError("objeto ad ausente em initial-data")

    list_id = str(ad.get("listId") or "").strip()
    if not list_id:
        raise ParseError("listId ausente no anuncio")
    if expected_id and list_id != str(expected_id).strip():
        raise ParseError(f"ID divergente: esperado {expected_id}, extraido {list_id}")

    props, prop_labels = property_map(ad.get("properties"))
    location = ad.get("location") if isinstance(ad.get("location"), dict) else {}
    description = html_description_to_text(ad.get("body"))
    area_raw = props.get("size", "")
    area_m2 = normalize_area(area_raw)
    area_candidates = extract_area_candidates(description)
    area_description_m2 = area_candidates[0] if len(area_candidates) == 1 else None

    structured_features = (
        prop_labels.get("re_features", [])
        + prop_labels.get("re_complex_features", [])
    )
    if not structured_features:
        for key in ("re_features", "re_complex_features"):
            structured_features.extend(
                [item.strip() for item in props.get(key, "").split(",") if item.strip()]
            )

    evidence = feature_evidence(
        title=str(ad.get("subject") or ""),
        description=description,
        structured_features=structured_features,
    )

    return ParsedAd(
        list_id=list_id,
        price=str(ad.get("priceValue") or "").strip(),
        city=str(location.get("municipality") or "").strip(),
        neighbourhood=str(location.get("neighbourhood") or "").strip(),
        area_raw=area_raw,
        area_m2=area_m2,
        area_description_candidates=area_candidates,
        area_description_m2=area_description_m2,
        rooms=props.get("rooms", ""),
        bathrooms=props.get("bathrooms", ""),
        garage_spaces=props.get("garage_spaces", ""),
        category=props.get("category", str(ad.get("categoryName") or "")),
        real_estate_type=props.get("real_estate_type", ""),
        re_features=prop_labels.get("re_features", [])
        or [item.strip() for item in props.get("re_features", "").split(",") if item.strip()],
        re_complex_features=prop_labels.get("re_complex_features", [])
        or [item.strip() for item in props.get("re_complex_features", "").split(",") if item.strip()],
        re_types=props.get("re_types", ""),
        description=description,
        feature_evidence=evidence,
    )


def make_driver(headless: bool, page_timeout: int):
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
    except ImportError as error:
        raise RuntimeError("Selenium nao instalado.") from error

    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.page_load_strategy = "eager"
    options.add_argument("--window-size=1366,768")
    options.add_argument("--lang=pt-BR")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-notifications")
    options.add_argument("--disable-background-networking")
    options.add_argument("--disable-sync")
    options.add_argument("--metrics-recording-only")
    options.add_argument("--disable-default-apps")

    # O HTML inicial ja contem os dados. Desabilitar JS impede que scripts de
    # publicidade, telemetria e fingerprint sejam executados e reduz muito as
    # requisicoes secundarias. Isso nao tenta contornar uma pagina de bloqueio:
    # se o documento principal vier bloqueado, a coleta para.
    options.add_experimental_option(
        "prefs",
        {
            "profile.managed_default_content_settings.javascript": 2,
            "profile.managed_default_content_settings.images": 2,
            "profile.managed_default_content_settings.stylesheets": 2,
            "profile.managed_default_content_settings.fonts": 2,
            "profile.default_content_setting_values.notifications": 2,
            "profile.default_content_setting_values.geolocation": 2,
        },
    )

    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(page_timeout)

    try:
        driver.execute_cdp_cmd("Network.enable", {})
        driver.execute_cdp_cmd(
            "Network.setBlockedURLs",
            {
                "urls": [
                    "*.js",
                    "*.css",
                    "*.png",
                    "*.jpg",
                    "*.jpeg",
                    "*.gif",
                    "*.webp",
                    "*.svg",
                    "*.ico",
                    "*.woff",
                    "*.woff2",
                    "*.ttf",
                    "*.mp4",
                    "*.webm",
                ]
            },
        )
    except Exception as error:
        print(f"Aviso: bloqueio complementar de recursos nao foi aplicado: {error}")

    return driver


def fetch_source_html(driver: Any, url: str, page_timeout: int) -> tuple[str, str]:
    from selenium.common.exceptions import TimeoutException
    from selenium.webdriver.support.ui import WebDriverWait

    try:
        driver.get(url)
    except TimeoutException:
        print("Timeout de carregamento; analisando o documento principal disponivel.")

    try:
        WebDriverWait(driver, min(page_timeout, 15), poll_frequency=0.25).until(
            lambda current: (
                'id="initial-data"' in current.page_source
                or bool(detect_block(current.page_source, current.title))
            )
        )
    except TimeoutException:
        pass

    html = driver.page_source or ""
    title = driver.title or ""
    block = detect_block(html, title)
    if block:
        raise BlockedError(block)
    if len(html) < 1000:
        raise RuntimeError("HTML vazio ou muito curto")
    return html, title


def round_robin_candidates(
    rows: list[dict[str, str]],
    max_items: int,
    completed: set[str],
    controls: int,
) -> list[dict[str, str]]:
    unique: dict[str, dict[str, str]] = {}
    for row in rows:
        ad_id = str(row.get("id_anuncio") or "").strip()
        if not ad_id or ad_id in completed or ad_id in unique:
            continue
        unique[ad_id] = row

    missing_area = [row for row in unique.values() if is_blank(row.get("area_raw"))]
    control_rows = [row for row in unique.values() if not is_blank(row.get("area_raw"))]

    groups: dict[tuple[str, str], deque[dict[str, str]]] = defaultdict(deque)
    for row in missing_area:
        key = (
            str(row.get("cidade_raw") or row.get("cidade_consultada") or "").strip(),
            str(row.get("tipo_consultado") or "").strip(),
        )
        groups[key].append(row)

    selected: list[dict[str, str]] = []
    keys = sorted(groups)
    while keys and len(selected) < max(0, max_items - controls):
        next_keys: list[tuple[str, str]] = []
        for key in keys:
            if groups[key] and len(selected) < max(0, max_items - controls):
                selected.append(groups[key].popleft())
            if groups[key]:
                next_keys.append(key)
        keys = next_keys

    for row in control_rows[: max(0, min(controls, max_items - len(selected)))]:
        selected.append(row)

    return selected[:max_items]


def build_result(row: dict[str, str], parsed: ParsedAd, requested_url: str) -> dict[str, Any]:
    original_area_missing = is_blank(row.get("area_raw"))
    area_final = parsed.area_m2 or parsed.area_description_m2
    if original_area_missing and area_final is not None:
        status = "area_recuperada"
    elif original_area_missing:
        status = "sem_area_confirmado"
    else:
        status = "controle_ok"

    if parsed.area_m2 is not None:
        area_source = "initial_data.properties.size"
    elif parsed.area_description_m2 is not None:
        area_source = "initial_data.body_unico_valor_m2"
    elif len(parsed.area_description_candidates) > 1:
        area_source = "descricao_ambigua"
    else:
        area_source = "ausente"

    description_excerpt = redact_personal_contacts(
        re.sub(r"\s+", " ", parsed.description).strip()
    )[:500]

    return {
        "data_utc": utc_now_iso(),
        "id_anuncio": row.get("id_anuncio", ""),
        "url_anuncio": row.get("url_anuncio", ""),
        "url_requisitada": requested_url,
        "cidade_raw_listagem": row.get("cidade_raw", ""),
        "tipo_consultado": row.get("tipo_consultado", ""),
        "area_raw_listagem": row.get("area_raw", ""),
        "status": status,
        "list_id_extraido": parsed.list_id,
        "id_confere": str(parsed.list_id) == str(row.get("id_anuncio") or "").strip(),
        "preco_individual": parsed.price,
        "cidade_individual": parsed.city,
        "bairro_individual": parsed.neighbourhood,
        "area_estruturada_raw": parsed.area_raw,
        "area_estruturada_m2": "" if parsed.area_m2 is None else parsed.area_m2,
        "areas_descricao_json": json.dumps(parsed.area_description_candidates, ensure_ascii=False),
        "area_descricao_m2": "" if parsed.area_description_m2 is None else parsed.area_description_m2,
        "area_final_m2": "" if area_final is None else area_final,
        "area_fonte": area_source,
        "quartos_individual": parsed.rooms,
        "banheiros_individual": parsed.bathrooms,
        "vagas_individual": parsed.garage_spaces,
        "categoria_individual": parsed.category,
        "tipo_detalhado_individual": parsed.real_estate_type,
        "re_features_json": json.dumps(parsed.re_features, ensure_ascii=False),
        "re_complex_features_json": json.dumps(parsed.re_complex_features, ensure_ascii=False),
        "re_types_raw": parsed.re_types,
        "descricao_tamanho": len(parsed.description),
        "descricao_trecho": description_excerpt,
        "evidencias_flags_json": json.dumps(parsed.feature_evidence, ensure_ascii=False),
        "observacao": (
            "mais de um valor de area na descricao; nao escolhido automaticamente"
            if len(parsed.area_description_candidates) > 1 and parsed.area_m2 is None
            else ""
        ),
    }


def write_report(result_path: Path, output_path: Path) -> None:
    rows = read_csv(result_path) if result_path.exists() else []
    latest: dict[str, dict[str, str]] = {}
    for row in rows:
        ad_id = str(row.get("id_anuncio") or "").strip()
        if ad_id:
            latest[ad_id] = row
    rows = list(latest.values())

    status_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        status_counts[row.get("status") or "sem_status"] += 1

    missing = [row for row in rows if is_blank(row.get("area_raw_listagem"))]
    recovered = [row for row in missing if row.get("status") == "area_recuperada"]
    structured = [row for row in recovered if row.get("area_fonte") == "initial_data.properties.size"]
    body = [row for row in recovered if row.get("area_fonte") == "initial_data.body_unico_valor_m2"]

    lines = [
        "RELATORIO DO PILOTO INDIVIDUAL POR HTML INICIAL",
        "=" * 76,
        f"Anuncios processados: {len(rows)}",
        f"Sem area na listagem: {len(missing)}",
        f"Area recuperada: {len(recovered)}",
        (
            f"Taxa de recuperacao: {len(recovered) / len(missing) * 100:.2f}%"
            if missing
            else "Taxa de recuperacao: n/a"
        ),
        f"Recuperada em propriedade estruturada: {len(structured)}",
        f"Recuperada apenas na descricao: {len(body)}",
        "",
        "STATUS",
        "-" * 76,
    ]
    for status, count in sorted(status_counts.items()):
        lines.append(f"{status}: {count}")

    lines += ["", "RECUPERACAO POR CIDADE/TIPO", "-" * 76]
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in missing:
        grouped[(row.get("cidade_raw_listagem", ""), row.get("tipo_consultado", ""))].append(row)
    for (city, kind), group in sorted(grouped.items()):
        count = sum(row.get("status") == "area_recuperada" for row in group)
        lines.append(f"{city} | {kind}: {count}/{len(group)}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_local_validation(files: list[Path]) -> int:
    print("VALIDACAO LOCAL DO PARSER")
    print("=" * 76)
    for path in files:
        html = path.read_text(encoding="utf-8")
        parsed = parse_initial_data(html)
        area_final = parsed.area_m2 or parsed.area_description_m2
        print(f"\nArquivo: {path}")
        print(f"ID: {parsed.list_id}")
        print(f"Cidade/bairro: {parsed.city} / {parsed.neighbourhood}")
        print(f"Area estruturada: {parsed.area_raw or '(ausente)'}")
        print(f"Areas na descricao: {parsed.area_description_candidates}")
        print(f"Area final: {area_final if area_final is not None else '(ausente)'}")
        print(f"Quartos/banheiros/vagas: {parsed.rooms} / {parsed.bathrooms} / {parsed.garage_spaces}")
        print(f"Detalhes imovel: {parsed.re_features}")
        print(f"Detalhes condominio: {parsed.re_complex_features}")
        print(f"Evidencias: {json.dumps(parsed.feature_evidence, ensure_ascii=False)}")
    return 0


def run_online(args: argparse.Namespace) -> int:
    queue_path = Path(args.queue)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "resultado_fonte_v3.csv"
    log_path = output_dir / "log_fonte_v3.csv"
    report_path = output_dir / "relatorio_fonte_v3.txt"
    debug_dir = output_dir / "debug_html"

    rows = read_csv(queue_path)
    completed = successful_ids(result_path)
    selected = round_robin_candidates(
        rows=rows,
        max_items=args.max_items,
        completed=completed,
        controls=args.controls,
    )

    if not selected:
        print("Nenhum anuncio pendente para este piloto.")
        write_report(result_path, report_path)
        return 0

    print(f"Anuncios selecionados nesta execucao: {len(selected)}")
    print("JavaScript, imagens, CSS e fontes estao desabilitados.")
    print("A execucao para imediatamente se o documento principal vier bloqueado.\n")

    driver = make_driver(headless=args.headless, page_timeout=args.page_timeout)
    try:
        for index, row in enumerate(selected, start=1):
            ad_id = str(row.get("id_anuncio") or "").strip()
            url = canonicalize_url(row.get("url_anuncio") or "")
            print(f"[{index}/{len(selected)}] {ad_id} | {url}")
            started = time.monotonic()
            try:
                html, _ = fetch_source_html(driver, url, args.page_timeout)
                parsed = parse_initial_data(html, expected_id=ad_id)
                result = build_result(row, parsed, url)
                elapsed = round(time.monotonic() - started, 2)
                append_csv(result_path, RESULT_COLUMNS, [result])
                append_csv(
                    log_path,
                    LOG_COLUMNS,
                    [
                        {
                            "data_utc": utc_now_iso(),
                            "id_anuncio": ad_id,
                            "url_requisitada": url,
                            "status": result["status"],
                            "area_recuperada": result["status"] == "area_recuperada",
                            "tempo_segundos": elapsed,
                            "observacao": result.get("observacao", ""),
                        }
                    ],
                )
                print(
                    f"  {result['status']} | area={result['area_final_m2'] or '(ausente)'} "
                    f"| fonte={result['area_fonte']} | {elapsed}s"
                )
            except BlockedError as error:
                elapsed = round(time.monotonic() - started, 2)
                try:
                    html = driver.page_source or ""
                except Exception:
                    html = ""
                debug_path = save_debug_html(debug_dir, ad_id, "bloqueio", html)
                append_csv(
                    log_path,
                    LOG_COLUMNS,
                    [
                        {
                            "data_utc": utc_now_iso(),
                            "id_anuncio": ad_id,
                            "url_requisitada": url,
                            "status": "bloqueado",
                            "area_recuperada": False,
                            "tempo_segundos": elapsed,
                            "observacao": f"{error}; HTML: {debug_path}",
                        }
                    ],
                )
                print(f"BLOQUEIO DETECTADO: {error}")
                print(f"HTML salvo em: {debug_path}")
                print("A execucao foi interrompida; nao ha tentativa de contornar o bloqueio.")
                write_report(result_path, report_path)
                return 2
            except Exception as error:
                elapsed = round(time.monotonic() - started, 2)
                try:
                    html = driver.page_source or ""
                except Exception:
                    html = ""
                debug_path = save_debug_html(debug_dir, ad_id, "erro", html)
                append_csv(
                    log_path,
                    LOG_COLUMNS,
                    [
                        {
                            "data_utc": utc_now_iso(),
                            "id_anuncio": ad_id,
                            "url_requisitada": url,
                            "status": "erro",
                            "area_recuperada": False,
                            "tempo_segundos": elapsed,
                            "observacao": f"{type(error).__name__}: {error}; HTML: {debug_path}",
                        }
                    ],
                )
                print(f"ERRO: {type(error).__name__}: {error}")
                print(f"HTML salvo em: {debug_path}")

            write_report(result_path, report_path)
            if index < len(selected):
                delay = random.uniform(args.min_sleep, args.max_sleep)
                print(f"  aguardando {delay:.1f}s...\n")
                time.sleep(delay)
    except KeyboardInterrupt:
        print("\nExecucao interrompida pelo usuario. Resultados ja gravados foram preservados.")
    finally:
        driver.quit()
        write_report(result_path, report_path)

    print("\nArquivos gerados:")
    print(result_path)
    print(log_path)
    print(report_path)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Piloto de paginas individuais usando apenas o HTML inicial do anuncio."
    )
    parser.add_argument("--queue", help="CSV da fila de anuncios.")
    parser.add_argument("--output-dir", default="piloto_individual_fonte_v3")
    parser.add_argument("--max-items", type=int, default=12)
    parser.add_argument("--controls", type=int, default=2)
    parser.add_argument("--min-sleep", type=float, default=15.0)
    parser.add_argument("--max-sleep", type=float, default=25.0)
    parser.add_argument("--page-timeout", type=int, default=25)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--local-html",
        nargs="+",
        help="Valida o parser em arquivos HTML locais, sem acessar a internet.",
    )
    args = parser.parse_args()
    if not args.local_html and not args.queue:
        parser.error("informe --queue ou --local-html")
    if args.max_items < 1:
        parser.error("--max-items deve ser >= 1")
    if args.controls < 0:
        parser.error("--controls deve ser >= 0")
    if args.min_sleep < 0 or args.max_sleep < args.min_sleep:
        parser.error("intervalo de sleep invalido")
    return args


def main() -> int:
    args = parse_args()
    if args.local_html:
        return run_local_validation([Path(item) for item in args.local_html])
    return run_online(args)


if __name__ == "__main__":
    raise SystemExit(main())
