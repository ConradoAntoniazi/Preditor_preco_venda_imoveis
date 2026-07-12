from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from bs4 import BeautifulSoup, SoupStrainer


# ============================================================
# CONFIGURACAO DAS ROTAS
# ============================================================

BASE_URL = "https://www.olx.com.br/imoveis/venda"

# Rotas atuais das cidades dentro da OLX. O tipo (casas/apartamentos)
# e acrescentado automaticamente pelo programa.
CITY_PATHS = {
    "Vitoria": ("Vitória", "vitoria"),
    "Vila Velha": ("Vila Velha", "vila-velha"),
    "Serra": ("Serra", "outras-cidades/serra"),
    "Cariacica": ("Cariacica", "outras-cidades/cariacica"),
    "Viana": ("Viana", "outras-cidades/viana"),
    "Fundao": ("Fundão", "outras-cidades/fundao"),
    "Guarapari": ("Guarapari", "guarapari-e-regiao/guarapari"),
}

TYPE_PATHS = {
    "casa": "casas",
    "apartamento": "apartamentos",
}

EXPECTED_PAGE_SIZE = 50
ABSOLUTE_MAX_PAGES = 40

RAW_COLUMNS = [
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

LOG_COLUMNS = [
    "data_utc",
    "cidade",
    "tipo",
    "pagina",
    "url",
    "status",
    "parser",
    "pagina_informada",
    "total_anuncios_informado",
    "tamanho_pagina_informado",
    "itens_brutos",
    "anuncios_organicos",
    "ids_novos",
    "tentativas",
    "observacao",
]

BLOCK_MARKERS = (
    "sorry, you have been blocked",
    "you have been blocked",
    "access denied",
    "request unsuccessful. incapsula incident id",
)


# ============================================================
# MODELOS E EXCECOES
# ============================================================


@dataclass(frozen=True)
class Target:
    cidade: str
    tipo: str
    base_url: str


@dataclass
class ListingPayload:
    ads: list[dict[str, Any]]
    source: str
    total_ads: Optional[int]
    page_size: Optional[int]
    page_index: Optional[int]
    listing_url: Optional[str]


@dataclass
class FetchResult:
    html: str
    payload: ListingPayload
    attempts: int


class BlockedError(RuntimeError):
    pass


class PageFetchError(RuntimeError):
    pass


class ListingParseError(RuntimeError):
    pass


# ============================================================
# UTILITARIOS
# ============================================================


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_execution_id() -> str:
    return datetime.now(timezone.utc).strftime("olx_%Y%m%dT%H%M%SZ")


def normalize_key(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value).strip().lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", "", text)


def safe_slug(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return text or "sem_nome"


def to_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        match = re.search(r"\d+", str(value).replace(".", ""))
        return int(match.group(0)) if match else None


def serialize_raw(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def build_page_url(base_url: str, page: int) -> str:
    parts = urlsplit(base_url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query["o"] = str(page)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def page_key(cidade: str, tipo: str, page: int) -> str:
    return f"{cidade}|{tipo}|{page}"


def target_key(cidade: str, tipo: str) -> str:
    return f"{cidade}|{tipo}"


def observation_key(cidade: str, tipo: str, page: int, ad_id: str) -> str:
    return f"{cidade}|{tipo}|{page}|{ad_id}"


def polite_sleep(min_seconds: float, max_seconds: float) -> None:
    delay = random.uniform(min_seconds, max_seconds)
    print(f"Aguardando {delay:.1f}s antes da proxima pagina...")
    time.sleep(delay)


def retry_sleep(attempt: int) -> None:
    delay = min(30.0, 4.0 * attempt) + random.uniform(0.5, 2.0)
    print(f"Nova tentativa em {delay:.1f}s...")
    time.sleep(delay)


# ============================================================
# PARSER DOS DADOS ESTRUTURADOS DA LISTAGEM
# ============================================================


def _decode_next_f_push(script_text: str) -> Optional[list[Any]]:
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


def _listing_candidates_from_tree(root: Any) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    stack = [root]

    while stack:
        current = stack.pop()

        if isinstance(current, dict):
            ads = current.get("ads")
            if isinstance(ads, list):
                candidates.append(current)
            stack.extend(current.values())

        elif isinstance(current, list):
            stack.extend(current)

    return candidates


def _choose_best_listing_candidate(
    candidates: Iterable[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    valid = [item for item in candidates if isinstance(item.get("ads"), list)]

    if not valid:
        return None

    def score(item: dict[str, Any]) -> tuple[int, int, int, int]:
        return (
            int(item.get("listingUrl") is not None),
            int(item.get("totalOfAds") is not None),
            int(item.get("pageSize") is not None),
            len(item.get("ads") or []),
        )

    return max(valid, key=score)


def _payload_from_candidate(candidate: dict[str, Any], source: str) -> ListingPayload:
    ads = candidate.get("ads")

    if not isinstance(ads, list):
        raise ListingParseError("O bloco encontrado nao contem uma lista de anuncios.")

    return ListingPayload(
        ads=[item for item in ads if isinstance(item, dict)],
        source=source,
        total_ads=to_int(candidate.get("totalOfAds")),
        page_size=to_int(candidate.get("pageSize")),
        page_index=to_int(candidate.get("pageIndex")),
        listing_url=candidate.get("listingUrl"),
    )


def extract_payload_from_react_flight(soup: BeautifulSoup) -> Optional[ListingPayload]:
    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    object_pattern = re.compile(r'\{"ads"\s*:\s*\[')

    for script in soup.find_all("script"):
        script_text = script.string or script.get_text() or ""
        pushed = _decode_next_f_push(script_text)

        if not pushed or len(pushed) < 2 or not isinstance(pushed[1], str):
            continue

        decoded_chunk = pushed[1]

        for match in object_pattern.finditer(decoded_chunk):
            try:
                value, _ = decoder.raw_decode(decoded_chunk[match.start():])
            except json.JSONDecodeError:
                continue

            if isinstance(value, dict) and isinstance(value.get("ads"), list):
                candidates.append(value)

    candidate = _choose_best_listing_candidate(candidates)
    return _payload_from_candidate(candidate, "react_flight") if candidate else None


def extract_payload_from_next_data(soup: BeautifulSoup) -> Optional[ListingPayload]:
    script = soup.find("script", {"id": "__NEXT_DATA__"})

    if not script:
        return None

    text = script.string or script.get_text() or ""

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    candidate = _choose_best_listing_candidate(_listing_candidates_from_tree(data))
    return _payload_from_candidate(candidate, "__NEXT_DATA__") if candidate else None


def fallback_total_ads(html: str) -> Optional[int]:
    patterns = (
        r'"totalOfAds"\s*:\s*(\d+)',
        r'\\"totalOfAds\\"\s*:\s*(\d+)',
        r"\bde\s+([\d\.]+)\s+resultados\b",
    )

    for pattern in patterns:
        match = re.search(pattern, html, flags=re.IGNORECASE)
        if match:
            return to_int(match.group(1))

    return None


def parse_listing_html(html: str) -> ListingPayload:
    # Apenas os elementos <script> interessam ao parser. Evitar construir a
    # arvore completa da pagina reduz CPU e memoria em HTMLs muito grandes.
    soup = BeautifulSoup(
        html,
        "html.parser",
        parse_only=SoupStrainer("script"),
    )

    payload = extract_payload_from_react_flight(soup)
    if payload is None:
        payload = extract_payload_from_next_data(soup)

    if payload is None:
        raise ListingParseError(
            "Nenhum bloco estruturado de listagem foi reconhecido. "
            "A pagina pode ter mudado ou nao ter carregado por completo."
        )

    if payload.total_ads is None:
        payload.total_ads = fallback_total_ads(html)

    if payload.page_size is None or payload.page_size <= 0:
        payload.page_size = EXPECTED_PAGE_SIZE

    return payload


# ============================================================
# NORMALIZACAO MINIMA PARA O CSV RAW
# ============================================================


def properties_to_dict(properties: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}

    if not isinstance(properties, list):
        return result

    for item in properties:
        if not isinstance(item, dict):
            continue

        name = item.get("name")
        if not name:
            continue

        value = item.get("value")

        if name not in result:
            result[name] = value
        elif result[name] != value:
            previous = result[name]
            if not isinstance(previous, list):
                previous = [previous]
            previous.append(value)
            result[name] = previous

    return result


def is_organic_listing(ad: dict[str, Any]) -> bool:
    if "advertisingId" in ad:
        return False

    return bool(ad.get("listId") and ad.get("url"))


def raw_row_from_ad(
    ad: dict[str, Any],
    target: Target,
    page: int,
    page_url: str,
    execution_id: str,
    collection_time: str,
    fallback_position: int,
) -> dict[str, Any]:
    props = properties_to_dict(ad.get("properties"))
    location = ad.get("locationDetails") or {}

    return {
        "execucao_id": execution_id,
        "data_coleta_utc": collection_time,
        "cidade_consultada": target.cidade,
        "tipo_consultado": target.tipo,
        "pagina": page,
        # A OLX informa a posição em base zero (0..49). O teste anterior
        # usava `or`, então a posição 0 virava 1 e colidia com a posição 1.
        # Aqui preservamos a informação e exportamos em base um (1..50).
        "posicao_pagina": (
            int(ad.get("position")) + 1
            if ad.get("position") is not None
            else fallback_position
        ),
        "url_listagem": page_url,
        "id_anuncio": str(ad.get("listId")),
        "url_anuncio": ad.get("url"),
        "titulo_raw": ad.get("subject"),
        "preco_raw": ad.get("priceValue") or ad.get("price"),
        "categoria_raw": ad.get("categoryName") or ad.get("category"),
        "tipo_detalhado_raw": serialize_raw(props.get("real_estate_type")),
        "re_types_raw": serialize_raw(props.get("re_types")),
        "cidade_raw": location.get("municipality"),
        "bairro_raw": location.get("neighbourhood"),
        "uf_raw": location.get("uf"),
        "area_raw": serialize_raw(props.get("size")),
        "quartos_raw": serialize_raw(props.get("rooms")),
        "banheiros_raw": serialize_raw(props.get("bathrooms")),
        "vagas_raw": serialize_raw(props.get("garage_spaces")),
        "re_features_raw": serialize_raw(props.get("re_features")),
        "re_complex_features_raw": serialize_raw(props.get("re_complex_features")),
        "anunciante_profissional_raw": ad.get("professionalAd"),
    }


# ============================================================
# ARQUIVOS: RAW, LOG E CHECKPOINT
# ============================================================


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


def append_log(path: Path, **values: Any) -> None:
    row = {column: values.get(column) for column in LOG_COLUMNS}
    append_csv(path, LOG_COLUMNS, [row])


def load_existing_raw_state(
    raw_path: Path,
) -> tuple[set[str], dict[str, set[str]], dict[tuple[str, int], set[str]]]:
    observation_keys: set[str] = set()
    ids_by_target: dict[str, set[str]] = defaultdict(set)
    ids_by_page: dict[tuple[str, int], set[str]] = defaultdict(set)

    if not raw_path.exists() or raw_path.stat().st_size == 0:
        return observation_keys, ids_by_target, ids_by_page

    with raw_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            cidade = row.get("cidade_consultada") or ""
            tipo = row.get("tipo_consultado") or ""
            ad_id = row.get("id_anuncio") or ""
            page = to_int(row.get("pagina"))

            if not cidade or not tipo or not ad_id or page is None:
                continue

            observation_keys.add(observation_key(cidade, tipo, page, ad_id))
            ids_by_target[target_key(cidade, tipo)].add(ad_id)
            ids_by_page[(target_key(cidade, tipo), page)].add(ad_id)

    return observation_keys, ids_by_target, ids_by_page


def default_checkpoint() -> dict[str, Any]:
    return {
        "execution_id": new_execution_id(),
        "completed_pages": [],
        "targets": {},
        "updated_at_utc": utc_now_iso(),
    }


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return default_checkpoint()

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise RuntimeError(
            f"Checkpoint invalido: {path}. Use outra pasta de saida ou corrija o arquivo."
        )

    if not isinstance(data, dict):
        raise RuntimeError(f"Checkpoint invalido: {path}")

    data.setdefault("execution_id", new_execution_id())
    data.setdefault("completed_pages", [])
    data.setdefault("targets", {})
    return data


def save_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    checkpoint["updated_at_utc"] = utc_now_iso()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(checkpoint, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def save_debug_html(debug_dir: Path, target: Target, page: int, reason: str, html: str) -> Path:
    debug_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = (
        f"{timestamp}_{safe_slug(target.cidade)}_{safe_slug(target.tipo)}_"
        f"pagina_{page}_{safe_slug(reason)}.html"
    )
    path = debug_dir / filename
    path.write_text(html, encoding="utf-8")
    return path


# ============================================================
# SELENIUM E COLETA
# ============================================================


def make_driver(headless: bool, page_timeout: int, block_resources: bool = True):
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
    except ImportError as error:
        raise RuntimeError(
            "Selenium nao esta instalado. Execute: "
            "pip install -r requirements_scraper_olx.txt"
        ) from error

    options = Options()

    if headless:
        options.add_argument("--headless=new")

    # A listagem estruturada vem no HTML. Nao precisamos aguardar imagens,
    # publicidade e todos os recursos secundarios terminarem de carregar.
    options.page_load_strategy = "eager"

    options.add_argument("--window-size=1440,1000")
    options.add_argument("--lang=pt-BR")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-notifications")
    options.add_argument("--mute-audio")
    options.add_argument("--no-first-run")
    options.add_argument("--disable-background-networking")
    options.add_argument("--disable-features=Translate,MediaRouter")

    if block_resources:
        options.add_experimental_option(
            "prefs",
            {
                "profile.managed_default_content_settings.images": 2,
                "profile.default_content_setting_values.notifications": 2,
                "profile.default_content_setting_values.geolocation": 2,
                "profile.default_content_setting_values.media_stream_mic": 2,
                "profile.default_content_setting_values.media_stream_camera": 2,
            },
        )

    # Selenium Manager, incluido nas versoes recentes do Selenium,
    # localiza ou instala o ChromeDriver compativel.
    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(page_timeout)

    if block_resources:
        # Recursos visuais e rastreadores nao sao usados pelo parser. O bloqueio
        # reduz trafego sem impedir os scripts e o HTML principais da OLX.
        try:
            driver.execute_cdp_cmd("Network.enable", {})
            driver.execute_cdp_cmd(
                "Network.setBlockedURLs",
                {
                    "urls": [
                        "*.png",
                        "*.jpg",
                        "*.jpeg",
                        "*.gif",
                        "*.webp",
                        "*.ico",
                        "*.woff",
                        "*.woff2",
                        "*.ttf",
                        "*.mp4",
                        "*.webm",
                        "*doubleclick.net/*",
                        "*googletagmanager.com/*",
                        "*google-analytics.com/*",
                        "*taboola.com/*",
                        "*criteo.com/*",
                        "*rubiconproject.com/*",
                        "*scorecardresearch.com/*",
                        "*mouseflow.com/*",
                        "*navdmp.com/*",
                        "*datadoghq.com/*",
                        "*adjust.com/*",
                    ]
                },
            )
        except Exception as error:
            print(f"Aviso: nao foi possivel bloquear recursos secundarios: {error}")

    return driver


def browser_has_listing_marker(driver: Any) -> bool:
    """Verifica os marcadores no navegador sem transferir todo o HTML a cada poll."""
    try:
        return bool(
            driver.execute_script(
                """
                const root = document.documentElement;
                if (!root) return false;
                if (document.getElementById('__NEXT_DATA__')) return true;

                const scripts = document.scripts || [];
                for (let i = 0; i < scripts.length; i++) {
                    const value = scripts[i].textContent || '';
                    if (
                        value.includes('totalOfAds') ||
                        value.includes('{\"ads\":[') ||
                        value.includes('\\"ads\\":[')
                    ) {
                        return true;
                    }
                }

                const bodyText = ((document.body && document.body.innerText) || '').toLowerCase();
                return (
                    bodyText.includes('you have been blocked') ||
                    bodyText.includes('access denied')
                );
                """
            )
        )
    except Exception:
        return False


def is_blocked_page(html: str) -> bool:
    lower = html.lower()
    return any(marker in lower for marker in BLOCK_MARKERS)


def has_listing_marker(html: str) -> bool:
    return (
        r'{\"ads\":[' in html
        or '{"ads":[' in html
        or "__NEXT_DATA__" in html
        or "totalOfAds" in html
    )


def fetch_listing_page(
    driver: Any,
    target: Target,
    page: int,
    page_url: str,
    page_timeout: int,
    marker_timeout: int,
    max_retries: int,
    debug_dir: Path,
) -> FetchResult:
    try:
        from selenium.common.exceptions import TimeoutException, WebDriverException
        from selenium.webdriver.support.ui import WebDriverWait
    except ImportError as error:
        raise RuntimeError("Selenium nao esta instalado corretamente.") from error

    last_error: Optional[Exception] = None
    last_html = ""

    for attempt in range(1, max_retries + 1):
        print(
            f"\n[{target.cidade} | {target.tipo} | pagina {page}] "
            f"tentativa {attempt}/{max_retries}"
        )
        print(page_url)

        try:
            driver.get(page_url)
        except TimeoutException:
            # Em varios casos o HTML util ja esta disponivel mesmo quando
            # recursos secundarios ultrapassam o timeout.
            print("O carregamento excedeu o timeout; tentando analisar o HTML disponivel.")
        except WebDriverException as error:
            last_error = error
            if attempt < max_retries:
                retry_sleep(attempt)
                continue
            raise PageFetchError(f"Falha do navegador: {error}") from error

        try:
            WebDriverWait(
                driver,
                marker_timeout,
                poll_frequency=0.25,
            ).until(browser_has_listing_marker)
        except TimeoutException:
            print("Os dados estruturados nao apareceram dentro do timeout curto.")

        # Assim que o bloco estruturado existe, interrompemos imagens, anuncios e
        # outros carregamentos de fundo. O HTML util permanece disponivel.
        try:
            driver.execute_script("window.stop();")
        except WebDriverException:
            pass

        last_html = driver.page_source

        if is_blocked_page(last_html):
            debug_path = save_debug_html(debug_dir, target, page, "bloqueio", last_html)
            raise BlockedError(
                f"Bloqueio detectado. HTML salvo em {debug_path}. "
                "A execucao foi interrompida para nao insistir na plataforma."
            )

        try:
            payload = parse_listing_html(last_html)
            return FetchResult(html=last_html, payload=payload, attempts=attempt)
        except ListingParseError as error:
            last_error = error
            print(f"Falha do parser: {error}")
            if attempt < max_retries:
                retry_sleep(attempt)

    debug_path = save_debug_html(debug_dir, target, page, "parser", last_html)
    raise PageFetchError(
        f"A pagina nao pode ser interpretada apos {max_retries} tentativas. "
        f"HTML salvo em {debug_path}. Motivo final: {last_error}"
    )


# ============================================================
# ALVOS E ARGUMENTOS
# ============================================================


def canonical_city_map() -> dict[str, str]:
    result: dict[str, str] = {}
    for key, (display_name, _) in CITY_PATHS.items():
        result[normalize_key(key)] = display_name
        result[normalize_key(display_name)] = display_name
    return result


def parse_cities(value: str) -> list[str]:
    if normalize_key(value) in {"todas", "todos", "all"}:
        return [display_name for display_name, _ in CITY_PATHS.values()]

    aliases = canonical_city_map()
    selected: list[str] = []

    for item in value.split(","):
        normalized = normalize_key(item)
        if normalized not in aliases:
            valid = ", ".join(display for display, _ in CITY_PATHS.values())
            raise argparse.ArgumentTypeError(
                f"Cidade desconhecida: {item.strip()}. Opcoes: {valid}, ou 'todas'."
            )
        city = aliases[normalized]
        if city not in selected:
            selected.append(city)

    return selected


def parse_types(value: str) -> list[str]:
    aliases = {
        "casa": "casa",
        "casas": "casa",
        "apartamento": "apartamento",
        "apartamentos": "apartamento",
    }
    selected: list[str] = []

    for item in value.split(","):
        normalized = normalize_key(item)
        if normalized not in aliases:
            raise argparse.ArgumentTypeError(
                "Tipo desconhecido. Use casa, apartamento ou ambos separados por virgula."
            )
        property_type = aliases[normalized]
        if property_type not in selected:
            selected.append(property_type)

    return selected


def city_path_for(display_name: str) -> str:
    for _, (display, path) in CITY_PATHS.items():
        if display == display_name:
            return path
    raise KeyError(display_name)


def build_targets(cities: list[str], types: list[str]) -> list[Target]:
    targets: list[Target] = []

    for city in cities:
        city_path = city_path_for(city)
        for property_type in types:
            plural = TYPE_PATHS[property_type]
            url = f"{BASE_URL}/{plural}/estado-es/norte-do-espirito-santo/{city_path}"
            targets.append(Target(cidade=city, tipo=property_type, base_url=url))

    return targets


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Coleta dados brutos das paginas de listagem da OLX. "
            "Nao visita paginas individuais e nao cria atributos derivados."
        )
    )
    parser.add_argument(
        "--cities",
        default="Vitória,Vila Velha",
        help="Cidades separadas por virgula, ou 'todas'. Padrao: Vitória,Vila Velha.",
    )
    parser.add_argument(
        "--types",
        default="casa,apartamento",
        help="Tipos separados por virgula. Padrao: casa,apartamento.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=2,
        help="Limite por combinacao cidade/tipo. Padrao do piloto: 2; maximo: 40.",
    )
    parser.add_argument(
        "--output-dir",
        default="coleta_olx_piloto",
        help="Pasta dos CSVs, checkpoint e HTMLs de erro.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Executa o Chrome sem janela. No primeiro teste, prefira deixar desativado.",
    )
    parser.add_argument("--min-sleep", type=float, default=4.0)
    parser.add_argument("--max-sleep", type=float, default=7.0)
    parser.add_argument(
        "--page-timeout",
        type=int,
        default=20,
        help="Limite do driver.get em segundos. Padrao otimizado: 20.",
    )
    parser.add_argument(
        "--marker-timeout",
        type=int,
        default=10,
        help="Espera adicional pelo bloco estruturado. Padrao: 10.",
    )
    parser.add_argument(
        "--restart-driver-every",
        type=int,
        default=40,
        help="Reinicia o Chrome apos N paginas concluidas; 0 desativa. Padrao: 40.",
    )
    parser.add_argument(
        "--no-block-resources",
        action="store_true",
        help="Nao bloqueia imagens, fontes e rastreadores. Use apenas para diagnostico.",
    )
    parser.add_argument("--retries", type=int, default=2)
    return parser


# ============================================================
# EXECUCAO PRINCIPAL
# ============================================================


def compute_planned_pages(total_ads: Optional[int], page_size: Optional[int], max_pages: int) -> int:
    if total_ads is None:
        return max_pages

    if total_ads <= 0:
        return 1

    effective_page_size = page_size or EXPECTED_PAGE_SIZE
    return min(max_pages, max(1, math.ceil(total_ads / effective_page_size)))


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    if not 1 <= args.max_pages <= ABSOLUTE_MAX_PAGES:
        parser.error(f"--max-pages deve estar entre 1 e {ABSOLUTE_MAX_PAGES}.")

    if args.min_sleep < 0 or args.max_sleep < args.min_sleep:
        parser.error("Intervalo de espera invalido.")

    if args.page_timeout <= 0 or args.marker_timeout <= 0:
        parser.error("Os timeouts devem ser positivos.")

    if args.restart_driver_every < 0:
        parser.error("--restart-driver-every nao pode ser negativo.")

    try:
        cities = parse_cities(args.cities)
        types = parse_types(args.types)
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))

    targets = build_targets(cities, types)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / "olx_listagens_raw.csv"
    log_path = output_dir / "olx_log_paginas.csv"
    checkpoint_path = output_dir / "checkpoint.json"
    debug_dir = output_dir / "debug_pages"

    checkpoint = load_checkpoint(checkpoint_path)
    completed_pages = set(checkpoint.get("completed_pages", []))
    checkpoint_targets = checkpoint.setdefault("targets", {})
    execution_id = str(checkpoint["execution_id"])

    observation_keys, ids_by_target, ids_by_page = load_existing_raw_state(raw_path)

    print("\nConfiguracao da coleta")
    print(f"Cidades: {', '.join(cities)}")
    print(f"Tipos: {', '.join(types)}")
    print(f"Limite por combinacao: {args.max_pages} pagina(s)")
    print(f"Espera entre paginas: {args.min_sleep:.1f}s a {args.max_sleep:.1f}s")
    print(
        f"Timeout de carga: {args.page_timeout}s | "
        f"espera pelo marcador: {args.marker_timeout}s"
    )
    print(
        "Recursos secundarios bloqueados: "
        f"{'nao' if args.no_block_resources else 'sim'}"
    )
    print(f"Saida: {output_dir.resolve()}")
    print(f"Execucao: {execution_id}")

    driver = None
    fatal_error: Optional[Exception] = None

    pages_since_driver_restart = 0

    try:
        driver = make_driver(
            args.headless,
            args.page_timeout,
            block_resources=not args.no_block_resources,
        )

        for target in targets:
            key = target_key(target.cidade, target.tipo)
            target_meta = checkpoint_targets.setdefault(key, {})
            total_ads = to_int(target_meta.get("total_ads"))
            page_size = to_int(target_meta.get("page_size")) or EXPECTED_PAGE_SIZE
            page_limit = compute_planned_pages(total_ads, page_size, args.max_pages)

            print("\n" + "=" * 76)
            print(f"ALVO: {target.cidade} | {target.tipo}")
            print(f"URL base: {target.base_url}")
            if total_ads is not None:
                print(f"Total ja conhecido: {total_ads}; paginas planejadas: {page_limit}")

            page = 1
            while page <= page_limit:
                current_page_key = page_key(target.cidade, target.tipo, page)

                if current_page_key in completed_pages:
                    print(f"Pagina {page} ja concluida; pulando.")
                    page += 1
                    continue

                if (
                    args.restart_driver_every > 0
                    and pages_since_driver_restart >= args.restart_driver_every
                ):
                    print(
                        f"Reiniciando o Chrome apos {pages_since_driver_restart} "
                        "pagina(s) para evitar degradacao de desempenho..."
                    )
                    driver.quit()
                    driver = make_driver(
                        args.headless,
                        args.page_timeout,
                        block_resources=not args.no_block_resources,
                    )
                    pages_since_driver_restart = 0

                page_url = build_page_url(target.base_url, page)

                try:
                    result = fetch_listing_page(
                        driver=driver,
                        target=target,
                        page=page,
                        page_url=page_url,
                        page_timeout=args.page_timeout,
                        marker_timeout=args.marker_timeout,
                        max_retries=args.retries,
                        debug_dir=debug_dir,
                    )
                except (BlockedError, PageFetchError) as error:
                    status = "bloqueado" if isinstance(error, BlockedError) else "erro_fatal"
                    append_log(
                        log_path,
                        data_utc=utc_now_iso(),
                        cidade=target.cidade,
                        tipo=target.tipo,
                        pagina=page,
                        url=page_url,
                        status=status,
                        tentativas=args.retries,
                        observacao=str(error),
                    )
                    fatal_error = error
                    raise

                pages_since_driver_restart += 1
                payload = result.payload
                organic_ads = [ad for ad in payload.ads if is_organic_listing(ad)]
                ad_ids = {str(ad.get("listId")) for ad in organic_ads}

                if payload.total_ads is not None:
                    total_ads = payload.total_ads
                    target_meta["total_ads"] = total_ads

                if payload.page_size is not None and payload.page_size > 0:
                    page_size = payload.page_size
                    target_meta["page_size"] = page_size

                page_limit = compute_planned_pages(total_ads, page_size, args.max_pages)
                target_meta["paginas_planejadas_nesta_execucao"] = page_limit
                target_meta["url_base"] = target.base_url

                print(
                    f"Parser: {payload.source} | itens brutos: {len(payload.ads)} | "
                    f"anuncios organicos: {len(organic_ads)}"
                )
                if total_ads is not None:
                    print(
                        f"Total informado pela OLX: {total_ads} | "
                        f"paginas a visitar com o limite atual: {page_limit}"
                    )

                if payload.page_index is not None and payload.page_index != page:
                    message = (
                        f"A URL solicitou a pagina {page}, mas a pagina informou "
                        f"indice {payload.page_index}."
                    )
                    append_log(
                        log_path,
                        data_utc=utc_now_iso(),
                        cidade=target.cidade,
                        tipo=target.tipo,
                        pagina=page,
                        url=page_url,
                        status="pagina_incorreta",
                        parser=payload.source,
                        pagina_informada=payload.page_index,
                        total_anuncios_informado=total_ads,
                        tamanho_pagina_informado=page_size,
                        itens_brutos=len(payload.ads),
                        anuncios_organicos=len(organic_ads),
                        ids_novos=0,
                        tentativas=result.attempts,
                        observacao=message,
                    )
                    fatal_error = PageFetchError(message)
                    raise fatal_error

                previous_page_ids = ids_by_page.get((key, page - 1), set())
                if page > 1 and ad_ids and ad_ids == previous_page_ids:
                    append_log(
                        log_path,
                        data_utc=utc_now_iso(),
                        cidade=target.cidade,
                        tipo=target.tipo,
                        pagina=page,
                        url=page_url,
                        status="fim_pagina_repetida",
                        parser=payload.source,
                        pagina_informada=payload.page_index,
                        total_anuncios_informado=total_ads,
                        tamanho_pagina_informado=page_size,
                        itens_brutos=len(payload.ads),
                        anuncios_organicos=len(organic_ads),
                        ids_novos=0,
                        tentativas=result.attempts,
                        observacao="O conjunto de IDs repetiu integralmente a pagina anterior.",
                    )
                    target_meta["motivo_parada"] = "pagina_repetida"
                    save_checkpoint(checkpoint_path, checkpoint)
                    break

                if not organic_ads:
                    status = "fim_sem_anuncios"
                    observation = "A pagina estruturada nao trouxe anuncios organicos."

                    if total_ads and page <= page_limit:
                        status = "vazia_inesperada"
                        observation = (
                            "A pagina ficou vazia antes do fim calculado pela contagem total. "
                            "A coleta foi interrompida para revisao."
                        )

                    append_log(
                        log_path,
                        data_utc=utc_now_iso(),
                        cidade=target.cidade,
                        tipo=target.tipo,
                        pagina=page,
                        url=page_url,
                        status=status,
                        parser=payload.source,
                        pagina_informada=payload.page_index,
                        total_anuncios_informado=total_ads,
                        tamanho_pagina_informado=page_size,
                        itens_brutos=len(payload.ads),
                        anuncios_organicos=0,
                        ids_novos=0,
                        tentativas=result.attempts,
                        observacao=observation,
                    )
                    target_meta["motivo_parada"] = status
                    save_checkpoint(checkpoint_path, checkpoint)

                    if status == "vazia_inesperada":
                        fatal_error = PageFetchError(observation)
                        raise fatal_error
                    break

                already_saved = all(
                    observation_key(target.cidade, target.tipo, page, ad_id)
                    in observation_keys
                    for ad_id in ad_ids
                )

                new_ids = ad_ids - ids_by_target[key]

                if page > 1 and not new_ids and not already_saved:
                    append_log(
                        log_path,
                        data_utc=utc_now_iso(),
                        cidade=target.cidade,
                        tipo=target.tipo,
                        pagina=page,
                        url=page_url,
                        status="fim_sem_ids_novos",
                        parser=payload.source,
                        pagina_informada=payload.page_index,
                        total_anuncios_informado=total_ads,
                        tamanho_pagina_informado=page_size,
                        itens_brutos=len(payload.ads),
                        anuncios_organicos=len(organic_ads),
                        ids_novos=0,
                        tentativas=result.attempts,
                        observacao="Todos os anuncios ja haviam aparecido em paginas anteriores.",
                    )
                    target_meta["motivo_parada"] = "sem_ids_novos"
                    save_checkpoint(checkpoint_path, checkpoint)
                    break

                collection_time = utc_now_iso()
                rows: list[dict[str, Any]] = []

                for position, ad in enumerate(organic_ads, start=1):
                    ad_id = str(ad.get("listId"))
                    obs_key = observation_key(target.cidade, target.tipo, page, ad_id)

                    if obs_key in observation_keys:
                        continue

                    rows.append(
                        raw_row_from_ad(
                            ad=ad,
                            target=target,
                            page=page,
                            page_url=page_url,
                            execution_id=execution_id,
                            collection_time=collection_time,
                            fallback_position=position,
                        )
                    )
                    observation_keys.add(obs_key)

                append_csv(raw_path, RAW_COLUMNS, rows)

                ids_by_target[key].update(ad_ids)
                ids_by_page[(key, page)] = ad_ids
                completed_pages.add(current_page_key)
                checkpoint["completed_pages"] = sorted(completed_pages)
                target_meta["ultima_pagina_concluida"] = page
                target_meta.pop("motivo_parada", None)
                save_checkpoint(checkpoint_path, checkpoint)

                append_log(
                    log_path,
                    data_utc=collection_time,
                    cidade=target.cidade,
                    tipo=target.tipo,
                    pagina=page,
                    url=page_url,
                    status="ok" if not already_saved else "ok_ja_estava_salva",
                    parser=payload.source,
                    pagina_informada=payload.page_index,
                    total_anuncios_informado=total_ads,
                    tamanho_pagina_informado=page_size,
                    itens_brutos=len(payload.ads),
                    anuncios_organicos=len(organic_ads),
                    ids_novos=len(new_ids),
                    tentativas=result.attempts,
                    observacao=f"{len(rows)} linha(s) acrescentada(s) ao CSV raw.",
                )

                print(
                    f"Pagina concluida: {len(rows)} linha(s) adicionada(s); "
                    f"{len(new_ids)} ID(s) novo(s) para este alvo."
                )

                if total_ads is None and len(organic_ads) < page_size:
                    target_meta["motivo_parada"] = "pagina_curta_sem_total"
                    save_checkpoint(checkpoint_path, checkpoint)
                    print("Pagina curta e sem contagem total; considerando fim da paginacao.")
                    break

                page += 1
                if page <= page_limit:
                    polite_sleep(args.min_sleep, args.max_sleep)

    except (BlockedError, PageFetchError) as error:
        print(f"\nCOLETA INTERROMPIDA: {error}")
    except KeyboardInterrupt:
        fatal_error = KeyboardInterrupt()
        print("\nColeta interrompida pelo usuario. O checkpoint foi preservado.")
    except Exception as error:
        fatal_error = error
        print(f"\nERRO NA EXECUCAO: {error}")
    finally:
        checkpoint["completed_pages"] = sorted(completed_pages)
        save_checkpoint(checkpoint_path, checkpoint)
        if driver is not None:
            driver.quit()

    print("\nArquivos da execucao")
    print(f"RAW: {raw_path.resolve()}")
    print(f"LOG: {log_path.resolve()}")
    print(f"CHECKPOINT: {checkpoint_path.resolve()}")

    if fatal_error is not None:
        return 1

    print("Coleta finalizada sem erro fatal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
