#!/usr/bin/env python3
"""
Enriquece as seis flags de imoveis OLX a partir do HTML inicial da pagina individual.

Principios:
- nao executa JavaScript;
- nao carrega imagens, CSS, fontes ou publicidade;
- le o fluxo da resposta somente ate o fim de script#initial-data;
- usa uma unica sessao HTTP, sem paralelismo;
- para no primeiro bloqueio confirmado;
- grava resultado e log apos cada anuncio, permitindo retomada;
- so transforma ausencia em False quando a pagina foi obtida, o listId conferiu e
  script#initial-data foi interpretado com sucesso.

Importante: False significa "a caracteristica nao foi declarada nas fontes
processadas do anuncio", nao uma garantia fisica de ausencia no imovel.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
import unicodedata
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup

FLAGS = [
    "tem_suite",
    "tem_piscina",
    "tem_varanda",
    "tem_lazer",
    "tem_mobiliado",
    "tem_elevador",
]

RESULT_COLUMNS = [
    "data_utc",
    "id_anuncio",
    "url_anuncio",
    "status",
    "http_status",
    "list_id_extraido",
    "bytes_lidos",
    "tempo_segundos",
    "titulo_individual",
    "descricao_tamanho",
    "re_features_json",
    "re_complex_features_json",
    "observacao",
]
for _flag in FLAGS:
    RESULT_COLUMNS.extend(
        [
            _flag,
            f"{_flag}_fonte",
            f"{_flag}_evidencia",
            f"{_flag}_conflito",
        ]
    )

LOG_COLUMNS = [
    "data_utc",
    "id_anuncio",
    "url_anuncio",
    "status",
    "http_status",
    "bytes_lidos",
    "tempo_segundos",
    "observacao",
]

BLOCK_MARKERS = (
    "sorry, you have been blocked",
    "you are unable to access",
    "attention required! | cloudflare",
    "cloudflare ray id",
    "cf-error-details",
    "verifique que voce e humano",
    "verifique que você é humano",
    "confirme que voce e humano",
    "confirme que você é humano",
)

# Termos aceitos como evidencia positiva. Em texto livre, contextos negativos,
# hipoteticos e de proximidade sao removidos antes da classificacao.
FEATURE_TERMS = {
    "tem_suite": ("suite", "suíte"),
    "tem_piscina": ("piscina",),
    "tem_varanda": ("varanda", "sacada"),
    "tem_lazer": (
        "area de lazer",
        "área de lazer",
        "lazer",
        "academia",
        "fitness",
        "playground",
        "salao de festas",
        "salão de festas",
        "brinquedoteca",
        "espaco brincar",
        "espaço brincar",
        "espaco gourmet",
        "espaço gourmet",
        "churrasqueira",
        "quadra",
        "sauna",
        "spa",
        "clube",
        "piscina",
    ),
    "tem_mobiliado": (
        "mobiliado",
        "mobiliada",
        "porteira fechada",
        "com mobilia",
        "com mobília",
    ),
    "tem_elevador": ("elevador",),
}

# Esses contextos nao devem virar evidencia positiva apenas porque contêm o termo.
NEGATIVE_CONTEXTS = (
    r"\bsem\b",
    r"\bnao\s+(?:tem|possui|conta|dispoe)\b",
    r"\bnão\s+(?:tem|possui|conta|dispõe)\b",
    r"\bausencia\s+de\b",
    r"\bausência\s+de\b",
)

HYPOTHETICAL_OR_EXTERNAL_CONTEXTS = (
    r"\bpossibilidade\s+de\b",
    r"\bprojeto\s+(?:de|para)\b",
    r"\bespaco\s+para\b",
    r"\bespaço\s+para\b",
    r"\bpreparad[oa]\s+para\b",
    r"\bprevist[oa]\b",
    r"\bfutur[oa]\b",
    r"\bproxim[oa]\s+(?:a|ao|da|do)\b",
    r"\bpróxim[oa]\s+(?:a|ao|da|do)\b",
    r"\bperto\s+(?:de|da|do)\b",
    r"\bao\s+lado\s+(?:de|da|do)\b",
)

# "moveis planejados" nao equivale a imovel mobiliado.
MOBILIADO_FALSE_FRIENDS = (
    "moveis planejados",
    "móveis planejados",
    "armarios planejados",
    "armários planejados",
)


class BlockedError(RuntimeError):
    pass


class ParseError(RuntimeError):
    pass


@dataclass
class PageData:
    list_id: str
    title: str
    description: str
    re_features: list[str]
    re_complex_features: list[str]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "").lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", text).strip()


def canonicalize_url(url: str) -> str:
    parts = urlsplit(str(url or "").strip())
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def is_blank(value: Any) -> bool:
    return value is None or str(value).strip() == ""


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


def successful_ids(path: Path) -> set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    return {
        str(row.get("id_anuncio") or "").strip()
        for row in read_csv(path)
        if row.get("status") == "ok" and str(row.get("id_anuncio") or "").strip()
    }


def save_debug_prefix(debug_dir: Path, ad_id: str, reason: str, data: bytes) -> Path:
    debug_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = debug_dir / f"{stamp}_{ad_id}_{reason}.html"
    path.write_bytes(data)
    return path


def detect_block(text: str) -> str:
    sample = normalize_text(text[:15000])
    for marker in BLOCK_MARKERS:
        if normalize_text(marker) in sample:
            return marker
    return ""


def parse_initial_data(html: str, expected_id: str) -> PageData:
    soup = BeautifulSoup(html, "html.parser")
    tag = soup.find("script", id="initial-data")
    if tag is None:
        raise ParseError("script#initial-data nao encontrado")
    raw_json = tag.get("data-json")
    if is_blank(raw_json):
        raise ParseError("data-json ausente ou vazio")
    try:
        data = json.loads(unescape(str(raw_json)))
    except json.JSONDecodeError as error:
        raise ParseError(f"data-json invalido: {error}") from error
    ad = data.get("ad")
    if not isinstance(ad, dict):
        raise ParseError("objeto ad ausente")
    list_id = str(ad.get("listId") or "").strip()
    if not list_id:
        raise ParseError("listId ausente")
    if list_id != str(expected_id).strip():
        raise ParseError(f"ID divergente: esperado {expected_id}, obtido {list_id}")

    title = str(ad.get("subject") or "").strip()
    description = BeautifulSoup(
        unescape(str(ad.get("body") or "")), "html.parser"
    ).get_text(" ", strip=True)

    re_features: list[str] = []
    re_complex_features: list[str] = []
    properties = ad.get("properties")
    if isinstance(properties, list):
        for item in properties:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            labels = [
                str(entry.get("label") or "").strip()
                for entry in (item.get("values") or [])
                if isinstance(entry, dict) and not is_blank(entry.get("label"))
            ]
            if not labels and not is_blank(item.get("value")):
                labels = [
                    part.strip()
                    for part in str(item.get("value")).split(",")
                    if part.strip()
                ]
            if name == "re_features":
                re_features.extend(labels)
            elif name == "re_complex_features":
                re_complex_features.extend(labels)

    return PageData(
        list_id=list_id,
        title=title,
        description=description,
        re_features=sorted(set(re_features)),
        re_complex_features=sorted(set(re_complex_features)),
    )


def context_for_match(text: str, start: int, end: int, before: int = 70, after: int = 45) -> str:
    return text[max(0, start - before) : min(len(text), end + after)]


def classify_text_term(text_norm: str, term_norm: str) -> tuple[list[str], list[str], list[str]]:
    positives: list[str] = []
    negatives: list[str] = []
    excluded: list[str] = []
    for match in re.finditer(rf"(?<!\w){re.escape(term_norm)}(?!\w)", text_norm):
        context = context_for_match(text_norm, match.start(), match.end())
        before = text_norm[max(0, match.start() - 70) : match.start()]
        negative = any(re.search(pattern + r"[^.!?;|]{0,35}$", before) for pattern in NEGATIVE_CONTEXTS)
        hypothetical = any(
            re.search(pattern + r"[^.!?;|]{0,45}$", before)
            for pattern in HYPOTHETICAL_OR_EXTERNAL_CONTEXTS
        )
        if negative:
            negatives.append(context)
        elif hypothetical:
            excluded.append(context)
        else:
            positives.append(context)
    return positives, negatives, excluded


def evaluate_flag(
    flag: str,
    existing_value: str,
    structured_labels: list[str],
    title: str,
    description: str,
) -> tuple[bool, str, str, bool]:
    existing = normalize_text(existing_value)
    structured_norm = normalize_text(" | ".join(structured_labels))
    free_text_norm = normalize_text(f"{title} | {description}")

    # Evita transformar "moveis planejados" em mobiliado.
    if flag == "tem_mobiliado":
        for phrase in MOBILIADO_FALSE_FRIENDS:
            free_text_norm = free_text_norm.replace(normalize_text(phrase), " ")

    structured_hits: list[str] = []
    text_positive: list[str] = []
    text_negative: list[str] = []
    text_excluded: list[str] = []

    for term in FEATURE_TERMS[flag]:
        norm_term = normalize_text(term)
        if re.search(rf"(?<!\w){re.escape(norm_term)}(?!\w)", structured_norm):
            structured_hits.append(term)
        positive, negative, excluded = classify_text_term(free_text_norm, norm_term)
        text_positive.extend(positive)
        text_negative.extend(negative)
        text_excluded.extend(excluded)

    list_positive = existing == "sim"
    list_negative = existing == "nao"
    page_structured_positive = bool(structured_hits)
    page_text_positive = bool(text_positive)
    page_negative = bool(text_negative)

    conflict = False
    if (list_positive or page_structured_positive or page_text_positive) and (list_negative or page_negative):
        conflict = True

    # Prioridade: evidencia estruturada positiva > positivo ja encontrado na
    # listagem > negacao explicita > positivo em descricao > ausencia.
    if page_structured_positive:
        value = True
        source = "pagina_estruturado"
    elif list_positive:
        value = True
        source = "listagem"
    elif page_negative or list_negative:
        value = False
        source = "negacao_explicita" if page_negative else "listagem_negacao"
    elif page_text_positive:
        value = True
        source = "pagina_descricao"
    else:
        value = False
        source = "ausente_na_pagina"

    evidence_parts: list[str] = []
    if structured_hits:
        evidence_parts.append("estruturado=" + "|".join(sorted(set(structured_hits))))
    if text_positive:
        evidence_parts.append("descricao=" + " || ".join(sorted(set(text_positive))[:3]))
    if text_negative:
        evidence_parts.append("negacao=" + " || ".join(sorted(set(text_negative))[:3]))
    if text_excluded:
        evidence_parts.append("ignorado=" + " || ".join(sorted(set(text_excluded))[:2]))
    if not evidence_parts:
        evidence_parts.append("nenhuma_mencao")

    return value, source, " ; ".join(evidence_parts)[:1200], conflict


def decode_bytes(data: bytes, encoding: Optional[str]) -> str:
    for candidate in (encoding, "utf-8", "latin-1"):
        if not candidate:
            continue
        try:
            return data.decode(candidate)
        except (LookupError, UnicodeDecodeError):
            continue
    return data.decode("utf-8", errors="replace")


def fetch_initial_fragment(
    session: requests.Session,
    url: str,
    max_bytes: int,
    connect_timeout: float,
    read_timeout: float,
) -> tuple[bytes, int]:
    marker_variants = (b'id="initial-data"', b"id='initial-data'")
    with session.get(
        url,
        stream=True,
        allow_redirects=True,
        timeout=(connect_timeout, read_timeout),
    ) as response:
        status = response.status_code
        buffer = bytearray()
        marker_index = -1
        for chunk in response.iter_content(chunk_size=16384):
            if not chunk:
                continue
            buffer.extend(chunk)
            lower = bytes(buffer).lower()
            if detect_block(decode_bytes(lower[:20000], response.encoding)):
                raise BlockedError(f"pagina de bloqueio; HTTP {status}")
            if marker_index < 0:
                indexes = [lower.find(marker) for marker in marker_variants]
                indexes = [index for index in indexes if index >= 0]
                if indexes:
                    marker_index = min(indexes)
            if marker_index >= 0 and lower.find(b"</script>", marker_index) >= 0:
                break
            if len(buffer) >= max_bytes:
                break

        data = bytes(buffer)
        text_sample = decode_bytes(data[:30000], response.encoding)
        block = detect_block(text_sample)
        if block or status in {403, 429}:
            raise BlockedError(f"{block or 'HTTP de bloqueio'}; HTTP {status}")
        if status in {404, 410}:
            raise FileNotFoundError(f"anuncio indisponivel; HTTP {status}")
        if status >= 500:
            raise requests.HTTPError(f"erro do servidor; HTTP {status}")
        if status >= 400:
            raise requests.HTTPError(f"HTTP {status}")
        return data, status


def build_session(user_agent: str) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.6",
            "Accept-Encoding": "gzip, deflate, br",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        }
    )
    return session


def round_robin_candidates(rows: list[dict[str, str]], completed: set[str]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str], deque[dict[str, str]]] = defaultdict(deque)
    seen: set[str] = set()
    for row in rows:
        ad_id = str(row.get("id_anuncio") or "").strip()
        url = str(row.get("url_anuncio") or "").strip()
        if not ad_id or not url or ad_id in completed or ad_id in seen:
            continue
        seen.add(ad_id)
        if not any(normalize_text(row.get(flag)) == "nao_informado" for flag in FLAGS):
            continue
        key = (
            str(row.get("cidade") or "").strip(),
            str(row.get("tipo_imovel") or "").strip(),
        )
        groups[key].append(row)

    result: list[dict[str, str]] = []
    active = sorted(groups)
    while active:
        next_active: list[tuple[str, str]] = []
        for key in active:
            if groups[key]:
                result.append(groups[key].popleft())
            if groups[key]:
                next_active.append(key)
        active = next_active
    return result


def process_local_html(paths: list[Path]) -> int:
    for path in paths:
        html = path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(html, "html.parser")
        tag = soup.find("script", id="initial-data")
        if tag is None or is_blank(tag.get("data-json")):
            print(f"{path.name}: initial-data ausente")
            continue
        data = json.loads(unescape(str(tag.get("data-json"))))
        ad_id = str((data.get("ad") or {}).get("listId") or "")
        parsed = parse_initial_data(html, ad_id)
        structured = parsed.re_features + parsed.re_complex_features
        print(f"\n{path.name} | ID {ad_id}")
        for flag in FLAGS:
            value, source, evidence, conflict = evaluate_flag(
                flag, "nao_informado", structured, parsed.title, parsed.description
            )
            print(f"  {flag}: {value} | {source} | conflito={conflict} | {evidence[:220]}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Enriquece flags da OLX lendo apenas script#initial-data do HTML inicial."
    )
    parser.add_argument("--input", help="dataset_modelagem_base_auditavel.csv")
    parser.add_argument("--output-dir", default="enriquecimento_flags_olx")
    parser.add_argument("--max-items", type=int, default=100)
    parser.add_argument("--min-sleep", type=float, default=8.0)
    parser.add_argument("--max-sleep", type=float, default=14.0)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--batch-pause", type=float, default=300.0)
    parser.add_argument("--max-bytes", type=int, default=262144)
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--read-timeout", type=float, default=25.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--retry-wait", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument(
        "--user-agent",
        default=(
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
        ),
    )
    parser.add_argument("--local-html", nargs="*", help="Valida parser sem rede")
    args = parser.parse_args()

    if args.local_html:
        return process_local_html([Path(item) for item in args.local_html])
    if not args.input:
        parser.error("--input e obrigatorio fora do modo --local-html")
    if args.max_items < 1:
        parser.error("--max-items deve ser >= 1")
    if args.min_sleep < 0 or args.max_sleep < args.min_sleep:
        parser.error("intervalo de sleep invalido")
    if args.max_bytes < 65536:
        parser.error("--max-bytes deve ser pelo menos 65536")

    random.seed(args.seed + int(time.time()))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "resultado_flags_olx.csv"
    log_path = output_dir / "log_flags_olx.csv"
    debug_dir = output_dir / "debug_html"

    rows = read_csv(Path(args.input))
    completed = successful_ids(result_path)
    candidates = round_robin_candidates(rows, completed)
    selected = candidates[: args.max_items]

    print(f"Registros na base: {len(rows)}")
    print(f"IDs ja concluidos: {len(completed)}")
    print(f"Pendentes com alguma flag nao informada: {len(candidates)}")
    print(f"Itens deste lote: {len(selected)}")
    if not selected:
        print("Nenhum item pendente neste lote.")
        return 0

    session = build_session(args.user_agent)
    successes = 0
    consecutive_errors = 0

    for index, row in enumerate(selected, start=1):
        ad_id = str(row.get("id_anuncio") or "").strip()
        url = canonicalize_url(str(row.get("url_anuncio") or ""))
        print(f"\n[{index}/{len(selected)}] ID {ad_id}")
        print(url)
        started = time.monotonic()
        status = "erro"
        http_status: Any = ""
        bytes_read = 0
        observation = ""
        raw_fragment = b""

        for attempt in range(1, args.retries + 2):
            try:
                raw_fragment, http_status = fetch_initial_fragment(
                    session=session,
                    url=url,
                    max_bytes=args.max_bytes,
                    connect_timeout=args.connect_timeout,
                    read_timeout=args.read_timeout,
                )
                bytes_read = len(raw_fragment)
                html = decode_bytes(raw_fragment, "utf-8")
                parsed = parse_initial_data(html, ad_id)
                structured = parsed.re_features + parsed.re_complex_features

                result: dict[str, Any] = {
                    "data_utc": utc_now_iso(),
                    "id_anuncio": ad_id,
                    "url_anuncio": url,
                    "status": "ok",
                    "http_status": http_status,
                    "list_id_extraido": parsed.list_id,
                    "bytes_lidos": bytes_read,
                    "tempo_segundos": round(time.monotonic() - started, 3),
                    "titulo_individual": parsed.title,
                    "descricao_tamanho": len(parsed.description),
                    "re_features_json": json.dumps(parsed.re_features, ensure_ascii=False),
                    "re_complex_features_json": json.dumps(parsed.re_complex_features, ensure_ascii=False),
                    "observacao": "",
                }
                for flag in FLAGS:
                    value, source, evidence, conflict = evaluate_flag(
                        flag=flag,
                        existing_value=str(row.get(flag) or "nao_informado"),
                        structured_labels=structured,
                        title=parsed.title,
                        description=parsed.description,
                    )
                    result[flag] = value
                    result[f"{flag}_fonte"] = source
                    result[f"{flag}_evidencia"] = evidence
                    result[f"{flag}_conflito"] = conflict

                append_csv(result_path, RESULT_COLUMNS, [result])
                status = "ok"
                observation = ""
                successes += 1
                consecutive_errors = 0
                true_flags = [flag for flag in FLAGS if result[flag] is True]
                print(f"OK | bytes lidos: {bytes_read} | True: {', '.join(true_flags) or '(nenhuma)'}")
                break

            except BlockedError as error:
                status = "bloqueado"
                observation = str(error)
                if raw_fragment:
                    path = save_debug_prefix(debug_dir, ad_id, "bloqueado", raw_fragment)
                    observation += f" | debug={path}"
                elapsed = round(time.monotonic() - started, 3)
                append_csv(
                    log_path,
                    LOG_COLUMNS,
                    [{
                        "data_utc": utc_now_iso(),
                        "id_anuncio": ad_id,
                        "url_anuncio": url,
                        "status": status,
                        "http_status": http_status,
                        "bytes_lidos": bytes_read,
                        "tempo_segundos": elapsed,
                        "observacao": observation,
                    }],
                )
                print(f"BLOQUEIO CONFIRMADO: {observation}")
                print("Execucao interrompida. Aguarde antes de retomar; nao aumente a agressividade.")
                return 2

            except FileNotFoundError as error:
                status = "indisponivel"
                observation = str(error)
                consecutive_errors = 0
                print(f"Indisponivel: {observation}")
                break

            except (requests.RequestException, ParseError, json.JSONDecodeError) as error:
                status = "erro"
                observation = f"{type(error).__name__}: {error}"
                if attempt <= args.retries:
                    wait = args.retry_wait * attempt
                    print(f"Tentativa {attempt} falhou: {observation}. Aguardando {wait:.0f}s.")
                    time.sleep(wait)
                    continue
                consecutive_errors += 1
                if raw_fragment:
                    path = save_debug_prefix(debug_dir, ad_id, "erro", raw_fragment)
                    observation += f" | debug={path}"
                print(f"Erro definitivo: {observation}")
                break

        elapsed = round(time.monotonic() - started, 3)
        append_csv(
            log_path,
            LOG_COLUMNS,
            [{
                "data_utc": utc_now_iso(),
                "id_anuncio": ad_id,
                "url_anuncio": url,
                "status": status,
                "http_status": http_status,
                "bytes_lidos": bytes_read,
                "tempo_segundos": elapsed,
                "observacao": observation,
            }],
        )

        if consecutive_errors >= 3:
            print("Tres erros consecutivos. Execucao interrompida para evitar insistencia improdutiva.")
            return 3

        if index < len(selected):
            if args.batch_size > 0 and successes > 0 and successes % args.batch_size == 0:
                print(f"Pausa de lote: {args.batch_pause:.0f}s")
                time.sleep(args.batch_pause)
            delay = random.uniform(args.min_sleep, args.max_sleep)
            print(f"Aguardando {delay:.1f}s...")
            time.sleep(delay)

    print("\nLote concluido.")
    print(f"Resultado: {result_path}")
    print(f"Log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
