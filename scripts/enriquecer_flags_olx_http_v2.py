#!/usr/bin/env python3
"""Enriquece flags booleanas da OLX usando apenas os metadados iniciais da pagina.

Esta versao foi desenhada para a base consolidada deste projeto:
- re_features/re_complex_features ja foram usados para criar as flags em tres estados;
- a pagina individual e consultada somente para obter titulo/descricao completos;
- o HTML nao e renderizado e JavaScript nao e executado;
- a leitura para assim que os metadados necessarios aparecem;
- um False so e produzido quando a pagina correta foi validada e nenhuma evidencia
  positiva foi encontrada nas fontes disponiveis.

Semantica final:
True  = evidencia positiva na base de listagem, no titulo ou na descricao completa.
False = pagina individual validada, mas sem evidencia positiva, ou negacao explicita.
False nao garante ausencia fisica; significa "nao declarado nas fontes processadas".
"""

from __future__ import annotations

import argparse
import csv
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
    # Mantidas por compatibilidade com o finalizador anterior. Nesta versao,
    # os dados estruturados ja vieram da listagem e nao sao baixados novamente.
    "re_features_json",
    "re_complex_features_json",
    "observacao",
]
for _flag in FLAGS:
    RESULT_COLUMNS.extend([
        _flag,
        f"{_flag}_fonte",
        f"{_flag}_evidencia",
        f"{_flag}_conflito",
    ])

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

FEATURE_TERMS = {
    "tem_suite": ("suite", "suíte", "suites", "suítes"),
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
        "clube",
        "brinquedoteca",
        "quadra",
        "spa",
        "sauna",
        "espaco gourmet",
        "espaço gourmet",
    ),
    "tem_mobiliado": (
        "mobiliado",
        "mobiliada",
        "mobiliados",
        "mobiliadas",
        "porteira fechada",
        "com mobilia",
        "com mobília",
    ),
    "tem_elevador": ("elevador", "elevadores"),
}

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

MOBILIADO_FALSE_FRIENDS = (
    "moveis planejados",
    "móveis planejados",
    "armarios planejados",
    "armários planejados",
)

# Casos obvios que nao devem consumir requisicoes para enriquecimento de flags
# residenciais. Eles permanecem pendentes para revisao/retirada em outra etapa.
STRONG_OUT_OF_SCOPE_TITLE_PATTERNS = (
    r"^\s*lotes?\b",
    r"^\s*terrenos?\b",
    r"^\s*galp(?:a|ã)o\b",
    r"^\s*loja\b",
    r"^\s*sala\s+comercial\b",
    r"^\s*ponto\s+comercial\b",
    r"^\s*predio\s+comercial\b",
    r"^\s*prédio\s+comercial\b",
)

HARD_REVIEW_REASONS = {
    "titulo_sugere_fora_escopo_residencial",
    "titulo_menciona_locacao",
}


class BlockedError(RuntimeError):
    pass


class ParseError(RuntimeError):
    pass


@dataclass
class PageData:
    list_id: str
    title: str
    description: str
    parser_source: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "").lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", text).strip()


def canonicalize_url(url: str) -> str:
    parts = urlsplit(str(url or "").strip())
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


def successful_ids(path: Path) -> set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    return {
        str(row.get("id_anuncio") or "").strip()
        for row in read_csv(path)
        if row.get("status") == "ok" and str(row.get("id_anuncio") or "").strip()
    }


def save_debug(debug_dir: Path, ad_id: str, reason: str, data: bytes) -> Path:
    debug_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = debug_dir / f"{stamp}_{ad_id}_{reason}.html"
    path.write_bytes(data)
    return path


def detect_block(text: str) -> str:
    sample = normalize_text(text[:30000])
    for marker in BLOCK_MARKERS:
        if normalize_text(marker) in sample:
            return marker
    return ""


def decode_bytes(data: bytes, encoding: Optional[str]) -> str:
    for candidate in (encoding, "utf-8", "latin-1"):
        if not candidate:
            continue
        try:
            return data.decode(candidate)
        except (LookupError, UnicodeDecodeError):
            continue
    return data.decode("utf-8", errors="replace")


def meta_content(soup: BeautifulSoup, *, prop: str | None = None, name: str | None = None) -> str:
    if prop is not None:
        tag = soup.find("meta", attrs={"property": prop})
    else:
        tag = soup.find("meta", attrs={"name": name})
    return unescape(str(tag.get("content") or "")).strip() if tag else ""


def extract_ids_from_url_like(value: str) -> set[str]:
    value = unescape(str(value or ""))
    ids = set(re.findall(r"(?<!\d)(\d{9,12})(?!\d)", value))
    ids.update(re.findall(r"(?:[?&]|/)id[=/](\d{9,12})(?!\d)", value, flags=re.I))
    return ids


def parse_page_metadata(html: str, expected_id: str) -> PageData:
    soup = BeautifulSoup(html, "html.parser")

    url_like_values: list[str] = []
    canonical = soup.find("link", rel=lambda value: value and "canonical" in value)
    alternate = soup.find("link", rel=lambda value: value and "alternate" in value)
    if canonical:
        url_like_values.append(str(canonical.get("href") or ""))
    if alternate:
        url_like_values.append(str(alternate.get("href") or ""))
    for prop in ("og:url",):
        value = meta_content(soup, prop=prop)
        if value:
            url_like_values.append(value)
    for name in ("twitter:url",):
        value = meta_content(soup, name=name)
        if value:
            url_like_values.append(value)

    candidate_ids: set[str] = set()
    for value in url_like_values:
        candidate_ids.update(extract_ids_from_url_like(value))

    # Fallback: o dataLayer aparece muito cedo na pagina, mas nao precisamos
    # decodifica-lo nem reextrair suas features; apenas usamos list_id para validar.
    candidate_ids.update(re.findall(r'"(?:list_id|listId)"\s*:\s*"?(\d{9,12})"?', html[:131072]))

    expected = str(expected_id).strip()
    if expected not in candidate_ids:
        if candidate_ids:
            raise ParseError(
                f"ID divergente: esperado {expected}; encontrados {sorted(candidate_ids)[:8]}"
            )
        raise ParseError("nao foi possivel confirmar o ID do anuncio")

    title = (
        meta_content(soup, prop="og:title")
        or meta_content(soup, name="twitter:title")
        or (soup.title.get_text(" ", strip=True) if soup.title else "")
    )
    description = (
        meta_content(soup, prop="og:description")
        or meta_content(soup, name="twitter:description")
        or meta_content(soup, name="description")
    )
    description = BeautifulSoup(unescape(description), "html.parser").get_text(" ", strip=True)

    if not title and not description:
        raise ParseError("titulo e descricao ausentes nos metadados iniciais")

    return PageData(
        list_id=expected,
        title=title,
        description=description,
        parser_source="metadados_og",
    )


def fragment_has_complete_description(data_lower: bytes) -> bool:
    markers = (
        b'property="og:description"',
        b"property='og:description'",
        b'name="twitter:description"',
        b"name='twitter:description'",
    )
    positions = [data_lower.find(marker) for marker in markers]
    positions = [pos for pos in positions if pos >= 0]
    if not positions:
        return False
    start = min(positions)
    return data_lower.find(b">", start) >= 0


def fetch_metadata_fragment(
    session: requests.Session,
    url: str,
    expected_id: str,
    max_bytes: int,
    connect_timeout: float,
    read_timeout: float,
) -> tuple[bytes, int]:
    with session.get(
        url,
        stream=True,
        allow_redirects=True,
        timeout=(connect_timeout, read_timeout),
    ) as response:
        status = response.status_code
        buffer = bytearray()
        expected_bytes = str(expected_id).encode("ascii", errors="ignore")

        for chunk in response.iter_content(chunk_size=8192):
            if not chunk:
                continue
            buffer.extend(chunk)
            data = bytes(buffer)
            text_sample = decode_bytes(data[:30000], response.encoding)
            block = detect_block(text_sample)
            if block:
                raise BlockedError(f"{block}; HTTP {status}")

            lower = data.lower()
            if expected_bytes in data and fragment_has_complete_description(lower):
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


def context_for_match(text: str, start: int, end: int, before: int = 70, after: int = 55) -> str:
    return text[max(0, start - before): min(len(text), end + after)]


def classify_text_term(text_norm: str, term_norm: str) -> tuple[list[str], list[str], list[str]]:
    positives: list[str] = []
    negatives: list[str] = []
    excluded: list[str] = []
    for match in re.finditer(rf"(?<!\w){re.escape(term_norm)}(?!\w)", text_norm):
        context = context_for_match(text_norm, match.start(), match.end())
        before = text_norm[max(0, match.start() - 75): match.start()]
        negative = any(
            re.search(pattern + r"[^.!?;|]{0,40}$", before)
            for pattern in NEGATIVE_CONTEXTS
        )
        hypothetical = any(
            re.search(pattern + r"[^.!?;|]{0,50}$", before)
            for pattern in HYPOTHETICAL_OR_EXTERNAL_CONTEXTS
        )
        if negative:
            negatives.append(context)
        elif hypothetical:
            excluded.append(context)
        else:
            positives.append(context)
    return positives, negatives, excluded


def sanitize_evidence(text: str) -> str:
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[email]", text)
    text = re.sub(r"(?:\+?55\s*)?(?:\(?\d{2}\)?[\s.-]*)?(?:\d[\s.-]*){8,9}", "[telefone]", text)
    return text


def evaluate_flag(
    flag: str,
    existing_value: str,
    title: str,
    description: str,
) -> tuple[bool, str, str, bool]:
    existing = normalize_text(existing_value)
    free_text_norm = normalize_text(f"{title} | {description}")

    if flag == "tem_mobiliado":
        for phrase in MOBILIADO_FALSE_FRIENDS:
            free_text_norm = free_text_norm.replace(normalize_text(phrase), " ")

    positives: list[str] = []
    negatives: list[str] = []
    excluded: list[str] = []
    for term in FEATURE_TERMS[flag]:
        pos, neg, exc = classify_text_term(free_text_norm, normalize_text(term))
        positives.extend(pos)
        negatives.extend(neg)
        excluded.extend(exc)

    base_positive = existing == "sim"
    base_negative = existing == "nao"
    page_positive = bool(positives)
    page_negative = bool(negatives)
    conflict = (base_positive or page_positive) and (base_negative or page_negative)

    # A base preserva as tags estruturadas da listagem. Elas tem prioridade.
    if base_positive:
        value = True
        source = "listagem"
    elif base_negative:
        value = False
        source = "listagem_negacao"
    elif page_negative:
        value = False
        source = "pagina_negacao"
    elif page_positive:
        value = True
        source = "pagina_descricao"
    else:
        value = False
        source = "ausente_na_listagem_e_pagina"

    evidence_parts: list[str] = []
    if base_positive:
        evidence_parts.append("base=sim")
    elif base_negative:
        evidence_parts.append("base=nao")
    if positives:
        evidence_parts.append("descricao=" + " || ".join(sorted(set(positives))[:3]))
    if negatives:
        evidence_parts.append("negacao=" + " || ".join(sorted(set(negatives))[:3]))
    if excluded:
        evidence_parts.append("ignorado=" + " || ".join(sorted(set(excluded))[:2]))
    if not evidence_parts:
        evidence_parts.append("nenhuma_mencao")

    return value, source, sanitize_evidence(" ; ".join(evidence_parts))[:1200], conflict


def looks_obviously_out_of_scope(row: dict[str, str]) -> bool:
    reasons = {
        part.strip()
        for part in str(row.get("motivos_revisao") or "").split("|")
        if part.strip()
    }
    if reasons & HARD_REVIEW_REASONS:
        return True
    title = normalize_text(row.get("titulo_raw"))
    return any(re.search(pattern, title) for pattern in STRONG_OUT_OF_SCOPE_TITLE_PATTERNS)


def round_robin_candidates(
    rows: list[dict[str, str]],
    completed: set[str],
    include_suspected_out_of_scope: bool,
) -> tuple[list[dict[str, str]], int]:
    groups: dict[tuple[str, str], deque[dict[str, str]]] = defaultdict(deque)
    seen: set[str] = set()
    skipped_scope = 0
    for row in rows:
        ad_id = str(row.get("id_anuncio") or "").strip()
        url = str(row.get("url_anuncio") or "").strip()
        if not ad_id or not url or ad_id in completed or ad_id in seen:
            continue
        seen.add(ad_id)
        if not any(normalize_text(row.get(flag)) == "nao_informado" for flag in FLAGS):
            continue
        if not include_suspected_out_of_scope and looks_obviously_out_of_scope(row):
            skipped_scope += 1
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
    return result, skipped_scope


def build_session(user_agent: str) -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.6",
        # Evita depender de suporte extra a Brotli.
        "Accept-Encoding": "gzip, deflate",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "DNT": "1",
    })
    return session


def process_local_html(paths: list[Path]) -> int:
    for path in paths:
        html = path.read_text(encoding="utf-8", errors="replace")
        ids = sorted(set(re.findall(r'(?<!\d)(\d{9,12})(?!\d)', html[:30000])))
        expected = ""
        for candidate in ids:
            if candidate in path.name:
                expected = candidate
                break
        if not expected:
            # Busca preferencialmente canonical/list_id dentro dos primeiros KB.
            match = re.search(r'(?:list_id|listId|/imoveis/[^"\']*-)(?:"?\s*[:=/]\s*"?)?(\d{9,12})', html[:50000])
            expected = match.group(1) if match else (ids[-1] if ids else "")
        if not expected:
            print(f"{path.name}: nao foi possivel inferir o ID")
            continue
        parsed = parse_page_metadata(html, expected)
        print(f"\n{path.name} | ID {expected} | titulo={parsed.title[:100]}")
        for flag in FLAGS:
            value, source, evidence, conflict = evaluate_flag(
                flag, "nao_informado", parsed.title, parsed.description
            )
            print(f"  {flag}: {value} | {source} | conflito={conflict} | {evidence[:220]}")
    return 0


def write_run_report(
    path: Path,
    base_count: int,
    candidate_count: int,
    selected_count: int,
    skipped_scope: int,
    success_count: int,
    status_counts: dict[str, int],
    bytes_values: list[int],
) -> None:
    lines = [
        "RELATORIO DO LOTE DE ENRIQUECIMENTO DE FLAGS",
        "=" * 72,
        f"Registros na base: {base_count}",
        f"Pendentes elegiveis neste inicio de execucao: {candidate_count}",
        f"Suspeitos de fora do escopo pulados: {skipped_scope}",
        f"Itens selecionados no lote: {selected_count}",
        f"Sucessos no lote: {success_count}",
        "",
        "STATUS DO LOTE",
        "-" * 72,
    ]
    if status_counts:
        lines.extend(f"{key}: {value}" for key, value in sorted(status_counts.items()))
    else:
        lines.append("(nenhum item processado)")
    if bytes_values:
        lines.extend([
            "",
            f"Bytes lidos por pagina - minimo: {min(bytes_values)}",
            f"Bytes lidos por pagina - media: {round(sum(bytes_values) / len(bytes_values), 1)}",
            f"Bytes lidos por pagina - maximo: {max(bytes_values)}",
        ])
    lines.extend([
        "",
        "OBSERVACAO",
        "-" * 72,
        "As propriedades re_features/re_complex_features ja foram incorporadas na base.",
        "A pagina individual foi usada somente para validar o anuncio e obter a descricao completa.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Enriquece flags da OLX usando metadados OG iniciais e a base de listagem."
    )
    parser.add_argument("--input", help="dataset_modelagem_base_auditavel.csv")
    parser.add_argument("--output-dir", default="enriquecimento_flags_olx_v2")
    parser.add_argument("--max-items", type=int, default=50)
    parser.add_argument("--min-sleep", type=float, default=8.0)
    parser.add_argument("--max-sleep", type=float, default=12.0)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--batch-pause", type=float, default=180.0)
    parser.add_argument("--max-bytes", type=int, default=131072)
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--read-timeout", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--retry-wait", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument(
        "--include-suspected-out-of-scope",
        action="store_true",
        help="Inclui anuncios com titulo/revisao fortemente sugestivos de lote, terreno ou locacao.",
    )
    parser.add_argument(
        "--user-agent",
        default=(
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
        ),
    )
    parser.add_argument("--local-html", nargs="*", help="Valida o parser sem rede")
    args = parser.parse_args()

    if args.local_html:
        return process_local_html([Path(item) for item in args.local_html])
    if not args.input:
        parser.error("--input e obrigatorio fora do modo --local-html")
    if args.max_items < 1:
        parser.error("--max-items deve ser >= 1")
    if args.min_sleep < 0 or args.max_sleep < args.min_sleep:
        parser.error("intervalo de sleep invalido")
    if args.max_bytes < 32768:
        parser.error("--max-bytes deve ser pelo menos 32768")

    random.seed(args.seed + int(time.time()))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "resultado_flags_olx.csv"
    log_path = output_dir / "log_flags_olx.csv"
    report_path = output_dir / "relatorio_lote_flags.txt"
    debug_dir = output_dir / "debug_html"

    rows = read_csv(Path(args.input))
    completed = successful_ids(result_path)
    candidates, skipped_scope = round_robin_candidates(
        rows,
        completed,
        include_suspected_out_of_scope=args.include_suspected_out_of_scope,
    )
    selected = candidates[:args.max_items]

    print(f"Registros na base: {len(rows)}")
    print(f"IDs ja concluidos: {len(completed)}")
    print(f"Pendentes com alguma flag nao informada: {len(candidates)}")
    print(f"Suspeitos de fora do escopo pulados: {skipped_scope}")
    print(f"Itens deste lote: {len(selected)}")
    if not selected:
        print("Nenhum item pendente neste lote.")
        write_run_report(
            report_path, len(rows), len(candidates), 0, skipped_scope, 0, {}, []
        )
        return 0

    session = build_session(args.user_agent)
    successes = 0
    consecutive_errors = 0
    status_counts: dict[str, int] = defaultdict(int)
    bytes_values: list[int] = []

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
                raw_fragment, http_status = fetch_metadata_fragment(
                    session=session,
                    url=url,
                    expected_id=ad_id,
                    max_bytes=args.max_bytes,
                    connect_timeout=args.connect_timeout,
                    read_timeout=args.read_timeout,
                )
                bytes_read = len(raw_fragment)
                html = decode_bytes(raw_fragment, "utf-8")
                parsed = parse_page_metadata(html, ad_id)

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
                    "re_features_json": "[]",
                    "re_complex_features_json": "[]",
                    "observacao": (
                        "parser=metadados_og; re_features/re_complex_features "
                        "reaproveitados da base de listagem"
                    ),
                }
                for flag in FLAGS:
                    value, source, evidence, conflict = evaluate_flag(
                        flag=flag,
                        existing_value=str(row.get(flag) or "nao_informado"),
                        title=parsed.title,
                        description=parsed.description,
                    )
                    result[flag] = value
                    result[f"{flag}_fonte"] = source
                    result[f"{flag}_evidencia"] = evidence
                    result[f"{flag}_conflito"] = conflict

                append_csv(result_path, RESULT_COLUMNS, [result])
                status = "ok"
                observation = result["observacao"]
                successes += 1
                consecutive_errors = 0
                bytes_values.append(bytes_read)
                true_flags = [flag for flag in FLAGS if result[flag] is True]
                print(
                    f"OK | HTTP {http_status} | bytes lidos: {bytes_read} | "
                    f"True: {', '.join(true_flags) or '(nenhuma)'}"
                )
                break

            except BlockedError as error:
                status = "bloqueado"
                observation = str(error)
                if raw_fragment:
                    path = save_debug(debug_dir, ad_id, "bloqueado", raw_fragment)
                    observation += f" | debug={path}"
                elapsed = round(time.monotonic() - started, 3)
                append_csv(log_path, LOG_COLUMNS, [{
                    "data_utc": utc_now_iso(),
                    "id_anuncio": ad_id,
                    "url_anuncio": url,
                    "status": status,
                    "http_status": http_status,
                    "bytes_lidos": bytes_read,
                    "tempo_segundos": elapsed,
                    "observacao": observation,
                }])
                status_counts[status] += 1
                write_run_report(
                    report_path, len(rows), len(candidates), len(selected),
                    skipped_scope, successes, dict(status_counts), bytes_values,
                )
                print(f"BLOQUEIO CONFIRMADO: {observation}")
                print("Execucao interrompida. Aguarde antes de retomar.")
                return 2

            except FileNotFoundError as error:
                status = "indisponivel"
                observation = str(error)
                consecutive_errors = 0
                print(f"Indisponivel: {observation}")
                break

            except ParseError as error:
                # Repetir a mesma pagina nao corrige uma estrutura valida mas inesperada.
                status = "erro_parser"
                observation = f"ParseError: {error}"
                consecutive_errors += 1
                if raw_fragment:
                    path = save_debug(debug_dir, ad_id, "erro_parser", raw_fragment)
                    observation += f" | debug={path}"
                print(f"Erro de parser: {observation}")
                break

            except requests.RequestException as error:
                status = "erro_rede"
                observation = f"{type(error).__name__}: {error}"
                if attempt <= args.retries:
                    wait = args.retry_wait * attempt
                    print(f"Tentativa {attempt} falhou: {observation}. Aguardando {wait:.0f}s.")
                    time.sleep(wait)
                    continue
                consecutive_errors += 1
                if raw_fragment:
                    path = save_debug(debug_dir, ad_id, "erro_rede", raw_fragment)
                    observation += f" | debug={path}"
                print(f"Erro definitivo de rede: {observation}")
                break

        elapsed = round(time.monotonic() - started, 3)
        append_csv(log_path, LOG_COLUMNS, [{
            "data_utc": utc_now_iso(),
            "id_anuncio": ad_id,
            "url_anuncio": url,
            "status": status,
            "http_status": http_status,
            "bytes_lidos": bytes_read,
            "tempo_segundos": elapsed,
            "observacao": observation,
        }])
        status_counts[status] += 1

        if consecutive_errors >= 3:
            write_run_report(
                report_path, len(rows), len(candidates), len(selected),
                skipped_scope, successes, dict(status_counts), bytes_values,
            )
            print("Tres erros consecutivos. Execucao interrompida para diagnostico.")
            return 3

        if index < len(selected):
            if args.batch_size > 0 and successes > 0 and successes % args.batch_size == 0:
                print(f"Pausa de lote: {args.batch_pause:.0f}s")
                time.sleep(args.batch_pause)
            delay = random.uniform(args.min_sleep, args.max_sleep)
            print(f"Aguardando {delay:.1f}s...")
            time.sleep(delay)

    write_run_report(
        report_path, len(rows), len(candidates), len(selected),
        skipped_scope, successes, dict(status_counts), bytes_values,
    )
    print("\nLote concluido.")
    print(f"Resultado: {result_path}")
    print(f"Log: {log_path}")
    print(f"Relatorio: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
