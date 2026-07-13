import asyncio
import re
from urllib.parse import urljoin

import pandas as pd
from playwright.async_api import async_playwright, TimeoutError as PWTimeoutError
from tqdm import tqdm

BASE_URL = (
    "https://www.remax.com.br/listings?City=6578465%2C6578464%2C6578404%2C6578413"
    "%2C6578416%2C6578461%2C6578457&CityNM=6578465-Esp%C3%ADrito+Santo%2C+Vit%C3%B3ria"
    "%2C6578464-Esp%C3%ADrito+Santo%2C+Vila+Velha%2C6578404-Esp%C3%ADrito+Santo%2C+"
    "Cariacica%2C6578413-Esp%C3%ADrito+Santo%2C+Fund%C3%A3o%2C6578416-Esp%C3%ADrito+"
    "Santo%2C+Guarapari%2C6578461-Esp%C3%ADrito+Santo%2C+Viana%2C6578457-Esp%C3%ADrito+"
    "Santo%2C+Serra&ListingClass=-1&TransactionTypeUID=261&MacroPropertyTypeUIDs="
    "3237%2C2667%2C3245%2C3246%2C3243%2C3249%2C3247"
)

# ID da característica no site
FEATURE_MAP = {
    "tem_piscina": [22],
    "tem_varanda": [143, 319],
    "tem_lazer": [688],
    "tem_gourmet": [320],
    "tem_garden": [58],
    "tem_mobiliado": [16],
    "tem_elevador": [156],
    "tem_frente_mar": [12],
}

# rótulo do atributo dinâmico
ATRIBUTOS_MAP = {
    "Área M²": "area_m2",
    "Dormitórios": "quartos",
    "Banheiros": "banheiros",
    "Vagas de Estacionamento": "vagas",
    "Suítes": "tem_suite",
}

CARD_SELECTOR = '[data-testid^="listing-card-grid-item-"]'
NEXT_BUTTON_SELECTOR = '[data-testid="NavigateNextIcon"]'

COLUNAS_FINAIS = [
    "id_anuncio",
    "preco_venda",
    "tipo_imovel",
    "cidade",
    "bairro",
    "area_m2",
    "quartos",
    "banheiros",
    "vagas",
    "tem_suite",
    "tem_piscina",
    "tem_varanda",
    "tem_gourmet",
    "tem_lazer",
    "tem_mobiliado",
    "tem_elevador",
    "tem_cobertura",
    "tem_garden",
    "tem_frente_mar",
    "erro",
]

OUTPUT_CSV = "bases/imoveis_remax.csv"

# R$ 690.000' -> 690000 | '393 m²' -> 393 | '1,5' -> 1.5 | None se não achar.
def parse_numero(texto):
    if not texto:
        return None
    limpo = texto.replace(".", "").replace(",", ".")
    match = re.search(r"\d+\.\d+|\d+", limpo)
    if not match:
        return None
    valor = match.group()
    return float(valor) if "." in valor else int(valor)


# 'APARTAMENTO - VENDA - PRAIA DA COSTA' -> 'APARTAMENTO'
def extrair_tipo(titulo: str) -> str:
    if not titulo:
        return "N/A"
    partes = re.split(r"\s*-\s*", titulo.strip())
    return partes[0].strip() if partes else titulo.strip()


#Endereço não tem formato fixo, mas sempre termina em: ..., BAIRRO, CIDADE, ESTADO, CEP
#Quando o trecho do bairro vem com "algo - BAIRRO", ficamos só com o BAIRRO.
def extrair_bairro_cidade(endereco: str) -> tuple[str, str]:

    if not endereco or endereco == "N/A":
        return "N/A", "N/A"

    partes = [p.strip() for p in endereco.split(",")]
    if len(partes) < 3:
        return "N/A", "N/A"

    cidade = partes[-3]
    bairro_raw = partes[-4] if len(partes) >= 4 else ""
    bairro = bairro_raw.split(" - ")[-1].strip() if bairro_raw else ""

    return (bairro or "N/A"), (cidade or "N/A")


# Retorna a lista de URLs (absolutas) de cada card visível na página atual.
async def coletar_links_da_pagina(page) -> list[str]:
    cards = page.locator(CARD_SELECTOR)
    total = await cards.count()
    links = []

    for i in range(total):
        card = cards.nth(i)
        href = None
        try:
            # 1) o próprio card pode ser (ou conter) um <a>
            href = await card.get_attribute("href")
            if not href:
                anchor = card.locator("a").first
                if await anchor.count() > 0:
                    href = await anchor.get_attribute("href")
            if not href:
                # 2) fallback via JS: sobe até achar o <a> mais próximo
                el = await card.element_handle()
                if el:
                    href = await page.evaluate(
                        "(el) => el.closest('a') ? el.closest('a').getAttribute('href') : null",
                        el,
                    )
        except Exception:
            href = None

        if href:
            links.append(urljoin("https://www.remax.com.br", href))

    return links


"""
Varre as colunas listing-attributes-left-column / right-column.
Cada item (listing-attribute-{side}-{n}) tem um sub-elemento *-label-*
e um *-value-*. Usa JS para evitar ambiguidade entre o item "pai" e os
sub-elementos que também começam com o mesmo prefixo de data-testid.
"""
async def extrair_atributos_dinamicos(page) -> dict:
    script = """
    () => {
        const resultado = {};
        ['left', 'right'].forEach((side) => {
            const coluna = document.querySelector(`[data-testid="listing-attributes-${side}-column"]`);
            if (!coluna) return;
            const itemRegex = new RegExp(`^listing-attribute-${side}-\\\\d+$`);
            const todos = coluna.querySelectorAll('[data-testid]');
            todos.forEach((item) => {
                const testid = item.getAttribute('data-testid') || '';
                if (!itemRegex.test(testid)) return;
                const labelEl = item.querySelector('[data-testid*="label"]');
                const valueEl = item.querySelector('[data-testid*="value"]');
                if (labelEl && valueEl) {
                    const label = labelEl.textContent.trim();
                    const value = valueEl.textContent.trim();
                    if (label) resultado[label] = value;
                }
            });
        });
        return resultado;
    }
    """
    try:
        return await page.evaluate(script)
    except Exception:
        return {}


# Colunas booleanas padronizadas, cada uma pode ter mais de um ID de origem.
async def extrair_caracteristicas(page) -> dict:
    resultado = {}
    for nome_coluna, ids in FEATURE_MAP.items():
        seletor = ", ".join(
            f'[data-testid="listing-feature-left-item-{i}"], '
            f'[data-testid="listing-feature-right-item-{i}"]'
            for i in ids
        )
        try:
            resultado[nome_coluna] = await page.locator(seletor).count() > 0
        except Exception:
            resultado[nome_coluna] = False
    return resultado


# Abre o link do imóvel em uma nova aba, extrai os dados e fecha a aba.
async def extrair_dados_do_imovel(context, link: str) -> dict:
    page = await context.new_page()
    dados = {}

    try:
        await page.goto(link, wait_until="domcontentloaded", timeout=45000)
        await page.wait_for_timeout(1500)

        try:
            id_bruto = (await page.locator("#mls-id").inner_text()).strip()
            dados["id_anuncio"] = re.sub(r"(?i)^\s*id\s*:\s*", "", id_bruto).strip()
        except Exception:
            dados["id_anuncio"] = "N/A"

        try:
            preco_texto = (
                await page.locator('[data-testid="card-first-price"]').first.inner_text()
            ).strip()
        except Exception:
            preco_texto = None
        dados["preco_venda"] = parse_numero(preco_texto)

        try:
            titulo = await page.locator('[data-testid="listing-h1-title"]').inner_text()
            tipo_imovel = extrair_tipo(titulo)
        except Exception:
            tipo_imovel = "N/A"
        dados["tipo_imovel"] = tipo_imovel
        dados["tem_cobertura"] = "cobertura" in tipo_imovel.lower()

        try:
            endereco = (
                await page.locator("#listing-address p").first.inner_text()
            ).strip()
        except Exception:
            endereco = "N/A"
        dados["bairro"], dados["cidade"] = extrair_bairro_cidade(endereco)

        atributos = await extrair_atributos_dinamicos(page)
        for rotulo, coluna in ATRIBUTOS_MAP.items():
            dados[coluna] = parse_numero(atributos.get(rotulo))
        dados["tem_suite"] = dados["tem_suite"] or 0

        dados.update(await extrair_caracteristicas(page))

    except PWTimeoutError:
        dados["erro"] = "timeout ao carregar página do imóvel"
    except Exception as e:
        dados["erro"] = str(e)
    finally:
        await page.close()

    return dados


# Clica no botão de próxima página. Retorna False se não houver mais páginas.
async def ir_para_proxima_pagina(page) -> bool:
    botao = page.locator(NEXT_BUTTON_SELECTOR).locator("xpath=..")
    try:
        if await botao.count() == 0:
            return False
        # verifica se está desabilitado/oculto
        visivel = await botao.is_visible()
        habilitado = await botao.is_enabled()
        if not visivel or not habilitado:
            return False

        await botao.scroll_into_view_if_needed()
        await botao.click()
        await page.wait_for_load_state("networkidle", timeout=30000)
        await page.wait_for_timeout(2000)
        return True
    except Exception:
        return False


async def extrair_dados_completos():
    todos_imoveis = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1366, "height": 768})
        page = await context.new_page()

        print("Acessando listagem inicial...")
        await page.goto(BASE_URL, wait_until="load", timeout=60000)
        await page.wait_for_timeout(5000)

        pagina_atual = 1
        while True:
            print(f"\n--- Coletando links da página {pagina_atual} ---")
            links = await coletar_links_da_pagina(page)
            print(f"Página {pagina_atual}: {len(links)} imóveis encontrados.")

            barra = tqdm(
                links,
                desc=f"Página {pagina_atual}",
                unit="imóvel",
            )
            for indice, link in enumerate(barra, start=1):
                barra.set_description(
                    f"Página {pagina_atual} | imóvel {indice}/{len(links)}"
                )
                dados = await extrair_dados_do_imovel(context, link)
                todos_imoveis.append(dados)

            avancou = await ir_para_proxima_pagina(page)
            if not avancou:
                print("\nNão há mais páginas. Encerrando paginação.")
                break
            pagina_atual += 1

        await browser.close()

    if todos_imoveis:
        df = pd.DataFrame(todos_imoveis)
        colunas_existentes = [c for c in COLUNAS_FINAIS if c in df.columns]
        outras = [c for c in df.columns if c not in colunas_existentes]
        df = df[colunas_existentes + outras]
        df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        print(f"\nSucesso! {len(df)} imóveis salvos em '{OUTPUT_CSV}'.")
    else:
        print("\nNenhum imóvel foi capturado.")


asyncio.run(extrair_dados_completos())