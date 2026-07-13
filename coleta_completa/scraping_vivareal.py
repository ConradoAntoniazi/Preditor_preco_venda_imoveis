import asyncio
import json
import random
import re
import pandas as pd
from playwright.async_api import async_playwright
from tqdm import tqdm

HEADLESS = False
PAUSA_ENTRE_PAGINAS = (8, 15)
PAUSA_APOS_CARREGAR = (3, 6)
MAX_TENTATIVAS_POR_PAGINA = 3

AMENIDADES_PISCINA = {"Pool", "Adult Pool", "Childrens Pool"}
AMENIDADES_VARANDA = {"Balcony", "Wall Balcony", "Barbecue Balcony", "Gourmet Balcony"}
AMENIDADES_GOURMET = {"Gourmet Balcony", "Gourmet Kitchen", "Gourmet Space"}
AMENIDADES_LAZER = {"Recreation Area", "Party Hall", "Games Room", "Playground", "Sports Court", "Tennis Court", "Squash"}

COLUNAS_FINAIS = [
    "id_anuncio", "preco_venda", "tipo_imovel", "cidade", "bairro", "area_m2",
    "quartos", "banheiros", "vagas", "tem_suite", "tem_piscina", "tem_varanda",
    "tem_gourmet", "tem_lazer", "tem_mobiliado", "tem_elevador", "tem_cobertura",
    "tem_garden", "tem_frente_mar",
]

CARD_SELECTOR = '[data-cy="rp-property-cd"]'
VAGAS_SELECTOR = '[data-cy="rp-cardProperty-parkingSpacesQuantity-txt"]'

# URLS DE CADA CIDADE
URLS_ES = {
    "Vitoria": (
        "https://www.vivareal.com.br/venda/espirito-santo/vitoria/apartamento_residencial/"
        "?onde=%2CEsp%C3%ADrito+Santo%2CVit%C3%B3ria%2C%2C%2C%2C%2Ccity%2CBR%3EEspirito+Santo"
        "%3ENULL%3EVitoria%2C-20.319671%2C-40.338525%2C&tipos=apartamento_residencial%2C"
        "casa_residencial%2Ccondominio_residencial%2Ccobertura_residencial%2Cflat_residencial%2C"
        "kitnet_residencial%2Csobrado_residencial%2Cedificio-residencial_comercial"
    ),
    "Fundao": (
        "https://www.vivareal.com.br/venda/espirito-santo/fundao/apartamento_residencial/"
        "?onde=%2CEsp%C3%ADrito+Santo%2CFund%C3%A3o%2C%2C%2C%2C%2Ccity%2CBR%3EEspirito+Santo"
        "%3ENULL%3EFundao%2C-19.933243%2C-40.406666%2C&tipos=apartamento_residencial%2C"
        "casa_residencial%2Ccondominio_residencial%2Ccobertura_residencial%2Cflat_residencial%2C"
        "kitnet_residencial%2Csobrado_residencial%2Cedificio-residencial_comercial"
    ),
    "Cariacica": (
        "https://www.vivareal.com.br/venda/espirito-santo/cariacica/apartamento_residencial/"
        "?onde=%2CEsp%C3%ADrito+Santo%2CCariacica%2C%2C%2C%2C%2Ccity%2CBR%3EEspirito+Santo"
        "%3ENULL%3ECariacica%2C-20.264292%2C-40.420343%2C&tipos=apartamento_residencial%2C"
        "casa_residencial%2Ccondominio_residencial%2Ccobertura_residencial%2Cflat_residencial%2C"
        "kitnet_residencial%2Csobrado_residencial%2Cedificio-residencial_comercial"
    ),
    "Guarapari": (
        "https://www.vivareal.com.br/venda/espirito-santo/guarapari/apartamento_residencial/"
        "?onde=%2CEsp%C3%ADrito+Santo%2CGuarapari%2C%2C%2C%2C%2Ccity%2CBR%3EEspirito+Santo"
        "%3ENULL%3EGuarapari%2C-20.673893%2C-40.499984%2C&tipos=apartamento_residencial%2C"
        "casa_residencial%2Ccondominio_residencial%2Ccobertura_residencial%2Cflat_residencial%2C"
        "kitnet_residencial%2Csobrado_residencial%2Cedificio-residencial_comercial"
    ),
    "Viana": (
        "https://www.vivareal.com.br/venda/espirito-santo/viana/apartamento_residencial/"
        "?onde=%2CEsp%C3%ADrito+Santo%2CViana%2C%2C%2C%2C%2Ccity%2CBR%3EEspirito+Santo"
        "%3ENULL%3EViana%2C-20.366867%2C-40.421581%2C&tipos=apartamento_residencial%2C"
        "casa_residencial%2Ccondominio_residencial%2Ccobertura_residencial%2Cflat_residencial%2C"
        "kitnet_residencial%2Csobrado_residencial%2Cedificio-residencial_comercial"
    ),
    "Serra": (
        "https://www.vivareal.com.br/venda/espirito-santo/serra/apartamento_residencial/"
        "?onde=%2CEsp%C3%ADrito+Santo%2CSerra%2C%2C%2C%2C%2Ccity%2CBR%3EEspirito+Santo"
        "%3ENULL%3ESerra%2C-20.127987%2C-40.308105%2C&tipos=apartamento_residencial%2C"
        "casa_residencial%2Ccondominio_residencial%2Ccobertura_residencial%2Cflat_residencial%2C"
        "kitnet_residencial%2Csobrado_residencial%2Cedificio-residencial_comercial"
    ),
    "Vila Velha": (
        "https://www.vivareal.com.br/venda/espirito-santo/vila-velha/apartamento_residencial/"
        "?onde=%2CEsp%C3%ADrito+Santo%2CVila+Velha%2C%2C%2C%2C%2Ccity%2CBR%3EEspirito+Santo"
        "%3ENULL%3EVila+Velha%2C-20.347782%2C-40.294918%2C&tipos=apartamento_residencial%2C"
        "casa_residencial%2Ccondominio_residencial%2Ccobertura_residencial%2Cflat_residencial%2C"
        "kitnet_residencial%2Csobrado_residencial%2Cedificio-residencial_comercial"
    )
}

# FUNÇÕES AUXILIARES
def parse_numero(texto):
    if not texto:
        return None
    limpo = texto.replace(".", "").replace(",", ".")
    match = re.search(r"\d+\.\d+|\d+", limpo)
    if not match:
        return None
    valor = match.group()
    return float(valor) if "." in valor else int(valor)

def tem_alguma(amenidades: set, candidatas: set) -> bool:
    return bool(amenidades & candidatas)

def extrair_bairro(nome: str) -> str:
    if not nome or "," not in nome: return "N/A"
    antes = nome.rsplit(",", 1)[0]
    m = re.search(r"\bem\s+(.+)$", antes)
    return m.group(1).strip() if m else "N/A"

def extrair_tipo(nome: str) -> str:
    return nome.split(" para comprar")[0].strip() if nome else "N/A"

def extrair_vagas(nome: str) -> int:
    m = re.search(r"(\d+)\s+vagas?\b", nome or "", re.I)
    return int(m.group(1)) if m else 0

def extrair_suites(texto: str) -> int:
    if not texto: return 0
    m = re.search(r"(\d+)\s*su[ií]tes?", texto, re.I)
    if m: return int(m.group(1))
    return 1 if re.search(r"su[ií]te", texto, re.I) else 0

def tem_frente_mar(texto: str) -> bool:
    padrao = r"frente\s+(ao|pro|para o)?\s*mar|beira[- ]mar|vista\s+(para\s+o\s+)?mar"
    return bool(re.search(padrao, texto or "", re.I))

def extrair_item_list(scripts_json: list) -> list:
    for bruto in scripts_json:
        try: dados = json.loads(bruto)
        except json.JSONDecodeError: continue
        if dados.get("@type") != "ItemList": continue
        itens = dados.get("itemListElement", [])
        if itens and itens[0].get("item", {}).get("@type") == "Product":
            return [el["item"] for el in itens]
    return []

def montar_registro(item: dict) -> dict:
    nome = item.get("name", "") or ""
    descricao = item.get("description", "") or ""
    endereco = item.get("address", {}) or {}
    floor = item.get("floorSize", {}) or {}
    oferta = item.get("offers", {}) or {}
    amenidades = {a.get("value") for a in item.get("amenityFeature", [])}
    tipo_imovel = extrair_tipo(nome)

    return {
        "id_anuncio": item.get("@id"),
        "preco_venda": oferta.get("price"),
        "tipo_imovel": tipo_imovel,
        "cidade": endereco.get("addressLocality", "N/A"),
        "bairro": extrair_bairro(nome),
        "area_m2": floor.get("value"),
        "quartos": item.get("numberOfBedrooms"),
        "banheiros": item.get("numberOfBathroomsTotal"),
        "vagas": extrair_vagas(nome),
        "tem_suite": extrair_suites(descricao),
        "tem_piscina": tem_alguma(amenidades, AMENIDADES_PISCINA),
        "tem_varanda": tem_alguma(amenidades, AMENIDADES_VARANDA),
        "tem_gourmet": tem_alguma(amenidades, AMENIDADES_GOURMET),
        "tem_lazer": tem_alguma(amenidades, AMENIDADES_LAZER),
        "tem_mobiliado": "Furnished" in amenidades,
        "tem_elevador": "Elevator" in amenidades,
        "tem_cobertura": tipo_imovel.lower() == "cobertura",
        "tem_garden": "Garden" in amenidades,
        "tem_frente_mar": tem_frente_mar(descricao + " " + nome),
    }

def pagina_bloqueada(html: str) -> bool:
    marcadores = ["captcha", "access denied", "acesso negado", "request blocked", "unusual traffic", "tráfego incomum", "cloudflare"]
    trecho = html.lower()[:5000]
    return any(m in trecho for m in marcadores)

async def coletar_vagas_por_id(page) -> dict:
    mapa = {}
    cards = page.locator(CARD_SELECTOR)
    total = await cards.count()
    for i in range(total):
        card = cards.nth(i)
        try:
            href = await card.locator("a").first.get_attribute("href")
            m = re.search(r"id-(\d+)", href or "")
            if not m: continue
            vagas_el = card.locator(VAGAS_SELECTOR)
            if await vagas_el.count() > 0:
                texto = await vagas_el.inner_text()
                mapa[m.group(1)] = parse_numero(texto) or 0
        except Exception:
            continue
    return mapa

# FUNÇÃO PRINCIPAL
async def extrair_dados_completos():
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=HEADLESS,
            args=["--disable-blink-features=AutomationControlled", "--start-maximized"]
        )

        # Loop principal: Itera sobre cada cidade do dicionário
        for nome_cidade, url_base in URLS_ES.items():
            print(f"\n{'='*50}\nIniciando extração para: {nome_cidade.upper()}\n{'='*50}")
            
            # Cria um contexto limpo para cada cidade
            context = await browser.new_context(
                no_viewport=True,
                locale="pt-BR",
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                extra_http_headers={"Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8"},
            )
            page = await context.new_page()

            todos_imoveis_cidade = []
            primeiro_id_pagina_anterior = None
            pagina = 1

            while True:
                html = None
                for tentativa in range(1, MAX_TENTATIVAS_POR_PAGINA + 1):
                    url = f"{url_base}&pagina={pagina}"
                    await page.goto(url, wait_until="load", timeout=60000)
                    
                    # Simulação de comportamento humano
                    await page.mouse.wheel(0, 600)
                    await page.wait_for_timeout(1500)
                    await page.mouse.wheel(0, 1200)
                    
                    await page.wait_for_timeout(random.uniform(*PAUSA_APOS_CARREGAR) * 1000)
                    html = await page.content()

                    if not pagina_bloqueada(html):
                        break

                    espera = 30 * tentativa
                    print(f"\nPossível bloqueio detectado na página {pagina} de {nome_cidade}. "
                          f"Aguardando {espera}s antes de tentar de novo ({tentativa}/{MAX_TENTATIVAS_POR_PAGINA})...")

                    if tentativa == 1:
                        await page.screenshot(path=f"bloqueio_{nome_cidade}_pag{pagina}.png", full_page=True)
                        with open(f"bloqueio_{nome_cidade}_pag{pagina}.html", "w", encoding="utf-8") as f:
                            f.write(html)

                    await page.wait_for_timeout(espera * 1000)
                else:
                    print(f"\nBloqueio persistente em {nome_cidade} (Página {pagina}). Pulando para a próxima cidade.")
                    break

                scripts = await page.locator('script[type="application/ld+json"]').all_inner_texts()
                itens = extrair_item_list(scripts)

                if not itens and pagina == 1:
                    print(f"\nPágina 1 de {nome_cidade} vazia. Possível bloqueio silencioso. Pulando cidade.")
                    break

                if not itens or itens[0].get("@id") == primeiro_id_pagina_anterior:
                    print(f"\nFim da paginação em {nome_cidade} (Total: {pagina - 1} páginas lidas).")
                    break

                primeiro_id_pagina_anterior = itens[0].get("@id")
                vagas_por_id = await coletar_vagas_por_id(page)

                for item in tqdm(itens, desc=f"{nome_cidade} - Página {pagina}", unit="imóvel"):
                    registro = montar_registro(item)
                    vagas_confirmadas = vagas_por_id.get(registro["id_anuncio"])
                    if vagas_confirmadas is not None:
                        registro["vagas"] = vagas_confirmadas
                    # Força o nome da cidade iterada no dicionário para evitar inconsistências
                    registro["cidade_busca"] = nome_cidade 
                    todos_imoveis_cidade.append(registro)

                pagina += 1
                await page.wait_for_timeout(random.uniform(*PAUSA_ENTRE_PAGINAS) * 1000)

            # Fecha o contexto desta cidade para limpar os rastros antes da próxima
            await context.close()

            # Salva o CSV específico da cidade atual
            if todos_imoveis_cidade:
                df = pd.DataFrame(todos_imoveis_cidade)
                # Adiciona a nova coluna na lista de colunas finais, se necessário
                colunas_exportacao = [c for c in COLUNAS_FINAIS + ["cidade_busca"] if c in df.columns]
                df = df[colunas_exportacao]
                
                # Formata o nome do arquivo (ex: 'Vila Velha' vira 'imoveis_vivareal_vila_velha.csv')
                nome_arquivo = f"imoveis_vivareal_{nome_cidade.lower().replace(' ', '_')}.csv"
                df.to_csv(rf"bases/{nome_arquivo}", index=False, encoding="utf-8-sig")
                print(f"Sucesso! {len(df)} imóveis de {nome_cidade} salvos em '{nome_arquivo}'.")
            else:
                print(f"Nenhum imóvel capturado para {nome_cidade}.")

        # Fecha o navegador apenas quando todas as cidades terminarem
        await browser.close()
        print("\nExtração em lote finalizada com sucesso!")


asyncio.run(extrair_dados_completos())