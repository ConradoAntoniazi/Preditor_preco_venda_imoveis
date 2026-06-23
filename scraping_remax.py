# "https://www.remax.com.br/listings?City=6578465%2C6578464%2C6578404%2C6578413%2C6578416%2C6578461%2C6578457&CityNM=6578465-Esp%C3%ADrito+Santo%2C+Vit%C3%B3ria%2C6578464-Esp%C3%ADrito+Santo%2C+Vila+Velha%2C6578404-Esp%C3%ADrito+Santo%2C+Cariacica%2C6578413-Esp%C3%ADrito+Santo%2C+Fund%C3%A3o%2C6578416-Esp%C3%ADrito+Santo%2C+Guarapari%2C6578461-Esp%C3%ADrito+Santo%2C+Viana%2C6578457-Esp%C3%ADrito+Santo%2C+Serra&ListingClass=-1&TransactionTypeUID=261&MacroPropertyTypeUIDs=3237%2C2667%2C3245%2C3246%2C3243%2C3249%2C3247%2C3248%2C3240"

import asyncio
import re
import pandas as pd
from playwright.async_api import async_playwright

def extrair_apenas_numeros(texto):
    if not texto or texto == "N/A":
        return "N/A"
    
    texto_limpo = texto.replace(".", "")
    texto_limpo = texto_limpo.replace(",", ".")

    numeros = re.findall(r'\d+\.\d+|\d+', texto_limpo)
    
    if numeros:
        valor = numeros[0]
        return float(valor) if "." in valor else int(valor)
    
    return "N/A"

async def extrair_dados_completos():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(viewport={"width": 1280, "height": 720})
        page = await context.new_page()
        
        url_busca = "https://www.remax.com.br/listings?City=6578465%2C6578464%2C6578404%2C6578413%2C6578416%2C6578461%2C6578457&CityNM=6578465-Esp%C3%ADrito+Santo%2C+Vit%C3%B3ria%2C6578464-Esp%C3%ADrito+Santo%2C+Vila+Velha%2C6578404-Esp%C3%ADrito+Santo%2C+Cariacica%2C6578413-Esp%C3%ADrito+Santo%2C+Fund%C3%A3o%2C6578416-Esp%C3%ADrito+Santo%2C+Guarapari%2C6578461-Esp%C3%ADrito+Santo%2C+Viana%2C6578457-Esp%C3%ADrito+Santo%2C+Serra&ListingClass=-1&TransactionTypeUID=261&MacroPropertyTypeUIDs=3237%2C2667%2C3245%2C3246%2C3243%2C3249%2C3247%2C3248%2C3240"
        await page.goto(url_busca, wait_until="load")
        await page.wait_for_timeout(20000)
        
        todos_imoveis = []
        id_contador = 1
        
        for pagina in range(1, 17):
            print(f"--- Processando Página {pagina} de 16 ---")
            
            cards = await page.locator('[data-testid^="listing-card-grid-item-"]').all()
            
            for card in cards:
                try:
                    preco = await card.locator('[data-testid="card-first-price"]').inner_text()
                    tipo = await card.locator('[data-testid="listing-card-property-type"]').inner_text()
                    endereco = await card.locator('[data-testid="listing-card-address"]').inner_text()
                    
                    quartos_raw = await card.locator('[aria-label*="Dormitórios"]').get_attribute("aria-label") if await card.locator('[aria-label*="Dormitórios"]').count() > 0 else "N/A"
                    banheiros_raw = await card.locator('[aria-label*="Banheiros"]').get_attribute("aria-label") if await card.locator('[aria-label*="Banheiros"]').count() > 0 else "N/A"
                    comodos_raw = await card.locator('[aria-label*="Ambientes Totais"]').get_attribute("aria-label") if await card.locator('[aria-label*="Ambientes Totais"]').count() > 0 else "N/A"
                    area_raw = await card.locator('[aria-label*="Área M²"]').get_attribute("aria-label") if await card.locator('[aria-label*="Área M²"]').count() > 0 else "N/A"
                    vagas_raw = await card.locator('[aria-label*="Vagas de Estacionamento"]').get_attribute("aria-label") if await card.locator('[aria-label*="Vagas de Estacionamento"]').count() > 0 else "N/A"
                    
                    imovel = {
                        "id": f"anuncio_{id_contador}",
                        "tipo": tipo.strip(),
                        "preco": preco.strip(),
                        "endereco": endereco.strip(),
                        "quartos": extrair_apenas_numeros(quartos_raw),
                        "banheiros": extrair_apenas_numeros(banheiros_raw),
                        "comodos": extrair_apenas_numeros(comodos_raw),
                        "area": extrair_apenas_numeros(area_raw),
                        "vagas": extrair_apenas_numeros(vagas_raw)
                    }
                    
                    todos_imoveis.append(imovel)
                    id_contador += 1
                    
                except Exception as e:
                    continue
            
            if pagina < 16:
                try:
                    botao_proximo = page.locator('[data-testid="NavigateNextIcon"]').locator('xpath=..')
                    
                    if await botao_proximo.count() > 0:
                        await botao_proximo.click()
                        await page.wait_for_load_state("networkidle")
                        await page.wait_for_timeout(3000)
                    else:
                        print("Botão próximo não encontrado. Encerrando paginação.")
                        break
                except Exception as error:
                    print(f"Erro ao tentar ir para a próxima página: {error}")
                    break
                
        await browser.close()
        
        if todos_imoveis:
            df = pd.DataFrame(todos_imoveis)
            df.to_csv("imoveis_remax.csv", index=False, encoding="utf-8-sig")
            print(f"\nSucesso! {len(df)} imóveis salvos em 'imoveis_remax.csv'.")
        else:
            print("\nNenhum imóvel foi capturado.")

asyncio.run(extrair_dados_completos())