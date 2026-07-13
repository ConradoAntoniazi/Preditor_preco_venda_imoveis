.DEFAULT_GOAL := help

.PHONY: help install dirs check coleta remax vivareal prepara_dataset verifica-dados treinar inferir inferir-estrito pipeline full clean

PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

MODELO ?= modelo_segmentado_final.joblib
ENTRADA ?= novos_imoveis.csv
SAIDA ?= previsoes_inferencia.csv

install:
	$(PIP) install -r requirements.txt
	$(PYTHON) -m playwright install chromium

dirs:
	mkdir -p bases dataset saidas

check:
	$(PYTHON) -m compileall -q coleta_completa scripts prepara_dataset.py utils.py treinamento.py inferencia.py

remax: dirs
	$(PYTHON) coleta_completa/scraping_remax.py

vivareal: dirs
	$(PYTHON) coleta_completa/scraping_vivareal.py

coleta: remax vivareal

prepara_dataset: dirs
	$(PYTHON) prepara_dataset.py

verifica-dados:
	@test -f dataset/dataset_modelagem_outros.csv || (echo "Erro: falta dataset/dataset_modelagem_outros.csv"; exit 1)
	@test -f dataset/dataset_modelagem_olx.csv || (echo "Erro: falta dataset/dataset_modelagem_olx.csv"; exit 1)

# O arquivo saidas/long_lat.csv e opcional: utils.py o gera quando necessario.
treinar: dirs verifica-dados
	$(PYTHON) treinamento.py

inferir:
	$(PYTHON) inferencia.py \
		--entrada "$(ENTRADA)" \
		--modelo "$(MODELO)" \
		--saida "$(SAIDA)"

inferir-estrito:
	$(PYTHON) inferencia.py \
		--entrada "$(ENTRADA)" \
		--modelo "$(MODELO)" \
		--saida "$(SAIDA)" \
		--estrito

pipeline: prepara_dataset treinar

full: install coleta prepara_dataset treinar

clean:
	rm -f modelo_segmentado_final.joblib
	rm -f saidas/relatorio_modelo_final.json
	rm -f saidas/previsoes_teste_final.csv
	rm -f previsoes_inferencia.csv
