.PHONY: install remax vivareal

install:
	pip install -r requirements.txt
	playwright install chromium

remax:
	python3 scraping_remax.py

vivareal:
	python3 scraping_vivareal.py