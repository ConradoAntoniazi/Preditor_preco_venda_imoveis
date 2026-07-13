## Execução do projeto

### Requisitos

O projeto utiliza `make` para centralizar os comandos de instalação, coleta, preparação dos dados, treinamento e inferência.

No Ubuntu, instale o `make` caso ele ainda não esteja disponível:

```bash
sudo apt update
sudo apt install make
```

Os arquivos principais devem permanecer na raiz do projeto:

```text
projeto/
├── Makefile
├── requirements.txt
├── scraping_remax.py
├── scraping_vivareal.py
├── prepara_dataset.py
├── treinamento.py
├── inferencia.py
├── utils.py
├── dataset_final_modelagem.csv
└── saidas/
    ├── dataset_modelagem.csv
    └── long_lat.csv
```

### Comandos disponíveis

Para consultar os comandos:

```bash
make help
```

### Instalação

Instale as dependências Python e o Chromium usado pelo Playwright:

```bash
make install
```

### Coleta dos dados

Executar somente o scraper da Remax:

```bash
make remax
```

Executar somente o scraper do Viva Real:

```bash
make vivareal
```

Executar os dois scrapers:

```bash
make coleta
```

### Preparação do dataset

```bash
make prepara_dataset
```

### Treinamento

```bash
make treinar
```

Ao final, serão gerados:

```text
modelo_segmentado_final.joblib
saidas/relatorio_modelo_final.json
saidas/previsoes_teste_final.csv
```

Para preparar o dataset e treinar em sequência:

```bash
make pipeline
```

Para executar todo o fluxo, incluindo instalação e coleta:

```bash
make full
```

### Verificação de sintaxe

Antes de executar o pipeline completo, é possível verificar a sintaxe dos arquivos principais:

```bash
make check
```

### Inferência

O CSV de entrada deve conter:

```text
tipo_imovel
cidade
bairro
area_m2
quartos
banheiros
vagas
tem_suite
tem_piscina
tem_varanda
tem_lazer
tem_mobiliado
tem_elevador
```

Executar a inferência com os nomes padrão:

```bash
make inferir
```

Os padrões são:

```text
ENTRADA=novos_imoveis.csv
MODELO=modelo_segmentado_final.joblib
SAIDA=previsoes_inferencia.csv
```

Para usar outros arquivos:

```bash
make inferir \
  ENTRADA=dados_para_prever.csv \
  MODELO=modelo_segmentado_final.joblib \
  SAIDA=resultado.csv
```

Para interromper a execução quando houver área ausente ou fora da faixa aceita:

```bash
make inferir-estrito \
  ENTRADA=dados_para_prever.csv \
  SAIDA=resultado.csv
```

O CSV de saída preserva as colunas originais e adiciona:

```text
preco_venda_estimado
```

Os valores booleanos podem ser informados como `True`/`False`, `1`/`0` ou `sim`/`não`.

### Limpeza dos artefatos gerados

O comando abaixo remove somente o modelo e os relatórios produzidos pelo treinamento:

```bash
make clean
```

Os datasets coletados e preparados não são removidos.