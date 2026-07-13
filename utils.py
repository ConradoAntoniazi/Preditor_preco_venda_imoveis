import os
import time
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from geopy.geocoders import Nominatim
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

NUM_NEIGHBORS = 8
PASTA_SAIDAS = 'saidas'

def tratar_outliers_preco_m2(df, limite_inferior=0.01, limite_superior=0.99):
    """
    Remove linhas com preço/m² fora dos percentis definidos, calculados
    SEPARADAMENTE por tipo_imovel.

    Por que por tipo: casas e apartamentos têm distribuições de preço/m²
    bem diferentes (casas tendem a ter preço/m² mais baixo por causa do
    terreno, que não entra em area_m2; apartamentos tendem a ter preço/m²
    mais alto). Um corte de percentil global mistura as duas distribuições:
    ou deixa passar erros de digitação num segmento, ou descarta variação
    legítima do outro. Esses casos costumam ser erro de digitação (ex: área
    ou preço trocado) em vez de imóveis genuinamente atípicos, e distorcem
    tanto o treino dos modelos quanto o score_vizinhanca e o
    preco_m2_medio_bairro, que dependem diretamente do preço por m².
    """
    preco_m2 = df['preco_venda'] / df['area_m2']
    mascara_valida = pd.Series(True, index=df.index)

    for tipo, grupo in df.groupby('tipo_imovel'):
        preco_m2_tipo = preco_m2.loc[grupo.index]
        p_inf, p_sup = preco_m2_tipo.quantile([limite_inferior, limite_superior])
        mascara_tipo = preco_m2_tipo.between(p_inf, p_sup)
        mascara_valida.loc[grupo.index] = mascara_tipo
        print(f"Outliers de preço/m² removidos ({tipo}): {(~mascara_tipo).sum()} "
              f"(fora da faixa R$ {p_inf:.0f} - R$ {p_sup:.0f} por m²)")

    print(f"Outliers de preço/m² removidos (total): {(~mascara_valida).sum()}")

    return df[mascara_valida].reset_index(drop=True)

def le_dados():
    df1 = pd.read_csv(os.path.join('saidas', 'dataset_modelagem.csv'))
    df2 = pd.read_csv(os.path.join('old_dataset_limpeza_duplicatas', 'dataset_final_modelagem.csv'))

    colunas_comuns = [coluna for coluna in df1.columns if coluna in df2.columns]

    df1_filtrado = df1[colunas_comuns]
    df2_filtrado = df2[colunas_comuns]
    
    df_combinado = pd.concat([df1_filtrado, df2_filtrado], ignore_index=True)

    df_combinado['tipo_imovel'] = df_combinado['tipo_imovel'].str.strip().str.lower()

    mapa_tipo = {
        'cobertura': 'apartamento',
        'flat': 'apartamento',
        'studio': 'apartamento',
        'sobrado': 'casa',
        'duplex': 'casa',
        'térrea': 'casa',
        'casa de condomínio': 'casa',
    }

    df_combinado['tipo_imovel'] = df_combinado['tipo_imovel'].replace(mapa_tipo)

    df_combinado['vagas'] = df_combinado['vagas'].fillna(0)
    df_combinado['banheiros'] = df_combinado['banheiros'].fillna(
        df_combinado.groupby('tipo_imovel')['banheiros'].transform('median')
    )
    df_combinado['quartos'] = df_combinado['quartos'].fillna(
        df_combinado.groupby('tipo_imovel')['quartos'].transform('median')
    )

    linhas_antes = len(df_combinado)
    df_combinado = df_combinado.drop_duplicates().reset_index(drop=True)
    print(f"Duplicatas removidas: {linhas_antes - len(df_combinado)}")

    df_combinado = tratar_outliers_preco_m2(df_combinado)

    os.makedirs(PASTA_SAIDAS, exist_ok=True)
    df_combinado.to_csv('dataset_combinado.csv', index=False)

    return df_combinado

def gera_score_vizinhanca(df_imoveis):
    geolocator = Nominatim(user_agent="app_localizacao")
    df_long_lat = df_imoveis[["bairro", "cidade"]].drop_duplicates().reset_index(drop=True)

    for index, row in tqdm(df_long_lat.iterrows(), total=df_long_lat.shape[0], desc="Geocodificando"):
        endereco_busca = f"{row['bairro']}, {row['cidade']}, ES, Brasil"
        try:
            location = geolocator.geocode(endereco_busca, timeout=10)
            if location:
                df_long_lat.at[index, 'latitude'] = location.latitude
                df_long_lat.at[index, 'longitude'] = location.longitude
            else:
                df_long_lat.at[index, 'latitude'] = None
                df_long_lat.at[index, 'longitude'] = None
        except Exception as e:
            print(f"Erro ao buscar {endereco_busca}: {e}")
            df_long_lat.at[index, 'latitude'] = None
            df_long_lat.at[index, 'longitude'] = None
        time.sleep(1.5)
        
    df_long_lat.to_csv("./saidas/long_lat.csv", index=False)

def particiona_dataset(df_imoveis):
    total_antes = len(df_imoveis)
    df_limpo = df_imoveis.dropna(subset=['latitude', 'longitude', 'preco_venda']).copy()
    perdidos = total_antes - len(df_limpo)
    if perdidos > 0:
        pct = 100 * perdidos / total_antes
        print(f"Aviso: {perdidos} linhas ({pct:.1f}%) descartadas por falha "
              f"de geocodificação (bairro não encontrado pelo Nominatim).")
    
    faixas_preco = pd.qcut(df_limpo['preco_venda'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
    
    df_limpo['estrato'] = df_limpo['cidade'].astype(str) + "_" + \
                          df_limpo['tipo_imovel'].astype(str) + "_" + \
                          faixas_preco.astype(str)

    contagem = df_limpo['estrato'].value_counts()
    raros = contagem[contagem < 2].index
    df_limpo.loc[df_limpo['estrato'].isin(raros), 'estrato'] = 'Outros'

    df_treino, df_teste = train_test_split(
        df_limpo, 
        test_size=0.2, 
        random_state=42,
        stratify=df_limpo['estrato']
    )
    
    df_treino = df_treino.drop(columns=['estrato'])
    df_teste = df_teste.drop(columns=['estrato'])
    
    return df_treino.copy(), df_teste.copy()

def treina_modelos_vizinhanca_por_tipo(df_treino):
    """
    Treina um KNN espacial POR tipo_imovel (em vez de um único KNN misturando
    casas e apartamentos). Como a geocodificação é feita por bairro (não por
    endereço exato), os "vizinhos" do KNN nada mais são do que outros imóveis
    do mesmo bairro -- então, sem separar por tipo, um bairro com poucas casas
    e muitos apartamentos faz o score da casa ser calculado majoritariamente
    a partir de preços de apartamentos, com dinâmica de preço diferente.
    """
    modelos = {}
    for tipo in df_treino['tipo_imovel'].unique():
        subset = df_treino[df_treino['tipo_imovel'] == tipo]
        n_vizinhos = min(NUM_NEIGHBORS + 1, len(subset))
        radar_knn = NearestNeighbors(n_neighbors=n_vizinhos, algorithm='auto', metric='euclidean')
        radar_knn.fit(subset[['latitude', 'longitude']])
        modelos[tipo] = radar_knn
    return modelos

def aplicar_score_knn_por_tipo(df_treino, df_teste, modelos_knn, epsilon=1e-6, m_smoothing=5):
    """
    Calcula o score de vizinhança separadamente por tipo_imovel e aplica
    suavização bayesiana em direção à média de preço do próprio tipo.

    Por que a suavização importa: como a geocodificação é por bairro, o score
    "bruto" é essencialmente a média ponderada do preço de outros imóveis do
    MESMO bairro (e agora também do mesmo tipo) no treino. Em bairros com
    poucas casas, essa média é calculada com base em 1-3 vizinhos, o que faz
    o score praticamente refletir o próprio preço do imóvel -- um vazamento
    de dados que deixa o modelo "decorar" o treino e generalizar mal (o gap
    grande entre MAPE treino/teste observado nas casas). A suavização
    (m_smoothing) puxa o score em direção à média do tipo quando há poucos
    vizinhos confiáveis, reduzindo esse vazamento.
    """
    df_treino = df_treino.copy()
    df_teste = df_teste.copy()
    df_treino['score_vizinhanca'] = np.nan
    df_teste['score_vizinhanca'] = np.nan

    for tipo, radar_knn in modelos_knn.items():
        sub_treino = df_treino[df_treino['tipo_imovel'] == tipo]
        precos_treino_tipo = sub_treino['preco_venda'].to_numpy()
        media_global_tipo = precos_treino_tipo.mean()
        k_efetivo = min(NUM_NEIGHBORS, len(sub_treino) - 1)

        if k_efetivo < 1:
            df_treino.loc[sub_treino.index, 'score_vizinhanca'] = media_global_tipo
        else:
            dist_treino, ind_treino = radar_knn.kneighbors(sub_treino[['latitude', 'longitude']])
            scores = []
            for i, (vizinhos, distancias) in enumerate(zip(ind_treino, dist_treino)):
                pares = [(v, d) for v, d in zip(vizinhos, distancias) if v != i][:k_efetivo]
                vizinhos_validos = [v for v, d in pares]
                distancias_validas = np.array([d for v, d in pares])

                pesos = 1.0 / (distancias_validas + epsilon)
                pesos /= pesos.sum()

                score_bruto = np.sum(precos_treino_tipo[vizinhos_validos] * pesos)
                n_efetivo = len(vizinhos_validos)
                score_suavizado = (n_efetivo * score_bruto + m_smoothing * media_global_tipo) / (n_efetivo + m_smoothing)
                scores.append(score_suavizado)
            df_treino.loc[sub_treino.index, 'score_vizinhanca'] = scores

        sub_teste = df_teste[df_teste['tipo_imovel'] == tipo]
        if len(sub_teste) == 0:
            continue

        if k_efetivo < 1:
            df_teste.loc[sub_teste.index, 'score_vizinhanca'] = media_global_tipo
        else:
            dist_teste, ind_teste = radar_knn.kneighbors(sub_teste[['latitude', 'longitude']])
            scores_teste = []
            for vizinhos, distancias in zip(ind_teste, dist_teste):
                vizinhos = vizinhos[:k_efetivo]
                distancias = distancias[:k_efetivo]

                pesos = 1.0 / (distancias + epsilon)
                pesos /= pesos.sum()

                score_bruto = np.sum(precos_treino_tipo[vizinhos] * pesos)
                score_suavizado = (k_efetivo * score_bruto + m_smoothing * media_global_tipo) / (k_efetivo + m_smoothing)
                scores_teste.append(score_suavizado)
            df_teste.loc[sub_teste.index, 'score_vizinhanca'] = scores_teste

    return df_treino, df_teste

def calcular_preco_m2_bairro(df_treino, df_teste, m_smoothing=15):
    """
    Calcula o preço médio por m² por bairro E por tipo_imovel (com
    suavização bayesiana), usando apenas o treino, e aplica em treino e
    teste.

    Por que separar por tipo: antes, essa média era calculada por bairro
    misturando casas e apartamentos. Como apartamentos são ~2/3 dos dados,
    a média de preço/m² de cada bairro ficava dominada pelo padrão de
    apartamento (que costuma ter preço/m² mais alto), e essa mesma média
    "contaminada" era aplicada às casas do bairro -- exatamente o mesmo
    tipo de distorção que já tínhamos corrigido no score_vizinhanca.
    """
    df_treino = df_treino.copy()
    df_teste = df_teste.copy()

    df_treino['preco_m2'] = df_treino['preco_venda'] / df_treino['area_m2']

    media_global_por_tipo = df_treino.groupby('tipo_imovel')['preco_m2'].mean()

    stats = df_treino.groupby(['bairro', 'tipo_imovel'])['preco_m2'].agg(['mean', 'count']).reset_index()
    stats['media_global_tipo'] = stats['tipo_imovel'].map(media_global_por_tipo)
    stats['preco_m2_medio_bairro'] = (
        (stats['count'] * stats['mean'] + m_smoothing * stats['media_global_tipo'])
        / (stats['count'] + m_smoothing)
    )

    mapa_bairro_tipo = stats.set_index(['bairro', 'tipo_imovel'])['preco_m2_medio_bairro'].to_dict()

    def aplica_mapa(df):
        chaves = list(zip(df['bairro'], df['tipo_imovel']))
        return [mapa_bairro_tipo.get(chave) for chave in chaves]

    df_treino['preco_m2_medio_bairro'] = aplica_mapa(df_treino)
    df_teste['preco_m2_medio_bairro'] = aplica_mapa(df_teste)

    df_treino['preco_m2_medio_bairro'] = df_treino['preco_m2_medio_bairro'].fillna(
        df_treino['tipo_imovel'].map(media_global_por_tipo))
    df_teste['preco_m2_medio_bairro'] = df_teste['preco_m2_medio_bairro'].fillna(
        df_teste['tipo_imovel'].map(media_global_por_tipo))

    df_treino = df_treino.drop(columns=['preco_m2'])

    return df_treino, df_teste

def adicionar_features_derivadas(df_treino, df_teste):
    """Cria features derivadas simples (razões entre variáveis existentes)."""
    for df in (df_treino, df_teste):
        df['area_por_quarto'] = df['area_m2'] / (df['quartos'] + 1)

    return df_treino, df_teste

def preparar_dados():
    """Função Principal: Orquestra a leitura, cálculo e retorna os dados prontos para os modelos."""
    df_imoveis = le_dados()
    
    try:
        df_long_lat = pd.read_csv("./saidas/long_lat.csv")
    except FileNotFoundError:
        print("Arquivo long_lat.csv não encontrado. Gerando coordenadas...")
        gera_score_vizinhanca(df_imoveis)
        df_long_lat = pd.read_csv("./saidas/long_lat.csv")
        
    df_imoveis = df_imoveis.merge(df_long_lat, on=["bairro", "cidade"], how="left")
    
    df_treino, df_teste = particiona_dataset(df_imoveis)
    
    modelos_knn = treina_modelos_vizinhanca_por_tipo(df_treino)
    df_treino, df_teste = aplicar_score_knn_por_tipo(df_treino, df_teste, modelos_knn)
    df_treino, df_teste = calcular_preco_m2_bairro(df_treino, df_teste)
    df_treino, df_teste = adicionar_features_derivadas(df_treino, df_teste)
    
    os.makedirs(PASTA_SAIDAS, exist_ok=True)
    joblib.dump(modelos_knn, './saidas/modelos_knn_vizinhanca.pkl')
    df_treino[['tipo_imovel', 'latitude', 'longitude', 'preco_venda']].to_csv('./saidas/referencia_treino_knn.csv', index=False)
    
    colunas_para_remover = ['bairro', 'latitude', 'longitude']
    df_treino = df_treino.drop(columns=colunas_para_remover)
    df_teste = df_teste.drop(columns=colunas_para_remover)
    
    return df_treino, df_teste

if __name__ == "__main__":
    print("Iniciando processamento dos dados e do score de vizinhança...")
    treino, teste = preparar_dados()
    pd.set_option('display.max_columns', None)
    print("\nTREINO:")
    print(treino.head(5))
    print("\nTESTE:")
    print(teste.head(5))
    print("Módulos exportados e dados prontos com sucesso!")