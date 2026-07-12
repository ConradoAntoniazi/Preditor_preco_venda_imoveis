import time
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from geopy.geocoders import Nominatim
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

NUM_NEIGHBORS = 8

def le_dados():
    df1 = pd.read_csv('saidas/dataset_modelagem.csv')
    df2 = pd.read_csv('dataset_final_modelagem.csv')

    colunas_comuns = [coluna for coluna in df1.columns if coluna in df2.columns]

    df1_filtrado = df1[colunas_comuns]
    df2_filtrado = df2[colunas_comuns]
    
    df_combinado = pd.concat([df1_filtrado, df2_filtrado], ignore_index=True)
    df_combinado_preenchido = df_combinado.fillna(0)
    df_combinado_preenchido.to_csv('dataset_combinado.csv', index=False)

    df_imoveis = pd.read_csv("dataset_combinado.csv")
    return df_imoveis

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
    df_limpo = df_imoveis.dropna(subset=['latitude', 'longitude', 'preco_venda']).copy()
    
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

def treina_modelo_vizinhanca(df_treino):
    radar_knn = NearestNeighbors(n_neighbors=NUM_NEIGHBORS + 1, algorithm='auto', metric='euclidean')
    radar_knn.fit(df_treino[['latitude', 'longitude']])
    return radar_knn

def aplicar_score_knn(df_treino, df_teste, radar_knn, epsilon=1e-6):
    """Calcula e aplica o score de vizinhança (ponderado pela distância) aos datasets."""
    precos_treino = df_treino['preco_venda'].to_numpy()

    # --- Score do Treino ---
    dist_treino, ind_treino = radar_knn.kneighbors(df_treino[['latitude', 'longitude']])

    scores_treino = []
    for i, (vizinhos, distancias) in enumerate(zip(ind_treino, dist_treino)):
        # remove o próprio imóvel da lista de vizinhos
        pares = [(v, d) for v, d in zip(vizinhos, distancias) if v != i][:NUM_NEIGHBORS]
        vizinhos_validos = [v for v, d in pares]
        distancias_validas = np.array([d for v, d in pares])

        pesos = 1.0 / (distancias_validas + epsilon)
        pesos /= pesos.sum()

        score = np.sum(precos_treino[vizinhos_validos] * pesos)
        scores_treino.append(score)

    df_treino['score_vizinhanca'] = scores_treino

    # --- Score do Teste ---
    dist_teste, ind_teste = radar_knn.kneighbors(df_teste[['latitude', 'longitude']])

    scores_teste = []
    for vizinhos, distancias in zip(ind_teste, dist_teste):
        vizinhos = vizinhos[:NUM_NEIGHBORS]
        distancias = distancias[:NUM_NEIGHBORS]

        pesos = 1.0 / (distancias + epsilon)
        pesos /= pesos.sum()

        score = np.sum(precos_treino[vizinhos] * pesos)
        scores_teste.append(score)

    df_teste['score_vizinhanca'] = scores_teste

    return df_treino, df_teste

def calcular_preco_m2_bairro(df_treino, df_teste, m_smoothing=15):
    """
    Calcula o preço médio por m² por bairro (com suavização bayesiana)
    usando apenas o treino, e aplica em treino e teste.
    """
    df_treino = df_treino.copy()
    df_teste = df_teste.copy()

    df_treino['preco_m2'] = df_treino['preco_venda'] / df_treino['area_m2']

    media_global = df_treino['preco_m2'].mean()

    stats_bairro = df_treino.groupby('bairro')['preco_m2'].agg(['mean', 'count'])

    stats_bairro['preco_m2_medio_bairro'] = (
        (stats_bairro['count'] * stats_bairro['mean'] + m_smoothing * media_global)
        / (stats_bairro['count'] + m_smoothing)
    )

    mapa_bairro = stats_bairro['preco_m2_medio_bairro'].to_dict()

    df_treino['preco_m2_medio_bairro'] = df_treino['bairro'].map(mapa_bairro)
    df_teste['preco_m2_medio_bairro'] = df_teste['bairro'].map(mapa_bairro).fillna(media_global)

    df_treino = df_treino.drop(columns=['preco_m2'])

    return df_treino, df_teste

def adicionar_features_derivadas(df_treino, df_teste):
    """Cria features derivadas simples (razões entre variáveis existentes)."""
    for df in (df_treino, df_teste):
        # +1 no denominador evita divisão por zero em imóveis com 0 quartos (studios/kitnets)
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
    
    # Particiona os dados
    df_treino, df_teste = particiona_dataset(df_imoveis)
    
    # Treina o modelo KNN e aplica os scores espaciais
    radar_knn = treina_modelo_vizinhanca(df_treino)
    df_treino, df_teste = aplicar_score_knn(df_treino, df_teste, radar_knn)
    df_treino, df_teste = calcular_preco_m2_bairro(df_treino, df_teste)
    df_treino, df_teste = adicionar_features_derivadas(df_treino, df_teste)
    
    # Exporta os arquivos necessários para o inferencia.py 
    joblib.dump(radar_knn, './saidas/modelo_knn_vizinhanca.pkl')
    df_treino[['latitude', 'longitude', 'preco_venda']].to_csv('./saidas/referencia_treino_knn.csv', index=False)
    
    # Remove colunas que já não são úteis para o Machine Learning
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