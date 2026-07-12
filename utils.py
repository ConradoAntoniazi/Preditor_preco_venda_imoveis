import time
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from geopy.geocoders import Nominatim
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

NUM_NEIGHBORS = 5

def le_dados():
    df_imoveis = pd.read_csv("./saidas/dataset_modelagem.csv")
    df_imoveis = df_imoveis.drop(columns=["tem_sol_manha", "tem_frente_mar", "area_suspeita", "preco_suspeito", "idade_anuncio_dias", "preco_m2", "preco_m2_suspeito"], errors='ignore')
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

    df_treino, df_teste = train_test_split(
        df_limpo, 
        test_size=0.2, 
        random_state=42, 
        stratify=df_limpo['cidade']
    )
    return df_treino.copy(), df_teste.copy()

def treina_modelo_vizinhanca(df_treino):
    radar_knn = NearestNeighbors(n_neighbors=NUM_NEIGHBORS + 1, algorithm='auto', metric='euclidean')
    radar_knn.fit(df_treino[['latitude', 'longitude']])
    return radar_knn

def aplicar_score_knn(df_treino, df_teste, radar_knn):
    """Calcula e aplica o score de vizinhança aos datasets."""
    # Score do Treino
    precos_treino = df_treino['preco_venda'].to_numpy()

    dist_treino, ind_treino = radar_knn.kneighbors(df_treino[['latitude', 'longitude']])
    df_treino['score_vizinhanca'] = [
        precos_treino[[v for v in vizinhos if v != i][:NUM_NEIGHBORS]].mean()
        for i, vizinhos in enumerate(ind_treino)
    ]

    # Score do Teste
    dist_teste, ind_teste = radar_knn.kneighbors(df_teste[['latitude', 'longitude']])
    df_teste['score_vizinhanca'] = [
        precos_treino[vizinhos[:NUM_NEIGHBORS]].mean() for vizinhos in ind_teste
    ]

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