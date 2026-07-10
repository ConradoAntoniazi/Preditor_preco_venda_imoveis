import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from geopy.geocoders import Nominatim
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

NUM_NEIGHBORS = 5

def le_dados():
    df_imoveis = pd.read_csv("./saidas/dataset_modelagem.csv")
    df_imoveis = df_imoveis.drop(columns=["tem_sol_manha","tem_frente_mar","area_suspeita", "preco_suspeito", "idade_anuncio_dias", "preco_m2", "preco_m2_suspeito"])
    return df_imoveis

def gera_score_vizinhanca(df_imoveis):
    geolocator = Nominatim(user_agent="app_localizacao")
    df_long_lat = df_imoveis[["bairro", "cidade"]].drop_duplicates().reset_index(drop=True)

    for index, row in tqdm(df_long_lat.iterrows(), total=df_long_lat.shape[0]):
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

        time.sleep(2)
    df_long_lat.to_csv("./saidas/long_lat.csv", index=False)

def particiona_dataset(df_imoveis):
    # Limpar nulos que vão quebrar o KNN
    df_limpo = df_imoveis.dropna(subset=['latitude', 'longitude', 'preco_venda']).copy()
    
    # Boa prática para tirar warnings
    df_treino, df_teste = train_test_split(df_limpo, test_size=0.2, random_state=42)
    return df_treino.copy(), df_teste.copy()

def treina_modelo_vizinhanca(df_treino):
    radar_knn = NearestNeighbors(n_neighbors=NUM_NEIGHBORS + 1, algorithm='auto', metric='euclidean')
    radar_knn.fit(df_treino[['latitude', 'longitude']])
    return radar_knn


# FLUXO PRINCIPAL ==============================================================================================

df_imoveis = le_dados()
df_long_lat = pd.read_csv("./saidas/long_lat.csv")
df_imoveis = df_imoveis.merge(df_long_lat, on=["bairro", "cidade"], how="left")

# Aplicando One-Hot Encoding na cidade 
df_imoveis = pd.get_dummies(df_imoveis, columns=['cidade'], dtype=int) # Remove automaticamente a coluna string 'cidade' original

df_treino, df_teste = particiona_dataset(df_imoveis)
radar_knn = treina_modelo_vizinhanca(df_treino)

# Gerando Score para o TREINO - Descartando o próprio imóvel para não enviesar a média
dist_treino, ind_treino = radar_knn.kneighbors(df_treino[['latitude', 'longitude']])
scores_treino = []
for vizinhos in ind_treino:
    # vizinhos[1:] -> Pega do 2º até o ultimo (total = NUM_NEIGHBORS vizinhos)
    media = df_treino.iloc[vizinhos[1:]]['preco_venda'].mean() 
    scores_treino.append(media)
df_treino['score_vizinhanca'] = scores_treino

# Gerando Score para o TESTE 
dist_teste, ind_teste = radar_knn.kneighbors(df_teste[['latitude', 'longitude']])
scores_teste = []
for vizinhos in ind_teste:
    # vizinhos[:NUM_NEIGHBORS] -> Pega os NUM_NEIGHBORS primeiros (pois o imóvel de teste não está no radar)
    media = df_treino.iloc[vizinhos[:NUM_NEIGHBORS]]['preco_venda'].mean()
    scores_teste.append(media)
df_teste['score_vizinhanca'] = scores_teste

# Removendo colunas que não serão usadas no modelo
colunas_para_remover = ['bairro', 'latitude', 'longitude']
df_treino = df_treino.drop(columns=colunas_para_remover)
df_teste = df_teste.drop(columns=colunas_para_remover)

pd.set_option('display.max_columns', None)
print("Treino:")
print(df_treino.head(10))
print("=====================================================================")
print("Teste:")
print(df_teste.head(10))

# DAQUI EM DIANTE TEMOS XGBOOST COM K-FOLD =====================================================================