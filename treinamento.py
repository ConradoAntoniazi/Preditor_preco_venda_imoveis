import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score

from utils import preparar_dados

def main():
    print("1. Recebendo os dados espaciais do utils.py...")
    
    df_treino, df_teste = preparar_dados()
    
    # Separar os Atributos (X) e o Alvo que queremos prever (y)
    X_train = df_treino.drop(columns=['preco_venda', 'id_anuncio'], errors='ignore')
    y_train = df_treino['preco_venda']
    
    X_test = df_teste.drop(columns=['preco_venda', 'id_anuncio'], errors='ignore')
    y_test = df_teste['preco_venda']
    
    print("2. Configurando o Pipeline de Pré-processamento com Memória...")
    # Definir que colunas sofrem qual tipo de transformação
    colunas_categoricas = ['cidade', 'tipo_imovel']
    colunas_numericas = ['area_m2', 'quartos', 'banheiros', 'vagas', 'score_vizinhanca', 'preco_m2_medio_bairro', 'area_por_quarto']
    
    # Detecta automaticamente todas as colunas booleanas (tem_piscina, tem_suite, etc.)
    colunas_booleanas = [col for col in X_train.columns if col.startswith('tem_')]
    
    preprocessor = ColumnTransformer(
        transformers=[
            # 1. O OneHotEncoder agora tem memória das cidades e tipos de imóveis
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), colunas_categoricas),
            
            # 2. O StandardScaler normaliza as grandezas (Crucial para a Rede Neural e KNN)
            ('num', StandardScaler(), colunas_numericas + colunas_booleanas)
        ])
    
    print("3. Inicializando os Modelos Base...")
    rf_model = RandomForestRegressor(n_estimators=1000, max_depth=25, min_samples_split=5, min_samples_leaf=2, random_state=42)
    xgb_model = XGBRegressor(
        n_estimators=500, 
        learning_rate=0.05, 
        max_depth=7, 
        subsample=0.8, 
        colsample_bytree=0.8, 
        random_state=42
    )
    mlp_model = MLPRegressor(hidden_layer_sizes=(256, 128), alpha=0.01, max_iter=2500, early_stopping=True, random_state=42)
    knn_model = KNeighborsRegressor(n_neighbors=8, weights='distance', p=1)
    ada_model = AdaBoostRegressor(n_estimators=500, learning_rate=0.1, random_state=42)

    estimadores_base = [
        ('rf', rf_model),
        ('xgb', xgb_model),
        ('mlp', mlp_model),
        ('knn', knn_model),
        ('ada', ada_model)
    ]
    
    print("4. Construindo o Stacking Regressor (Meta-Modelo)...")
    stacking_model = StackingRegressor(
        estimators=estimadores_base,
        final_estimator=LinearRegression(),
        cv=5,
        n_jobs=-1
    )

    stacking_com_log = TransformedTargetRegressor(
        regressor=stacking_model,
        func=np.log1p,        
        inverse_func=np.expm1
    )
    
    pipeline_final = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('stacking', stacking_com_log)
    ])
    
    print("5. Treinando a solução unificada (Pode demorar alguns minutos)...")
    pipeline_final.fit(X_train, y_train)
    
    print("6. Avaliando o modelo no conjunto de teste...")
    previsoes = pipeline_final.predict(X_test)
    
    erro_percentual = mean_absolute_percentage_error(y_test, previsoes) * 100
    r2 = r2_score(y_test, previsoes)
    
    print("-" * 30)
    print("Resultados Finais na Validação:")
    print(f"R² Score (Precisão): {r2:.4f}")
    print(f"Erro Percentual Médio (MAPE): {erro_percentual:.2f}%")
    print("-" * 30)
    
    print("7. Exportando o modelo para inferência...")
    joblib.dump(pipeline_final, 'modelo_stacking_final.pkl')
    print("Sucesso! O modelo 'modelo_stacking_final.pkl' guardou a memória do treino e está pronto.")

if __name__ == "__main__":
    main()