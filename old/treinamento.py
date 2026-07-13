import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import mean_absolute_percentage_error, r2_score, make_scorer
from xgboost import XGBRegressor

from utils import preparar_dados

LIMIAR_SEGMENTO_PEQUENO = 6000

def constroi_pipeline_busca(colunas_categoricas, colunas_numericas, colunas_booleanas):
    """Pipeline completo (preprocessor + XGBoost com alvo em log1p) usado
    dentro do RandomizedSearchCV -- assim o preprocessor é ajustado
    separadamente em cada fold da CV, sem vazar informação entre folds."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), colunas_categoricas),
            ('num', StandardScaler(), colunas_numericas + colunas_booleanas)
        ])

    xgb_model = XGBRegressor(random_state=42, n_jobs=1)

    modelo = TransformedTargetRegressor(
        regressor=xgb_model,
        func=np.log1p,
        inverse_func=np.expm1
    )

    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('modelo', modelo)
    ])


def espaco_de_busca():
    """Espaço de hiperparâmetros do XGBoost a ser explorado pela CV.
    Cobre desde configurações bem regularizadas/rasas (para segmentos
    pequenos, propensos a overfitting) até configurações mais permissivas."""
    return {
        'modelo__regressor__max_depth': [3, 4, 5, 6, 8],
        'modelo__regressor__learning_rate': [0.01, 0.02, 0.03, 0.05, 0.08],
        'modelo__regressor__n_estimators': [150, 300, 450, 600, 900, 1200],
        'modelo__regressor__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'modelo__regressor__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'modelo__regressor__reg_alpha': [0, 0.1, 0.5, 1, 2, 5],
        'modelo__regressor__reg_lambda': [0.5, 1, 2, 3, 5, 10],
        'modelo__regressor__min_child_weight': [1, 3, 5, 8, 12],
    }

def busca_hiperparametros(X_tr, y_tr, colunas_categoricas, colunas_numericas, colunas_booleanas, n_iter, seed=42):
    """Busca aleatória com validação cruzada de 5 folds, otimizando o MAPE
    real (não uma proxy em espaço log) -- isso evita escolher hiperparâmetros
    que parecem bons no MAE em log1p mas não se traduzem em MAPE melhor."""
    pipeline = constroi_pipeline_busca(colunas_categoricas, colunas_numericas, colunas_booleanas)

    mape_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)

    busca = RandomizedSearchCV(
        pipeline,
        param_distributions=espaco_de_busca(),
        n_iter=n_iter,
        scoring=mape_scorer,
        cv=cv,
        random_state=seed,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    busca.fit(X_tr, y_tr)
    return busca

def percentual_dentro_da_faixa(y_true, y_pred, tolerancia=0.15):
    erro_percentual = np.abs((y_true - y_pred) / y_true)
    return (erro_percentual <= tolerancia).mean() * 100

def imprime_metricas(nome, y_true, y_pred, mape_cv=None):
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    mdape = np.median(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    pct_ok = percentual_dentro_da_faixa(y_true, y_pred, tolerancia=0.15)

    print(f"\n--- {nome} ---")
    print(f"R² Score: {r2:.4f}")
    print(f"MAPE (erro médio): {mape:.2f}%")
    print(f"MdAPE (erro mediano): {mdape:.2f}%")
    print(f"% de imóveis com erro <= 15%: {pct_ok:.1f}%")

    if mape_cv is not None:
        gap = mape - mape_cv
        print(f"MAPE CV (5-fold, sem ver o teste): {mape_cv:.2f}%  |  MAPE teste: {mape:.2f}%  (diferença: {gap:+.1f} pts)")
        if abs(gap) > 5:
            print("Aviso: teste bem diferente da estimativa por CV -- pode indicar mudança de "
                  "distribuição entre treino e teste, não overfitting de hiperparâmetros.")
        else:
            print("Estimativa por CV bate com o teste: o modelo está generalizando de forma consistente.")

def main():
    print("1. Recebendo os dados espaciais do utils.py...")
    df_treino, df_teste = preparar_dados()

    X_train_full = df_treino.drop(columns=['preco_venda', 'id_anuncio'], errors='ignore')
    y_train_full = df_treino['preco_venda']

    X_test_full = df_teste.drop(columns=['preco_venda', 'id_anuncio'], errors='ignore')
    y_test_full = df_teste['preco_venda']

    colunas_numericas = ['area_m2', 'quartos', 'banheiros', 'vagas', 'score_vizinhanca',
                          'preco_m2_medio_bairro', 'area_por_quarto']
    colunas_booleanas = [col for col in X_train_full.columns if col.startswith('tem_')]

    colunas_categoricas_segmento = ['cidade']

    tipos = df_treino['tipo_imovel'].unique()
    print(f"2. Segmentos detectados: {list(tipos)}")

    modelos = {}
    previsoes_teste_geral = pd.Series(index=X_test_full.index, dtype=float)
    mape_cv_por_segmento = {}
    tamanho_por_segmento = {}

    for tipo in tipos:
        print(f"\n3. Treinando modelo especializado para '{tipo}'...")

        mask_train = X_train_full['tipo_imovel'] == tipo
        mask_test = X_test_full['tipo_imovel'] == tipo

        X_tr = X_train_full[mask_train].drop(columns=['tipo_imovel'])
        y_tr = y_train_full[mask_train]

        X_te = X_test_full[mask_test].drop(columns=['tipo_imovel'])
        y_te = y_test_full[mask_test]

        print(f"   Amostras -> treino: {len(X_tr)} | teste: {len(X_te)}")

        n_iter = 40 if len(X_tr) < LIMIAR_SEGMENTO_PEQUENO else 25
        print(f"   Buscando hiperparâmetros com CV de 5 folds (n_iter={n_iter})...")

        busca = busca_hiperparametros(
            X_tr, y_tr, colunas_categoricas_segmento, colunas_numericas, colunas_booleanas, n_iter=n_iter
        )
        pipeline = busca.best_estimator_
        mape_cv = -busca.best_score_ * 100

        print(f"   Melhores hiperparâmetros: { {k.split('__')[-1]: v for k, v in busca.best_params_.items()} }")
        print(f"   MAPE médio na CV (5-fold, sem ver o teste): {mape_cv:.2f}%")

        pred_te = pipeline.predict(X_te)
        previsoes_teste_geral.loc[X_te.index] = pred_te

        imprime_metricas(f"Segmento: {tipo}", y_te, pred_te, mape_cv=mape_cv)

        modelos[tipo] = pipeline
        mape_cv_por_segmento[tipo] = mape_cv
        tamanho_por_segmento[tipo] = len(X_tr)

    print("\n" + "=" * 30)
    print("RESULTADO GERAL (todos os segmentos combinados)")
    print("=" * 30)
    total_amostras = sum(tamanho_por_segmento.values())
    mape_cv_geral = sum(
        mape_cv_por_segmento[tipo] * tamanho_por_segmento[tipo] for tipo in tipos
    ) / total_amostras
    imprime_metricas(
        "Geral (segmentado)",
        y_test_full, previsoes_teste_geral.values,
        mape_cv=mape_cv_geral
    )

    #print("\n7. Exportando os modelos segmentados para inferência...")
    #joblib.dump(modelos, 'modelos_segmentados_por_tipo.pkl')
    #print("Sucesso! 'modelos_segmentados_por_tipo.pkl' contém um pipeline por tipo_imovel.")
    #print("Na inferência: escolha o pipeline pelo valor de 'tipo_imovel' da linha antes de prever.")


if __name__ == "__main__":
    main()