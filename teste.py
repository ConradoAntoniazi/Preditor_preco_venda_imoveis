import pandas as pd

df =  pd.read_csv("./saidas/dataset_modelagem.csv")

print(df[df["bairro"] == ''])