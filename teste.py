import pandas as pd

df = pd.read_csv("./bases/imoveis_agregado.csv")

print(df[df["quartos"] >=  10])