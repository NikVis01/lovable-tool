import pandas as pd

df = pd.read_csv("src/graph/data/candidates.csv")

print(df.head())

print(df.columns)

print(df.info())

print(df.describe())
