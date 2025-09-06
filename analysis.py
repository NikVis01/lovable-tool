import pandas as pd

df = pd.read_csv("src/data/agent_output.csv")

print(df.head())

print(df.columns)

print(df.info())

print(df.describe())
