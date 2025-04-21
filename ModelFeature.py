import pandas as pd

path = r"C:\Users\frank\Downloads\PyUtilities\Stoch-Mort-With-ML\Docs\ITA\InputDB\ITAdeath.txt"
df = pd.read_csv(path, delimiter= ",")

print(df)
