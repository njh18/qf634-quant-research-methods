import pandas as pd 
import numpy as np
import os 

df_crypto = pd.read_csv('data/crypto.csv')
df_stocks = pd.read_csv('data/stocks.csv')
df_crypto = df_crypto[['Date', 'Open', 'Close', 'High', 'Low', 'Ticker']]
df_stocks = df_stocks[['Date', 'Open', 'Close', 'High', 'Low', 'Ticker']]
print(df_stocks)



