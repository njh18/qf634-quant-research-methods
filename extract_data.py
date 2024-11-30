import yfinance as yf
import pandas as pd
import numpy as np
import os

start_date = '2018-01-01'
end_date = '2023-12-31'

## etfs extraction
etfs = ['XLC', 'XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLB', 'XLRE', 'XLK', 'XLU']

## crypto extraction
cryptos = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'LTC-USD', 'ADA-USD', 'BCH-USD', 'BNB-USD', 'DASH-USD', 'EOS-USD', 'IOT-USD', 'LINK-USD', 'TRX-USD', 'USDT-USD', 'XLM-USD', 'XMR-USD']

def extract_data(ticker_lst, start, end):
    df = yf.download(ticker_lst, start=start_date, end=end_date)
    df = df.stack().reset_index()
    return df

etfs_data = extract_data(etfs, start_date, end_date)
crypto_data = extract_data(etfs, start_date, end_date)

etfs_data.to_csv('data/stocks.csv')
crypto_data.to_csv('data/crypto.csv')
