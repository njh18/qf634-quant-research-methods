from environment import Environment
import numpy as np
import pandas as pd
import os 
import holidays
import matplotlib.pyplot as plt


csv_directory = "historical_data" 

data_frames = {}

for file in os.listdir(csv_directory):
    if file.endswith(".csv"): 
        asset_name = file.split(".csv")[0]
        file_path = os.path.join(csv_directory, file)
        
        df = pd.read_csv(file_path, index_col="Date", parse_dates=True)
        
        # CHANGE HERE IF U DONT WANT TO USE ADJ CLOSE
        data_frames[asset_name] = df["Adj Close"].rename(asset_name)

combined_df = pd.concat(data_frames.values(), axis=1)
combined_df = combined_df.reset_index()
#Taking 2018 as start date to account for crypto
df_filtered = combined_df[combined_df['Date'] >= '2018-01-01']
#Taking a subset of the stocks
filtered_df = df_filtered.loc[:, ['Date', 'XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP',
       'XLRE', 'XLU', 'XLV', 'XLY', 'ADA-USD', 'BTC-USD', 'XRP-USD']]

#Filter out days that are NOT trading days and are Holidays
us_holidays = holidays.US()
filtered_df['Holiday'] = filtered_df.Date.apply(lambda x: x in us_holidays)
filtered_df['Day'] = filtered_df['Date'].dt.day_name()
filtered_df = filtered_df.loc[filtered_df['Holiday'] == False]
filtered_df = filtered_df.loc[~filtered_df.Day.isin(['Saturday', 'Sunday'])]

df_price = filtered_df.copy()
df_price.drop(['XLC', 'Holiday', 'Day'], axis=1, inplace=True)
df_price.to_csv('df_price.csv', index=False)

#calculate the percentage change row wise
df_pct_change = filtered_df.iloc[:, 1:-2].pct_change()
# Add the Date column back to the percentage change DataFrame
df_pct_change['Date'] = filtered_df['Date']
# Rearrange the columns to put Date first
df_pct_change = df_pct_change[['Date'] + [col for col in df_pct_change.columns if col != 'Date']]

#handle missing values 
df_pct_change.drop('XLC', axis=1, inplace=True)
df_pct_change.dropna(inplace=True)
df_pct_change.set_index('Date', inplace=True)
df_pct_change.to_csv('df_pct_change.csv', index=True)




