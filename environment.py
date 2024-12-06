
import pandas as pd

class Environment:

    def __init__(self, filename = "combined_adj_close.csv"):
        self.data = self.load_data(filename)
        self.returns = self.load_returns()

    
    def load_data(self, filename):
        return pd.read_csv(filename)

    def load_returns(self):
        df = self.data
        df['Date'] = pd.to_datetime(df['Date'])

        # Find the earliest date for each column where values are not NaN
        earliest_dates = {
            col: df.loc[df[col].notna(), 'Date'].min() for col in df.columns if col != 'Date'
        }
        date_df = pd.Series(earliest_dates, name='Earliest Date')

        df_filtered = df[df['Date'] >= date_df.max()] 

        # drop all na and remove indexing
        df_filtered.reset_index(inplace=True)
        df_filtered.dropna(axis=0, inplace=True)
        df_filtered = df_filtered.drop(columns=["index"])

        # get pct returns
        return_df = df_filtered.drop(columns=['Date']).pct_change() * 100
        return_df.dropna(inplace=True)

        return return_df
    

    def get_state(self, start, length) -> pd.DataFrame:
        # if end exceeds the window
        end = start + length - 1
        end = min(end, len(self.returns) - 1)
        return self.returns.iloc[start:end+1]  # iloc is integer-based indexing
