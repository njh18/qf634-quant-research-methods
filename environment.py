
import pandas as pd

class Environment:

    def __init__(self):
        self.returns = self.load_returns();
        self.prices = self.load_prices(); 

    def load_returns(self):
        df = pd.read_csv("df_pct_change.csv")
        df = df.drop(columns=["Date"])
        return df * 100;

    def load_prices(self):
        df = pd.read_csv("df_price.csv")
        df = df.drop(columns=["Date"])
        return df

    def get_state(self, start, length) -> pd.DataFrame:
        # if end exceeds the window
        end = start + length - 1
        end = min(end, len(self.returns) - 1)
        return self.returns.iloc[start:end+1] 

    def get_prices(self, start, length) -> pd.DataFrame:
        # if end exceeds the window
        end = start + length - 1
        end = min(end, len(self.prices) - 1)
        return self.prices.iloc[start:end+1] 