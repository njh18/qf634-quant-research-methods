
import pandas as pd

class Environment:

    def __init__(self):
        self.prices = self.load_prices(); 
        self.returns = self.load_returns();

    def load_returns(self):
        df = self.prices.pct_change()
        df = df.dropna()
        return df * 100;

    def load_prices(self):
        df = pd.read_csv("df_price.csv")
        df = df.drop(columns=["Date"])
        df = df.dropna()
        return df

    def get_state(self, end, lookback) -> pd.DataFrame:
        assert lookback <= end
        return self.returns.iloc[end-lookback:end] 

    def get_prices(self, end, lookback) -> pd.DataFrame:
        assert lookback <= end
        return self.prices.iloc[end-lookback:end] 