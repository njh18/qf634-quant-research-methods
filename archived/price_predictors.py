from benchmarks import MaxSharpeMethod
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

class ArimaModel:
    def __init__(
                    self, 
                    allow_short = False,
                ):
        self.allow_short = allow_short
    
    def get_optimal_weights(self, returns, holding_period):

        forecasted_returns = {}
        for asset in returns.columns:
            model = ARIMA(returns[asset], order=(1, 0, 1))
            fitted_model = model.fit()
            forecast = fitted_model.forecast(steps=holding_period)
            forecasted_returns[asset] = forecast.values 
        forecast_df = pd.DataFrame(forecasted_returns)
        forecast_returns = forecast_df.pct_change().dropna() * 100
        method = MaxSharpeMethod(allow_short=True)
        return method.get_optimal_weights(returns=forecast_returns)