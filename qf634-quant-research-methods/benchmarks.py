import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

class MinVarianceMethod:
    
    def __init__(
                     self, 
                     allow_short = False,
                 ):
        self.allow_short = allow_short

    
    def get_optimal_weights(self, returns):
        cov_matrix = returns.cov() * 252
        num_assets = len(returns.columns)

        if self.allow_short:
            bounds = Bounds(-1,1)
        else:
            bounds = Bounds(0, 1)

        constraints = LinearConstraint(np.ones((num_assets), dtype=int),1,1) 
        initial_weights = np.ones(num_assets) / num_assets 
    
        portfvola = lambda w: np.sqrt(np.dot(w,np.dot(w,cov_matrix)))

        opt_S = minimize(
            fun=portfvola,
            x0=initial_weights,
            method='trust-constr',
            constraints=constraints,
            bounds=bounds
        )

        if not opt_S.success:
            raise ValueError(f"Optimization failed: {opt_S.message}")

        optimal_weights = opt_S.x
        assert np.sum(np.array(optimal_weights)).round() == 1 
        return optimal_weights
    

class MaxSharpeMethod:
    
    def __init__(
                     self, 
                     allow_short = False,
                 ):
        self.allow_short = allow_short

    
    def get_optimal_weights(self, returns):
        expected_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        num_assets = len(returns.columns)

        invSharpe = lambda w: np.sqrt(np.dot(w,np.dot(w,cov_matrix)))/expected_returns.dot(w)
        constraints = LinearConstraint(np.ones((num_assets), dtype=int),1,1) 
        initial_weights = np.ones(num_assets) / num_assets 
        
        if self.allow_short:
            bounds = Bounds(-1,1)
        else:
            bounds = Bounds(0, 1)

        opt_S = minimize(
            fun=invSharpe,
            x0=initial_weights,
            method='trust-constr',
            constraints=constraints,
            bounds=bounds
        )

        if not opt_S.success:
            raise ValueError(f"Optimization failed: {opt_S.message}")

        optimal_weights = opt_S.x        
        assert np.sum(np.array(optimal_weights)).round() == 1 
        return optimal_weights
    
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
        
        # combine data into forecast_df
        forecast_df = pd.DataFrame(forecasted_returns)
        print(forecast_df)
        minVarMethod = MinVarianceMethod(allow_short=self.allow_short)
        return minVarMethod.get_optimal_weights(forecast_df)