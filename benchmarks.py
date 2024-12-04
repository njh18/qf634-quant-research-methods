import numpy as np
from scipy.optimize import minimize

class MinVarianceMethod:
    
    def __init__(
                     self, 
                     allow_short = False,
                 ):
        self.allow_short = allow_short

    
    def get_optimal_weights(self, returns):
        
        def objective_function(weights, cov_matrix):
            # trying to minimize variance
            return np.dot(weights.T, np.dot(cov_matrix, weights))
            
        expected_returns = returns.mean()
        cov_matrix = returns.cov()
        num_assets = len(expected_returns)
        
        if self.allow_short:
            bnds = [(-1, 1) for _ in range(num_assets)]
            cons =({'type': 'eq', 'fun': lambda x : 1.0 - np.sum(np.abs(x))})
        else:
            bnds = [(0, 1) for _ in range(num_assets)]
            cons =({'type': 'eq', 'fun': lambda x : 1.0 - np.sum(x)})
        

        initial_weights = np.ones(num_assets) / num_assets 
        opt_S = minimize(
            fun=objective_function,
            x0=initial_weights,
            args=(cov_matrix,),
            method='SLSQP',
            constraints=cons,
            bounds=bnds
        )

        optimal_weights = opt_S.x
        
        # sometimes optimization fails with constraints, need to be fixed by hands
        if self.allow_short:
            optimal_weights /= sum(np.abs(optimal_weights))
        else:
            optimal_weights += np.abs(np.min(optimal_weights))
            optimal_weights /= sum(optimal_weights)
        
        return optimal_weights
    