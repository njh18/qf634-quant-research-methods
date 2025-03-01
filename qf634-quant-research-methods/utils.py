import numpy as np

def get_stats(daily_returns: list) -> dict:
    if daily_returns is None:
        raise ValueError("Missing Daily returns!")

    risk_free_rate = 0;
    trading_days = 252;

    cumulative_return = calculate_cumulative_return(daily_returns)
    sharpe_ratio = calculate_sharpe_ratio(daily_returns, risk_free_rate, trading_days)
    max_drawdown = calculate_maximum_drawdown(daily_returns)
    value_at_risk = calculate_value_at_risk(daily_returns)
    return {
        "Cumulative Return": cumulative_return,
        "Sharpe Ratio": sharpe_ratio,
        "Maximum Drawdown": max_drawdown,
        "Value at Risk": value_at_risk,
    };

def calculate_cumulative_return(daily_returns: list) -> float:
    cum_pct_change = cumulative_pct_change(daily_returns)
    return cum_pct_change[-1]

def calculate_sharpe_ratio(daily_returns: list, risk_free_rate: float, trading_days: float) -> float:
    daily_risk_free_rate = (1 + risk_free_rate) ** (1 / trading_days) - 1
    excess_returns = np.array(daily_returns) - daily_risk_free_rate
    return excess_returns.mean() / excess_returns.std()

def calculate_maximum_drawdown(daily_returns: list) -> float:
    print("UPDATED returns")
    cumulative_returns = cumulative_pct_change(daily_returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = (running_max - cumulative_returns) / running_max
    return drawdowns.max()

def calculate_value_at_risk(daily_returns: list, confidence_level=0.95):
    return -np.percentile(daily_returns, 100 * (1 - confidence_level), method='nearest')

def cumulative_pct_change(daily_returns: list) -> list:
    cum_pct_change = (1 + np.array(daily_returns) / 100).cumprod() - 1 
    return cum_pct_change * 100