import numpy as np

def backtest(df):
    returns = df["Close"].pct_change().dropna()
    win_rate = (returns > 0).mean()*100
    sharpe = returns.mean()/returns.std()
    max_drawdown = (returns.cumsum().cummax() - returns.cumsum()).max()
    return round(win_rate,2), round(sharpe,2), round(max_drawdown,2)
