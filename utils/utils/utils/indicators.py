import ta

def add_indicators(df):
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
    df["SMA"] = ta.trend.sma_indicator(df["Close"], window=20)
    df["EMA"] = ta.trend.ema_indicator(df["Close"], window=20)
    df["MACD"] = ta.trend.macd(df["Close"])
    df["VOL"] = df["Close"].pct_change().rolling(20).std()
    df.dropna(inplace=True)
    return df
