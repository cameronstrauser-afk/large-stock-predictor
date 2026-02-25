import yfinance as yf

def load_stock(ticker, period="5y"):
    df = yf.Ticker(ticker).history(period=period)
    df.dropna(inplace=True)
    return df
