from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
import joblib, os

def train_or_load_boost(ticker, df):
    path = f"models/{ticker}_boost.pkl"

    X = df[["Open","High","Low","Close","Volume","RSI","SMA","EMA","MACD","VOL"]]
    y = df["Close"].shift(-1).dropna()
    X = X.iloc[:-1]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    if os.path.exists(path):
        return joblib.load(path)

    model = GradientBoostingRegressor()
    model.fit(X_scaled, y)
    joblib.dump(model, path)
    joblib.dump(scaler, f"models/{ticker}_boost_scaler.save")
    return model
