import numpy as np
import joblib

from utils.lstm_model import LOOKBACK
from utils.transformer_model import LOOKBACK as T_LOOKBACK

def predict_sequence(model, scaler, df, lookback):
    data = scaler.transform(df[["Close"]])
    last = data[-lookback:]
    pred = model.predict(last.reshape(1, lookback, 1), verbose=0)
    return scaler.inverse_transform(pred)[0][0]

def generate_signal(df, ticker, lstm_model, transformer_model, boost_model):

    lstm_scaler = joblib.load(f"models/{ticker}_lstm_scaler.save")
    trans_scaler = joblib.load(f"models/{ticker}_transformer_scaler.save")
    boost_scaler = joblib.load(f"models/{ticker}_boost_scaler.save")

    lstm_pred = predict_sequence(lstm_model, lstm_scaler, df, LOOKBACK)
    trans_pred = predict_sequence(transformer_model, trans_scaler, df, T_LOOKBACK)

    X = df[["Open","High","Low","Close","Volume","RSI","SMA","EMA","MACD","VOL"]].iloc[-1:]
    boost_pred = boost_model.predict(boost_scaler.transform(X))[0]

    final_pred = np.mean([lstm_pred, trans_pred, boost_pred])
    current = df["Close"].iloc[-1]

    change_pct = (final_pred-current)/current*100

    rsi = df["RSI"].iloc[-1]
    macd = df["MACD"].iloc[-1]
    vol = df["VOL"].iloc[-1]

    threshold = 1.5 + vol*100

    if change_pct > threshold and rsi < 70 and macd > 0:
        signal = "BUY"
    elif change_pct < -threshold and rsi > 30 and macd < 0:
        signal = "SELL"
    else:
        signal = "HOLD"

    probability = min(abs(change_pct)*4,95)

    return signal, final_pred, round(probability,2)
