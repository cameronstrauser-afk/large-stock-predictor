import streamlit as st
import plotly.graph_objects as go

from utils.data import load_stock
from utils.indicators import add_indicators
from utils.lstm_model import train_or_load_lstm
from utils.transformer_model import train_or_load_transformer
from utils.boost_model import train_or_load_boost
from utils.signal_engine import generate_signal
from utils.backtest import backtest
from utils.watchlist import load_watchlist, save_watchlist

st.set_page_config(layout="wide")
st.title("🚀 AI Quant Stock Engine")

# WATCHLIST
watchlist = load_watchlist()
new = st.sidebar.text_input("Add Stock")

if st.sidebar.button("Add"):
    if new.upper() not in watchlist:
        watchlist.append(new.upper())
        save_watchlist(watchlist)

st.sidebar.write("Watchlist:", watchlist)

ticker = st.text_input("Stock Ticker", value="AAPL").upper()

if st.button("Run AI Analysis"):

    df = load_stock(ticker)
    df = add_indicators(df)

    lstm_model = train_or_load_lstm(ticker, df)
    transformer_model = train_or_load_transformer(ticker, df)
    boost_model = train_or_load_boost(ticker, df)

    signal, price, prob = generate_signal(
        df, ticker, lstm_model, transformer_model, boost_model
    )

    st.metric("AI Signal", signal)
    st.metric("Confidence %", prob)
    st.metric("Predicted Price", round(price,2))

    win, sharpe, draw = backtest(df)

    st.subheader("Backtest Stats")
    st.write("Win Rate:", win,"%")
    st.write("Sharpe:", sharpe)
    st.write("Max Drawdown:", draw)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"]
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA"], name="SMA"))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA"], name="EMA"))

    st.plotly_chart(fig, use_container_width=True)

# Fractional shares
st.subheader("💰 Fractional Shares Calculator")
investment = st.number_input("Investment ($)", value=1000)
if ticker:
    df = load_stock(ticker)
    price = df["Close"].iloc[-1]
    shares = investment/price
    st.write(f"You can buy {round(shares,4)} shares")
