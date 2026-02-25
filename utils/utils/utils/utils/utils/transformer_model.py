import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, MultiHeadAttention
from tensorflow.keras.models import Model
import numpy as np
import os, joblib
from sklearn.preprocessing import MinMaxScaler

LOOKBACK = 60

def transformer_encoder(inputs):
    x = MultiHeadAttention(num_heads=4, key_dim=32)(inputs, inputs)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = Dense(64, activation="relu")(x)
    return x

def build_transformer():
    inputs = tf.keras.Input(shape=(LOOKBACK,1))
    x = transformer_encoder(inputs)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    return model

def train_or_load_transformer(ticker, df):
    path = f"models/{ticker}_transformer.h5"

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[["Close"]])

    X, y = [], []
    for i in range(LOOKBACK, len(scaled)):
        X.append(scaled[i-LOOKBACK:i])
        y.append(scaled[i])

    X, y = np.array(X), np.array(y)

    if os.path.exists(path):
        return tf.keras.models.load_model(path)

    model = build_transformer()
    model.fit(X, y, epochs=6, batch_size=32, verbose=0)
    model.save(path)
    joblib.dump(scaler, f"models/{ticker}_transformer_scaler.save")
    return model
