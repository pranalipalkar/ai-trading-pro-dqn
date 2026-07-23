import os
import pandas as pd
import numpy as np
from model import build_dqn_model
from enviroment import PortfolioEnv

# TensorFlow to hide irrelevant message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 1. load the dada
try:
    df = pd.read_csv('SPX.csv')

    # --- calculate RSI ---
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # --- calculate SMA_20 ---
    df['SMA_20'] = df['Close'].rolling(window=20).mean()

    # remove sells to avoid errors (NaN)
    df.dropna(inplace=True)
    df = df.reset_index(drop=True)

    print("Columns added successfully:", df.columns)
    print("--- File Loaded Successfully! ---")
except FileNotFoundError:
    print("Error: file not found")
    exit()

# 2. Environment and Model setup
env = PortfolioEnv(df.tail(500))
model = build_dqn_model(state_shape=3, n_actions=3)

# 3. small working test
print("--- Starting AI Model ---")
state = env.reset().reshape(1, 3)
action_values = model.predict(state, verbose=0)
action = np.argmax(action_values[0])
print(f"AI took action: {action} (0=Hold, 1=Buy, 2=Sell)")
print("Working Model is Ready!")