# Advanced AI Portfolio Optimizer (DQN)

A Streamlit web app that simulates an AI trading agent for stocks/indexes. The agent uses a Deep Q-Network (DQN)–style neural network to choose **Hold**, **Buy**, or **Sell** based on price and technical indicators, then compares its portfolio against a simple **Buy & Hold** benchmark.

> **Disclaimer:** This project is for learning and demonstration only. It is **not** financial advice and should not be used for real trading decisions.

---

## What this project does

1. Loads market data from a local CSV (`SPX.csv`) or live Yahoo Finance.
2. Builds features: **RSI**, **SMA (20-day)**, and daily returns.
3. Runs a DQN-style agent inside a simulated portfolio environment (starts with **$10,000**).
4. Shows performance metrics, buy/sell markers on a price chart, portfolio growth, and a downloadable trade report.

---

## Project structure

```
ai-trading-pro-dqn/
├── app.py              # Main Streamlit UI and strategy pipeline
├── enviroment.py       # Portfolio trading simulation (gym-like env)
├── model.py            # NumPy DQN neural network
├── main.py             # CLI smoke test (no UI)
├── SPX.csv             # Sample S&P 500 historical data
├── requirements.txt    # Python dependencies
├── .streamlit/
│   └── config.toml     # Streamlit server settings
└── README.md           # This file
```

---

## How it works (end-to-end)

```
Data (CSV / Yahoo)
        │
        ▼
Feature engineering (RSI, SMA_20, returns)
        │
        ▼
Split into Train window + Test window
        │
        ▼
PortfolioEnv + DQN model
        │
        ├── Training loop (explore vs exploit actions)
        └── Testing loop (agent trades day by day)
                │
                ▼
Compare AI portfolio vs Buy & Hold → charts + report
```

### 1. Data & features (`app.py` → `load_data`)

- **Local CSV:** reads `SPX.csv` (must include a `Close` column).
- **Live Yahoo Finance:** downloads ~2 years of data for a ticker (e.g. `AAPL`, `MSFT`, `^GSPC`).

Then it computes:

| Feature   | Meaning |
|-----------|---------|
| `Close`   | Closing price |
| `RSI`     | Relative Strength Index (14-day) — overbought/oversold signal |
| `SMA_20`  | 20-day simple moving average — trend filter |
| `Daily_Return` | Day-to-day % change |

Rows with missing values (from rolling windows) are dropped.

### 2. Trading environment (`enviroment.py`)

`PortfolioEnv` is a simple reinforcement-learning style environment:

| Item | Detail |
|------|--------|
| Starting cash | `$10,000` |
| State (observation) | `[Close, RSI, SMA_20]` |
| Actions | `0 = Hold`, `1 = Buy`, `2 = Sell` |
| Buy | Spends all cash to buy shares (all-in) |
| Sell | Converts all shares back to cash |
| Reward | Current net worth |
| Done | Reached the last day of the dataframe |

Each `step(action)` moves one day forward and updates balance, shares, and net worth.

### 3. DQN model (`model.py`)

`NumpyDQN` is a small feed-forward network (no TensorFlow, so it deploys cleanly on Streamlit Cloud):

```
Input (3) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(3) Q-values
```

- Output Q-values estimate how good each action is for the current state.
- The agent picks `argmax(Q)` → Hold / Buy / Sell.
- `build_dqn_model(state_shape=3, n_actions=3)` creates the model.

### 4. Strategy pipeline in the UI (`app.py`)

When you click **Execute Strategy**:

1. **Split data**
   - Test set = last N days (sidebar slider).
   - Train set = ~300 days before the test window.
2. **Training loop** (5 episodes)
   - 30% of the time: random action (exploration).
   - 70% of the time: action from the model (exploitation).
   - Slight bias against “Hold” (`q_values[0] -= 0.1`) so the agent trades more.
3. **Testing loop**
   - Agent trades only with model predictions (no random actions).
   - Records net worth history and every action.
4. **Benchmark**
   - Buy & Hold = invest $10,000 on day 1 and hold until the last day.
   - **Alpha** = AI final value − Buy & Hold value.
5. **Results UI**
   - Metrics, current strategy message, trade counts, charts, and CSV download.

### 5. CLI test script (`main.py`)

A lightweight script (no Streamlit) that:

- Loads `SPX.csv`
- Adds RSI + SMA
- Creates the env + model
- Prints one sample action

Useful to verify the core logic works before running the web app.

---

## File-by-file explanation

### `app.py`
Main user-facing app:

- Page title, custom CSS, floating help popover
- Sidebar: data source, ticker, testing window
- Loads/caches data, runs train/test, renders metrics and plots

### `enviroment.py`
Defines `PortfolioEnv` — the simulated brokerage account the agent interacts with.

### `model.py`
Defines `NumpyDQN` and `build_dqn_model()` — the brain that scores Hold/Buy/Sell.

### `main.py`
Command-line sanity check of data loading + one model prediction.

### `SPX.csv`
Historical market data used for offline demos (no internet required).

### `requirements.txt`
Dependencies: `streamlit`, `pandas`, `numpy`, `matplotlib`, `yfinance`.

---

## Step-by-step: run locally

### 1. Prerequisites
- Python 3.10+ recommended
- `pip` available

### 2. Open the project folder
```bash
cd /path/to/ai-trading-pro-dqn
```

### 3. (Optional) Create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Quick CLI test (optional)
```bash
python main.py
```
You should see something like:
```text
AI took action: 1 (0=Hold, 1=Buy, 2=Sell)
Working Model is Ready!
```

### 6. Start the Streamlit app
```bash
streamlit run app.py
```

### 7. Open the browser
Streamlit prints a local URL, usually:
```text
http://localhost:8501
```

---

## Step-by-step: use the web app

1. Open the app (local or Streamlit Cloud URL).
2. In the **sidebar**, choose a data source:
   - **Local CSV (SPX.csv)** — uses the included dataset
   - **Live Yahoo Finance** — enter a ticker (e.g. `AAPL`)
3. Set **Testing Window (Days)** (default `150`).
4. Click **🚀 Execute Strategy**.
5. Wait for the spinner to finish.
6. Review results:
   - **AI Final Value** — portfolio after simulated trading
   - **Buy & Hold Value** — passive benchmark
   - **Alpha** — AI minus Buy & Hold (positive = beat the market in this simulation)
7. Read **Current Strategy** (Buy / Sell / Hold) and the reason.
8. Check charts:
   - Green `▲` = Buy
   - Red `▼` = Sell
   - Blue line = portfolio net worth over time
9. Expand **View Detailed Trade Summary** and optionally **Download Trade Report**.
10. Use the floating **Ask AI Assistant** button (bottom-right) for quick help on signals.

---

## Step-by-step: deploy on Streamlit Community Cloud

1. Push this repo to GitHub (already available at  
   `https://github.com/pranalipalkar/ai-trading-pro-dqn`).
2. Go to [https://share.streamlit.io](https://share.streamlit.io).
3. Sign in with GitHub.
4. Click **New app**.
5. Select:
   - Repository: `pranalipalkar/ai-trading-pro-dqn`
   - Branch: `main`
   - Main file path: `app.py`
6. (Recommended) In **Advanced settings**, pick **Python 3.11** or **3.12** if available.
7. Click **Deploy**.
8. When the build finishes, open your public app URL.

---

## Understanding the signals

| Marker / status | Meaning |
|-----------------|---------|
| Green arrow / BUY | Agent chose to enter (or stay fully invested) |
| Red arrow / SELL | Agent chose to exit to cash |
| Yellow / HOLD | Agent kept the current position without trading |
| Positive Alpha | AI ended above Buy & Hold in this backtest window |
| Negative Alpha | AI underperformed Buy & Hold in this window |

---

## Tech stack

- **UI:** Streamlit
- **Data:** pandas, yfinance
- **Math / model:** NumPy (lightweight DQN)
- **Charts:** Matplotlib
- **Hosting:** Streamlit Community Cloud (or any host that can run `streamlit run app.py`)

---

## Limitations (important)

- This is a **simulation / demo**, not a production trading system.
- The network is a simplified DQN-style model; results vary by data window and randomness during exploration.
- Live Yahoo Finance depends on network access and Yahoo’s availability.
- Past simulated performance does **not** predict future results.

---

## License

Educational / personal use. Add a formal license file if you plan to redistribute.
