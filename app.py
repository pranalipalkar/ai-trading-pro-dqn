import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import yfinance as yf
from enviroment import PortfolioEnv
from model import build_dqn_model

st.set_page_config(page_title="AI Trading Pro", layout="wide")

# CSS to make the popover button a compact widget pinned to the bottom-right corner
st.markdown(
    """
    <style>
    /* Target the container wrapping the popover button */
    div[data-testid="stPopover"] {
        position: fixed !important;
        bottom: 30px !important;
        right: 30px !important;
        z-index: 999999 !important;
        width: auto !important;
    }
    
    /* Ensure the button inside stays compact and styled like a floating badge */
    div[data-testid="stPopover"] > button {
        width: auto !important;
        border-radius: 20px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
        padding: 8px 16px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- FLOATING CHATBOT (No Refresh) ---
with st.popover("💬 Ask AI Assistant"):
    st.markdown("### 🤖 Trading Guide")
    st.write("Click on a question below to get instant help:")

    if st.button("What does green arrow indicate?"):
        st.success(
            "🟢 **BUY:** This means the AI predicts the price will rise soon. It suggests this is a good time to invest."
        )

    if st.button("What does red arrow indicate?"):
        st.error(
            "🔴 **SELL:** This means the AI predicts the price might drop. It suggests selling now to protect your capital or book profits."
        )

    if st.button("Why is the AI 'Holding'(yellow)?"):
        st.warning(
            "🟡 **HOLD:** This means the market trend is currently unclear. The AI is waiting for a better opportunity before making a move."
        )

    if st.button("How can I trust this AI?"):
        st.info(
            "🛡️ This AI analyzes historical patterns and technical data. If the **'Alpha'** metric is positive, it means the AI is outperforming the general market benchmark."
        )

st.title("🚀 Advanced AI Portfolio Optimizer (DQN)")

# --- Data Selection Logic ---
st.sidebar.header("Data Source")
source = st.sidebar.radio("Select Source", ["Local CSV (SPX.csv)", "Live Yahoo Finance"])


@st.cache_data
def load_data(source_type, ticker_symbol):
    if source_type == "Live Yahoo Finance":
        df = yf.download(ticker_symbol, period="2y")
        df.columns = df.columns.get_level_values(0)
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, 'SPX.csv')
        df = pd.read_csv(file_path)

    # Feature Engineering
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['Daily_Return'] = df['Close'].pct_change()
    df.dropna(inplace=True)
    return df


ticker_input = "AAPL"
if source == "Live Yahoo Finance":
    ticker_input = st.sidebar.text_input("Ticker Symbol", "AAPL")

df = load_data(source, ticker_input)

if df is not None:
    test_days = st.sidebar.slider("Testing Window (Days)", 30, 500, 150)

    if st.button('🚀 Execute Strategy'):
        with st.spinner('AI Agent is Training and Trading...'):
            # 1. Data Prep
            train_df = df.iloc[-(test_days + 300):-test_days].reset_index(drop=True)
            test_df = df.tail(test_days).reset_index(drop=True)

            # 2. Training
            env = PortfolioEnv(train_df)
            model = build_dqn_model(state_shape=3, n_actions=3)

            for episode in range(5):
                state = env.reset()
                for _ in range(len(train_df) - 1):
                    state_input = state.reshape(1, 3)
                    if np.random.rand() < 0.3:
                        action = np.random.choice([0, 1, 2])
                    else:
                        q_values = model.predict(state_input, verbose=0)[0]
                        q_values[0] -= 0.1
                        action = np.argmax(q_values)
                    state, reward, done = env.step(action)

          # 3. Testing (DYNAMIC ADAPTIVE DECISION LOGIC)
            test_env = PortfolioEnv(test_df)
            state = test_env.reset()
            history = []
            actions_taken = []
            has_shares = False  # Track position internally

            for i in range(len(test_df) - 1):
                state_input = state.reshape(1, 3)
                
                # Get raw DQN output predictions
                q_vals = model.predict(state_input, verbose=0)[0]
                
                # Dynamic Momentum Check (Close Price vs SMA_20)
                current_price = test_df['Close'].iloc[i]
                sma_20 = test_df['SMA_20'].iloc[i]
                rsi_val = test_df['RSI'].iloc[i]

                # If Price is above SMA and RSI is healthy -> Strong Uptrend (BUY / HOLD)
                if current_price > sma_20 and rsi_val > 45:
                    if not has_shares:
                        action = 1  # BUY if not already holding
                        has_shares = True
                    else:
                        action = 0  # KEEP HOLDING to capture gains
                # If Price drops below SMA or RSI is overbought -> Downtrend (SELL)
                elif current_price < sma_20 or rsi_val > 70:
                    if has_shares:
                        action = 2  # SELL to protect profit
                        has_shares = False
                    else:
                        action = 0  # HOLD CASH
                else:
                    # Let the DQN Model decide
                    action = np.argmax(q_vals)
                    if action == 1:
                        has_shares = True
                    elif action == 2:
                        has_shares = False

                state, reward, done = test_env.step(action)
                history.append(test_env.net_worth)
                actions_taken.append(action)

            # 4. Benchmarking
            bh_return = (test_df['Close'].iloc[-1] / test_df['Close'].iloc[0]) * 10000

            # 5. UI - Metrics
            st.subheader("📊 Performance Analytics")
            m1, m2, m3 = st.columns(3)

            ai_perf_pct = ((test_env.net_worth - 10000) / 100)
            alpha_val = test_env.net_worth - bh_return

            m1.metric("AI Final Value", f"${test_env.net_worth:,.2f}", f"{ai_perf_pct:.2f}%")
            m2.metric("Buy & Hold Value", f"${bh_return:,.2f}", f"{((bh_return - 10000) / 100):.2f}%")
            m3.metric("Alpha (AI vs Market)", f"${alpha_val:,.2f}")

            # Result Logic
            if alpha_val < 0:
                st.error(f"❌ Result: AI agent loses the benchmark by ${abs(alpha_val):,.2f}")
            elif ai_perf_pct > 0:
                st.success(f"✅ Result: You gain {ai_perf_pct:.2f}% of profit!")
            else:
                st.warning("⚖️ Result: No profit or loss (Neutral Performance)")

            # Strategy Status with Reason
            last_action = actions_taken[-1] if actions_taken else 0
            current_rsi = test_df['RSI'].iloc[-1]

            status_text = ""
            reason = ""
            if last_action == 1:
                status_text = "BUY STOCK 🟢"
                reason = "RSI is low (Oversold), suggesting a price bounce-back." if current_rsi < 42 else "AI detects strong upward momentum."
            elif last_action == 2:
                status_text = "SELL STOCK 🔴"
                reason = "RSI is high (Overbought), suggesting the price might fall." if current_rsi > 58 else "AI is protecting profits from a downward trend."
            else:
                status_text = "KEEP HOLDING 🟡"
                reason = "Market indicators are neutral. Waiting for a better entry point."

            st.info(f"📢 **Current Strategy:** {status_text}  \n**Reason:** {reason}")

            # --- 🚀 Trade Statistics Section ---
            st.write("### 📈 AI Trade Statistics")
            total_buys = actions_taken.count(1)
            total_sells = actions_taken.count(2)

            s1, s2, s3 = st.columns(3)
            s1.write(f"**Total Trades:** {total_buys + total_sells}")
            s2.write(f"**Buy Signals:** {total_buys}")
            s3.write(f"**Sell Signals:** {total_sells}")

            # 6. UI - Advanced Graph
            st.write("### AI Trading Signals & Portfolio Growth")
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [1, 2]})

            ax1.plot(test_df['Close'].values, label="Price", color='purple', alpha=0.5)
            for i, action in enumerate(actions_taken):
                if action == 1:
                    ax1.scatter(i, test_df['Close'].iloc[i], marker='^', color='green', s=100)
                elif action == 2:
                    ax1.scatter(i, test_df['Close'].iloc[i], marker='v', color='red', s=100)
            ax1.set_title("Price Buy/Sell Markers")
            ax1.legend()

            ax2.plot(history, label="DQN Agent", color="#1E90FF", linewidth=2)
            ax2.axhline(y=10000, color='black', linestyle='--', alpha=0.3)
            ax2.set_ylabel("Net Worth ($)")
            ax2.legend()

            st.pyplot(fig)

            # --- View Summary Expander & Download ---
            st.markdown("---")

            trade_logs = []
            for i, action in enumerate(actions_taken):
                if action != 0:
                    trade_logs.append({
                        "Day": i + 1,
                        "Action": "BUY 🟢" if action == 1 else "SELL 🔴",
                        "Price": f"${test_df['Close'].iloc[i]:,.2f}",
                        "RSI": f"{test_df['RSI'].iloc[i]:.2f}",
                        "Portfolio Value": f"${history[i]:,.2f}"
                    })

            if trade_logs:
                final_df = pd.DataFrame(trade_logs)

                with st.expander("📄 View Detailed Trade Summary", expanded=True):
                    st.dataframe(final_df, use_container_width=True, height=400)

                csv = final_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Trade Report",
                    data=csv,
                    file_name='trading_report.csv',
                    mime='text/csv',
                )
            else:
                st.write("No trades were made by the AI during this period.")

            st.success("Analysis Complete!")
