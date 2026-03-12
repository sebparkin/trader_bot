import streamlit as st
import torch
import joblib
import pandas as pd
import pytz
from datetime import datetime
import plotly.graph_objects as go
import time
import schedule
import threading
from alpaca.broker import BrokerClient
from core.data_handler import DataHandler
from core.paper_trader import PaperTrader
from core.trading_lstm import TradingLTSM
from core.constants import *

ET = pytz.timezone("America/New_York")

st.set_page_config(page_title="Trading Bot Dashboard", page_icon="🚀", layout="wide")

def bot_schedule_loop():

    print(f"Bot schedule started at {datetime.now(ET).strftime('%H:%M:%S ET')}")

    def open_all():
        print(f"Opening positions at {datetime.now(ET).strftime('%H:%M:%S ET')}")
        try:
            traders["AAPL"].open_position(scalers["AAPL"])
        except Exception as e:
            print(f"AAPL open error: {e}")
        try:
            traders["MSFT"].open_position(scalers["MSFT"])
        except Exception as e:
            print(f"MSFT open error: {e}")

    def close_all():
        print(f"Closing positions at {datetime.now(ET).strftime('%H:%M:%S ET')}")
        try:
            traders["AAPL"].close_position()
        except Exception as e:
            print(f"AAPL close error: {e}")
        try:
            traders["MSFT"].close_position()
        except Exception as e:
            print(f"MSFT close error: {e}")

    # Explicit per-day scheduling — no loop, no closure bug
    schedule.every().monday.at("14:31").do(open_all)
    schedule.every().tuesday.at("14:31").do(open_all)
    schedule.every().wednesday.at("14:31").do(open_all)
    schedule.every().thursday.at("14:31").do(open_all)
    schedule.every().friday.at("14:31").do(open_all)

    schedule.every().monday.at("20:45").do(close_all)
    schedule.every().tuesday.at("20:45").do(close_all)
    schedule.every().wednesday.at("20:45").do(close_all)
    schedule.every().thursday.at("20:45").do(close_all)
    schedule.every().friday.at("20:45").do(close_all)

    while True:
        try:
            # Print heartbeat every hour so you can see thread is alive
            print(f"Bot heartbeat: {datetime.now(ET).strftime('%H:%M:%S ET')}")
            schedule.run_pending()
            time.sleep(300)
        except Exception as e:
            print(f"Schedule loop error: {e} — restarting loop")
            time.sleep(30)  # Wait then retry rather than dying

def get_bot_thread():
    """Returns the bot thread if it exists and is alive, else None"""
    for t in threading.enumerate():
        if t.name == "trading_bot":
            return t if t.is_alive() else None
    return None

def ensure_bot_running():
    """Start bot thread if not already running"""
    existing = get_bot_thread()
    if existing is None:
        print(f"Starting bot thread at {datetime.now(ET).strftime('%H:%M:%S ET')}")
        thread = threading.Thread(
            target = bot_schedule_loop,
            daemon = True,
            name   = "trading_bot"
        )
        thread.start()
        time.sleep(0.5)  # Brief pause to let thread start up
        print(f"Bot thread started — alive: {thread.is_alive()}")
        return thread
    else:
        print(f"Bot thread already running — skipping start")
        return existing

# Call this near the top of your script, before any st.* calls
ensure_bot_running()

@st.cache_resource
def load_model_and_scaler(ticker):
    model = TradingLTSM(
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        feature_cols=FEATURES,
    )

    ticker = ticker.lower()

    model.load_state_dict(torch.load(f"models/{ticker}_model.pth", map_location="cpu"))
    model.eval()

    scaler = joblib.load(f"models/{ticker}_scaler.pkl")
    print("Model and scaler loaded")
    return model, scaler


@st.cache_resource
def load_papertrader(_model, ticker):
    return PaperTrader(_model, ticker, cash_fraction=CASH_FRACTION, threshold=THRESHOLD)


tickers = ["AAPL", "MSFT"]
models = {}
scalers = {}
traders = {}
for ticker in tickers:
    models[ticker], scalers[ticker] = load_model_and_scaler(ticker)
    traders[ticker] = load_papertrader(models[ticker], ticker)

trading_client = list(traders.values())[0].trading_client


def get_account_stats():
    account = trading_client.get_account()
    return {
        "portfolio_value": float(account.portfolio_value),
        "cash": float(account.cash),
        "equity": float(account.equity),
        "day_pl": float(account.equity) - float(account.last_equity),
        "buying_power": float(account.buying_power),
    }


def get_position(ticker):
    try:
        pos = trading_client.get_open_position(ticker)
        return {
            "qty": pos.qty,
            "entry_price": float(pos.avg_entry_price),
            "current_price": float(pos.current_price),
            "unrealised_pl": float(pos.unrealized_pl),
            "pl_pct": float(pos.unrealized_plpc) * 100,
        }
    except:
        return None


def get_portfolio_history():

    history = trading_client.get_portfolio_history()
    df = pd.DataFrame(
        {"Date": pd.to_datetime(history.timestamp, unit="s"), "Value": history.equity}
    )
    return df


def get_prediction_now(ticker):
    try:
        pred, date_used = models[ticker].get_yesterdays_prediction(ticker, scalers[ticker])
        return pred, date_used, None
    except Exception as e:
        return None, None, str(e)


st.title("🚀 Trading Bot Dashboard")
st.caption(f"Last refreshed: {datetime.now(ET).strftime('%Y-%m-%d %H:%M:%S ET')}")

# Account metrics
st.subheader("Account Overview")
stats = get_account_stats()
col1, col2, col3, col4 = st.columns(4)
col1.metric("Portfolio Value", f"${stats['portfolio_value']:,.2f}")
col2.metric("Cash", f"${stats['cash']:,.2f}")
col3.metric(
    "Day P&L",
    f"${stats['day_pl']:+,.2f}",
    delta_color="normal" if stats["day_pl"] >= 0 else "inverse",
)
col4.metric("Buying Power", f"${stats['buying_power']:,.2f}")

# Portfolio chart
st.subheader("Portfolio History (1 Month)")
history_df = get_portfolio_history()
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=history_df["Date"],
        y=history_df["Value"],
        mode="lines",
        fill="tozeroy",
        line=dict(color="#00C805"),
    )
)
fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
st.plotly_chart(fig, width="stretch")

st.divider()

# ---- Per-ticker sections ----
TICKER_COLORS = {"AAPL": "#B93939", "MSFT": "#00A4EF"}

for ticker in tickers:
    color = TICKER_COLORS.get(ticker, "#888888")
    st.markdown(f"<h2 style='color:{color}'>{ticker}</h2>", unsafe_allow_html=True)

    pred, date_used, err = get_prediction_now(ticker)
    position = get_position(ticker)

    # Prediction + position metrics
    m1, m2, m3, m4, m5 = st.columns(5)

    if err:
        m1.error(f"Prediction error: {err}")
    else:
        signal = (
            "📈 LONG" if pred > THRESHOLD else "📉 SHORT" if pred < (1 - THRESHOLD) else "⏸ NEUTRAL"
        )
        m1.metric("Confidence", f"{pred:.2%}")
        m2.metric("Signal", signal)
        m3.metric("Data From", str(date_used))

    if position:
        pl_color = "normal" if position["unrealised_pl"] >= 0 else "inverse"
        m4.metric("Position", f"{position['qty']} shares")
        m5.metric(
            "Unrealised P&L",
            f"${position['unrealised_pl']:+.2f}",
            delta=f"{position['pl_pct']:+.2f}%",
            delta_color=pl_color,
        )
    else:
        m4.metric("Position", "None")
        m5.metric("Unrealised P&L", "—")

    # Recent orders for this ticker
    with st.expander(f"Recent {ticker} Orders"):
        orders = traders[ticker].get_recent_orders()
        if not orders.empty:
            st.dataframe(orders, width="stretch")
        else:
            st.info(f"No recent orders for {ticker}")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        if st.button(f"▶️ Open {ticker} Position", key=f"open_{ticker}", use_container_width=True):
            result = traders[ticker].open_position(scalers[ticker])
            st.info(result)
    with c2:
        if st.button(
            f"⏹ Close {ticker} Position",
            key=f"close_{ticker}",
            use_container_width=True,
        ):
            result = traders[ticker].close_position()
            st.info(result)

    st.divider()

# ---- Global controls ----
st.subheader("Global Controls")
g1, g2 = st.columns(2)
with g1:
    if st.button("⏹ Close ALL Positions", width="stretch"):
        trading_client.close_all_positions()
        st.success("All positions closed")
with g2:
    if st.button("🔄 Refresh Dashboard", width="stretch"):
        st.rerun()


# Check what time your server thinks it is
print(f"Server local time: {datetime.now().strftime('%H:%M:%S')}")
print(f"ET time:           {datetime.now(ET).strftime('%H:%M:%S')}")


# if 'bot_started' not in st.session_state:
#     st.session_state['bot_started'] = True
#     thread = threading.Thread(target=bot_schedule_loop, daemon=True, name="trading_bot")
#     thread.start()
#     print(f"Bot thread started: {thread.name} | Alive: {thread.is_alive()}")


# # Verify thread is still alive on each rerun
# threads = {t.name: t for t in threading.enumerate()}
# if "trading_bot" in threads:
#     st.sidebar.success("🟢 Bot thread running")
# else:
#     st.sidebar.error("🔴 Bot thread not running — restart app")

with st.expander("Bot Diagnostics"):
    st.write(f"Server time (local): {datetime.now().strftime('%H:%M:%S')}")
    st.write(f"Server time (ET):    {datetime.now(ET).strftime('%H:%M:%S')}")
    st.caption("Scheduled Jobs")
    if schedule.jobs:
        for job in schedule.jobs:
            # Extract the useful parts — day, time and function name
            day  = str(job.next_run.strftime('%A'))   # e.g. "Monday"
            time_str = str(job.next_run.strftime('%H:%M UTC'))
            func_name = job.job_func.__name__ if hasattr(job.job_func, '__name__') else str(job.job_func)
            st.write(f"• {day} {time_str} → {func_name}")
    else:
        st.warning("No jobs scheduled")


# ---- Auto refresh ----
st.caption("Auto-refreshes every 30 minutes")
time.sleep(1800)
st.rerun()
