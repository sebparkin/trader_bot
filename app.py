import streamlit as st
import torch
import joblib
import pandas as pd
import pytz
from datetime import datetime
import plotly.graph_objects as go
import time
import schedule
from alpaca.broker import BrokerClient
from core.data_handler import DataHandler
from core.paper_trader import PaperTrader
from core.trading_lstm import TradingLTSM
from core.constants import *

ET = pytz.timezone("America/New_York")

st.set_page_config(
    page_title = "Trading Bot Dashboard",
    page_icon =  "🚀",
    layout = "wide"
)

@st.cache_resource
def load_model_and_scaler(ticker):
    model = TradingLTSM(
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        feature_cols=FEATURES
    )

    ticker = ticker.lower()

    model.load_state_dict(torch.load(f'models/{ticker}_model.pth', map_location='cpu'))
    model.eval()

    scaler = joblib.load(f'models/{ticker}_scaler.pkl')
    print("Model and scaler loaded")
    return model, scaler

@st.cache_resource
def load_papertrader(_model, ticker):
    return PaperTrader(_model, ticker, cash_fraction=CASH_FRACTION, threshold=THRESHOLD)

tickers = ['AAPL', 'MSFT']
models = {}
scalers = {}
traders = {}
for ticker in tickers:
    models[ticker], scalers[ticker] =  load_model_and_scaler(ticker)
    traders[ticker] = load_papertrader(models[ticker], ticker)

trading_client = list(traders.values())[0].trading_client

def get_account_stats():
    account = trading_client.get_account()
    return {
        'portfolio_value' : float(account.portfolio_value),
        'cash'            : float(account.cash),
        'equity'          : float(account.equity),
        'day_pl'          : float(account.equity) - float(account.last_equity),
        'buying_power'    : float(account.buying_power)
    }

def get_position(ticker):
    try:
        pos = trading_client.get_position(ticker)
        return {
            'qty'           : pos.qty,
            'entry_price'   : float(pos.avg_entry_price),
            'current_price' : float(pos.current_price),
            'unrealised_pl' : float(pos.unrealized_pl),
            'pl_pct'        : float(pos.unrealized_plpc) * 100
        }
    except:
        return None

def get_portfolio_history():

    history = trading_client.get_portfolio_history()
    df = pd.DataFrame({
        'Date' : pd.to_datetime(history.timestamp, unit='s'),
        'Value': history.equity
    })
    return df

def get_recent_orders():
    orders = trading_client.get_orders(status='all', limit=20)
    if not orders:
        return pd.DataFrame()
    return pd.DataFrame([{
        'Time'  : pd.to_datetime(o.submitted_at).strftime('%Y-%m-%d %H:%M'),
        'Symbol': o.symbol,
        'Side'  : o.side.upper(),
        'Amount': f"${float(o.notional):.2f}" if o.notional else f"{o.qty} shares",
        'Status': o.status
    } for o in orders])

def get_prediction_now(ticker):
    try:
        pred, date_used = models[ticker].get_yesterdays_prediction(
            ticker,
            scalers[ticker]
        )
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
col2.metric("Cash",            f"${stats['cash']:,.2f}")
col3.metric("Day P&L",         f"${stats['day_pl']:+,.2f}",
            delta_color="normal" if stats['day_pl'] >= 0 else "inverse")
col4.metric("Buying Power",    f"${stats['buying_power']:,.2f}")

# Portfolio chart
st.subheader("Portfolio History (1 Month)")
history_df = get_portfolio_history()
fig = go.Figure()
fig.add_trace(go.Scatter(
    x    = history_df['Date'],
    y    = history_df['Value'],
    mode = 'lines',
    fill = 'tozeroy',
    line = dict(color='#00C805')
))
fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
st.plotly_chart(fig, width='stretch')

st.divider()

# ---- Per-ticker sections ----
TICKER_COLORS = {'AAPL': "#B93939", 'MSFT': '#00A4EF'}

for ticker in tickers:
    color = TICKER_COLORS.get(ticker, '#888888')
    st.markdown(f"<h2 style='color:{color}'>{ticker}</h2>", unsafe_allow_html=True)

    pred, date_used, err = get_prediction_now(ticker)
    position             = get_position(ticker)

    # Prediction + position metrics
    m1, m2, m3, m4, m5 = st.columns(5)

    if err:
        m1.error(f"Prediction error: {err}")
    else:
        signal = ("📈 LONG"    if pred > THRESHOLD
                  else "📉 SHORT" if pred < (1 - THRESHOLD)
                  else "⏸ NEUTRAL")
        m1.metric("Confidence", f"{pred:.2%}")
        m2.metric("Signal",     signal)
        m3.metric("Data From",  str(date_used))

    if position:
        pl_color = "normal" if position['unrealised_pl'] >= 0 else "inverse"
        m4.metric("Position",      f"{position['qty']} shares")
        m5.metric("Unrealised P&L",f"${position['unrealised_pl']:+.2f}",
                  delta=f"{position['pl_pct']:+.2f}%",
                  delta_color=pl_color)
    else:
        m4.metric("Position",       "None")
        m5.metric("Unrealised P&L", "—")

    # Recent orders for this ticker
    with st.expander(f"Recent {ticker} Orders"):
        all_orders = trading_client.get_orders()
        ticker_orders = [o for o in all_orders if o.symbol == ticker]
        if ticker_orders:
            orders_df = pd.DataFrame([{
                'Time'  : pd.to_datetime(o.submitted_at).strftime('%Y-%m-%d %H:%M'),
                'Side'  : o.side.upper(),
                'Amount': f"${float(o.notional):.2f}" if o.notional else f"{o.qty} shares",
                'Status': o.status
            } for o in ticker_orders])
            st.dataframe(orders_df, width='stretch')
        else:
            st.info(f"No recent orders for {ticker}")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        if st.button(f"▶️ Open {ticker} Position", 
                     key=f"open_{ticker}", 
                     use_container_width=True):
            result = traders[ticker].open_position(scalers[ticker])
            st.info(result)
    with c2:
        if st.button(f"⏹ Close {ticker} Position", 
                     key=f"close_{ticker}", 
                     use_container_width=True):
            result = traders[ticker].close_position()
            st.info(result)

    st.divider()

# ---- Global controls ----
st.subheader("Global Controls")
g1, g2 = st.columns(2)
with g1:
    if st.button("⏹ Close ALL Positions", width='stretch'):
        trading_client.close_all_positions()
        st.success("All positions closed")
with g2:
    if st.button("🔄 Refresh Dashboard", width='stretch'):
        st.rerun()

# ---- Auto refresh ----
st.caption("Auto-refreshes every 30 minutes")
time.sleep(1800)
st.rerun()



def bot_schedule_loop():

    for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']:
        getattr(schedule.every(), day).at("14:31").do(traders["AAPL"].open_position(scalers["AAPL"]))
        getattr(schedule.every(), day).at("14:31").do(traders["MSFT"].open_position(scalers["MSFT"]))
        getattr(schedule.every(), day).at("20:45").do(traders["AAPL"].close_position())

    while True:
        schedule.run_pending()
        time.sleep(30)

# Start bot in background thread when Streamlit loads
# st.session_state ensures it only starts once
if 'bot_started' not in st.session_state:
    thread = threading.Thread(target=bot_schedule_loop, daemon=True)
    thread.start()
    st.session_state['bot_started'] = True