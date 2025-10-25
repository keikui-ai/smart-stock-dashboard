# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ----------------------------
# Utility Functions
# ----------------------------

def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def detect_fvg(df):
    fvg_bullish = []
    fvg_bearish = []
    for i in range(2, len(df)):
        if df['Low'].iloc[i] > df['High'].iloc[i-2] and df['Low'].iloc[i-1] > df['High'].iloc[i-2]:
            fvg_bullish.append(i)
        if df['High'].iloc[i] < df['Low'].iloc[i-2] and df['High'].iloc[i-1] < df['Low'].iloc[i-2]:
            fvg_bearish.append(i)
    return fvg_bullish, fvg_bearish

def detect_market_structure(df):
    hh = (df['High'] > df['High'].shift(1)) & (df['High'].shift(1) > df['High'].shift(2))
    hl = (df['Low'] > df['Low'].shift(1)) & (df['Low'].shift(1) > df['Low'].shift(2))
    return hh, hl

def run_backtest(df, risk_tolerance="Medium"):
    df = df.copy()
    df['RSI'] = calculate_rsi(df['Close'])
    hh, hl = detect_market_structure(df)
    fvg_bull, _ = detect_fvg(df)
    
    df['FVG_Support'] = np.nan
    for i in fvg_bull:
        if i < len(df):
            df.loc[df.index[i]:, 'FVG_Support'] = df['Low'].iloc[i-2]

    position = 0
    portfolio = [1.0]
    positions = [0]

    for i in range(2, len(df)):
        close = df['Close'].iloc[i]
        rsi = df['RSI'].iloc[i]
        above_fvg = not pd.isna(df['FVG_Support'].iloc[i]) and close > df['FVG_Support'].iloc[i]
        bullish_struct = hh.iloc[i] and hl.iloc[i]
        bearish_struct = (df['High'].iloc[i] < df['High'].iloc[i-1]) and (df['Low'].iloc[i] < df['Low'].iloc[i-1])

        if position == 0 and bullish_struct and above_fvg and 50 < rsi < 70:
            position = 1
        elif position == 1 and (rsi > 70 or bearish_struct):
            position = 0

        # Daily return
        daily_ret = df['Close'].iloc[i] / df['Close'].iloc[i-1]
        if position == 1:
            portfolio.append(portfolio[-1] * daily_ret)
        else:
            portfolio.append(portfolio[-1])
        positions.append(position)

    portfolio = portfolio[:len(df)]
    df_bt = df.copy()
    df_bt['Portfolio'] = portfolio
    df_bt['Buy_and_Hold'] = df['Close'] / df['Close'].iloc[0]
    df_bt['Position'] = [0] + positions[:-1]

    returns = df_bt['Portfolio'].pct_change().dropna()
    cumulative = df_bt['Portfolio'].iloc[-1]
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
    rolling_max = df_bt['Portfolio'].cummax()
    drawdown = (df_bt['Portfolio'] - rolling_max) / rolling_max
    max_dd = drawdown.min() if not drawdown.empty else 0

    return df_bt, {
        "Cumulative Return": f"{(cumulative - 1) * 100:.2f}%",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Max Drawdown": f"{max_dd * 100:.2f}%"
    }

# ----------------------------
# Dashboard
# ----------------------------

st.set_page_config(page_title="Smart Stock Analyzer", layout="wide")
st.title("📈 Smart Stock Analyzer & Backtester")
st.sidebar.header("⚙️ Settings")

ticker = st.sidebar.text_input("Stock Ticker", value="NVDA").upper().strip()
timeframe = st.sidebar.selectbox("Historical Period", ["6mo", "1y", "2y", "5y"], index=1)
enable_backtest = st.sidebar.checkbox("Enable Backtesting", True)

if not ticker:
    st.warning("Please enter a stock ticker (e.g., AAPL, NVDA).")
    st.stop()

@st.cache_data(ttl=3600)
def load_data(ticker, period):
    data = yf.download(ticker, period=period, interval="1d", progress=False)
    return data

with st.spinner(f"Fetching data for {ticker}..."):
    df = load_data(ticker, timeframe)

# 🔴 CRITICAL: Validate data
if df is None or df.empty:
    st.error(f"❌ No data found for ticker **{ticker}** with period **{timeframe}**. Try a different symbol or longer period.")
    st.stop()

# Ensure expected columns exist
expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
if not all(col in df.columns for col in expected_cols):
    st.error(f"❌ Incomplete data received. Missing columns: {set(expected_cols) - set(df.columns)}")
    st.stop()

# Now safe to use df['Close']
latest_price = df['Close'].iloc[-1]
prev_close = df['Close'].iloc[-2] if len(df) > 1 else latest_price
change_pct = ((latest_price - prev_close) / prev_close * 100) if prev_close != 0 else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Price", f"${latest_price:.2f}")
col2.metric("24h Change", f"{change_pct:.2f}%", delta_color="normal")
col3.metric("Period High", f"${df['High'].max():.2f}")
col4.metric("Period Low", f"${df['Low'].min():.2f}")

# Backtesting
if enable_backtest and len(df) > 30:
    try:
        with st.spinner("Running backtest..."):
            df_bt, perf = run_backtest(df)
        
        st.subheader("📊 Backtest Results")
        c1, c2, c3 = st.columns(3)
        c1.metric("Cumulative Return", perf["Cumulative Return"])
        c2.metric("Sharpe Ratio", perf["Sharpe Ratio"])
        c3.metric("Max Drawdown", perf["Max Drawdown"])

        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=df_bt.index, y=df_bt['Portfolio'], name='Strategy', line=dict(color='green')))
        fig_bt.add_trace(go.Scatter(x=df_bt.index, y=df_bt['Buy_and_Hold'], name='Buy & Hold', line=dict(color='blue')))
        fig_bt.update_layout(title="Equity Curve", height=350, showlegend=True)
        st.plotly_chart(fig_bt, use_container_width=True)
    except Exception as e:
        st.warning(f"⚠️ Backtest failed (insufficient data or error): {str(e)}")

# Live analysis
df_live = df.copy()
df_live['RSI'] = calculate_rsi(df_live['Close'])
hh, hl = detect_market_structure(df_live)
fvg_bull, fvg_bear = detect_fvg(df_live)

last_idx = -1
bullish = hh.iloc[last_idx] and hl.iloc[last_idx]
rsi_val = df_live['RSI'].iloc[last_idx]
above_fvg = len(fvg_bull) > 0 and df_live['Close'].iloc[last_idx] > df_live['Low'].iloc[fvg_bull[-1] - 2]

decision = "Hold"
if bullish and above_fvg and 50 < rsi_val < 70:
    decision = "Buy"
elif rsi_val > 75:
    decision = "Sell"

st.subheader(f"🧠 Recommendation: {ticker}")
st.markdown(f"### **Action: {decision}**")
st.markdown(f"- RSI: {rsi_val:.1f}")
st.markdown(f"- Market Structure: {'✅ Bullish' if bullish else '⚠️ Neutral/Bearish'}")
st.markdown(f"- Price vs FVG: {'🟢 Above support' if above_fvg else '🔴 Below/No FVG'}")

# Chart
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)

# FVG zones (last 3)
for i in fvg_bull[-3:]:
    if 2 <= i < len(df):
        fig.add_shape(type="rect", x0=df.index[i-2], x1=df.index[i],
                      y0=df['Low'].iloc[i-2], y1=df['High'].iloc[i],
                      fillcolor="green", opacity=0.15, line_width=0, row=1, col=1)
for i in fvg_bear[-3:]:
    if 2 <= i < len(df):
        fig.add_shape(type="rect", x0=df.index[i-2], x1=df.index[i],
                      y0=df['Low'].iloc[i], y1=df['High'].iloc[i-2],
                      fillcolor="red", opacity=0.15, line_width=0, row=1, col=1)

fig.add_trace(go.Scatter(x=df.index, y=df_live['RSI'], mode='lines', name='RSI'), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
fig.update_layout(height=650, title=f"{ticker} Price & RSI", xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

st.caption("💡 Strategy based on FVG, market structure, and RSI. Not financial advice. Use at your own risk.")
