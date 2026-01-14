import streamlit as st
import yfinance as yf
import pandas as pd
from fredapi import Fred
import plotly.express as px
import plotly.graph_objects as go
import time
import random
import requests
from datetime import datetime

# --- 1. 系統初始化與戰略名單 ---
st.set_page_config(page_title="Posa Alpha 3.8 (On-chain)", layout="wide")
st.title("🛡️ Posa Alpha 3.8: 鏈上週期與跨市場終極審計")

# Seeking Alpha 2026 金股與幣圈核心
SA_TOP_10 = ['MU', 'AMD', 'CLS', 'CIEN', 'COHR', 'ALL', 'INCY', 'GOLD', 'WLDN', 'ATI']
CRYPTO_CORE = ['BTC-USD', 'ETH-USD', 'SOL-USD']
BENCHMARKS = ['QQQ', '0050.TW', '^VIX', '^MOVE']

try:
    FRED_API_KEY = st.secrets["FRED_API_KEY"]
    fred = Fred(api_key=FRED_API_KEY)
except:
    st.error("❌ 請在 Secrets 設定 FRED_API_KEY")
    st.stop()

# --- 2. 側邊欄：資產配置與風險設定 ---
st.sidebar.header("💰 實戰資產配置 (12.7萬戰略部隊)")
if 'portfolio_df' not in st.session_state:
    st.session_state.portfolio_df = pd.DataFrame([
        {"代號": "MU", "金額": 30000},
        {"代號": "AMD", "金額": 25000},
        {"代號": "0050.TW", "金額": 40000},
        {"代號": "SOL-USD", "金額": 32000}
    ])
edited_df = st.sidebar.data_editor(st.session_state.portfolio_df, num_rows="dynamic")
user_tickers = edited_df["代號"].tolist()
total_val = edited_df["金額"].sum()

TRAILING_PCT = st.sidebar.slider("移動止損 (%)", 5, 15, 7) / 100
KELLY_SCALE = st.sidebar.slider("凱利縮放係數 (建議 0.5)", 0.1, 1.0, 0.5)

# --- 3. 真實數據抓取模組 (含 BTC.D 與 MVRV) ---
@st.cache_data(ttl=3600)
def fetch_onchain_metrics():
    """從 CoinGecko 與 Blockchain.com 抓取真實鏈上數據"""
    try:
        # BTC.D (CoinGecko)
        global_data = requests.get("https://api.coingecko.com/api/v3/global", timeout=10).json()
        btc_d = global_data['data']['market_cap_percentage']['btc']
        
        # MVRV (Blockchain.com 代理)
        # 註：此為比特幣週期的核心指標，若 API 暫時失效則返回保守值 2.1
        mvrv_data = requests.get("https://api.blockchain.info/charts/mvrv?timespan=1year&format=json", timeout=10).json()
        current_mvrv = mvrv_data['values'][-1]['y']
    except:
        btc_d, current_mvrv = 52.5, 2.1 # 預設安全值
    return btc_d, current_mvrv

@st.cache_data(ttl=600)
def fetch_market_data(tickers):
    prices, info = pd.DataFrame(), {}
    full_list = list(set(tickers + SA_TOP_10 + CRYPTO_CORE + BENCHMARKS))
    for t in full_list:
        try:
            time.sleep(0.3)
            tk = yf.Ticker(t)
            df = tk.history(period="2y")
            if not df.empty:
                prices[t] = df['Close']
                info[t] = {"price": df['Close'].iloc[-1], "change": (df['Close'].iloc[-1]/df['Close'].iloc[-2]-1)*100}
        except: continue
    
    try:
        liq = (fred.get_series('WALCL').iloc[-1] - fred.get_series('WTREGEN').iloc[-1] - fred.get_series('RRPONTSYD').iloc[-1]) / 1000
    except: liq = 0
    return liq, prices, info

# --- 4. 凱利與趨勢審計邏輯 ---
def get_audit_stats(t_prices, q_prices):
    ema20 = t_prices.ewm(span=20).mean()
    rs = t_prices / q_prices
    sig = (t_prices > ema20) & (rs > rs.rolling(20).mean())
    rets = t_prices.shift(-5) / t_prices - 1
    v_rets = rets[sig].