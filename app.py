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

# --- 1. 系統初始化 ---
st.set_page_config(page_title="Posa Alpha 3.8 (On-chain)", layout="wide")
st.title("🛡️ Posa Alpha 3.8: 鏈上週期與跨市場終極審計")

# Seeking Alpha 十大金股與核心標的
SA_TOP_10 = ['MU', 'AMD', 'CLS', 'CIEN', 'COHR', 'ALL', 'INCY', 'GOLD', 'WLDN', 'ATI']
BENCHMARKS = ['QQQ', '0050.TW', '^VIX', 'BTC-USD', 'SOL-USD', 'ETH-USD']

try:
    FRED_API_KEY = st.secrets["FRED_API_KEY"]
    fred = Fred(api_key=FRED_API_KEY)
except:
    st.error("❌ 請在 Secrets 設定 FRED_API_KEY")
    st.stop()

# --- 2. 數據抓取：真實 BTC.D 與 MVRV ---
@st.cache_data(ttl=3600)
def fetch_onchain_data():
    """自動抓取鏈上數據"""
    try:
        # BTC.D 從 CoinGecko 抓取
        global_resp = requests.get("https://api.coingecko.com/api/v3/global", timeout=10).json()
        btc_d = global_resp['data']['market_cap_percentage']['btc']
        
        # MVRV 從 Blockchain.com 代理抓取
        mvrv_resp = requests.get("https://api.blockchain.info/charts/mvrv?timespan=1year&format=json", timeout=10).json()
        current_mvrv = mvrv_resp['values'][-1]['y']
    except:
        btc_d, current_mvrv = 52.5, 2.1 # 預設安全值
    return btc_d, current_mvrv

@st.cache_data(ttl=300)
def fetch_market_data(tickers):
    prices, info = pd.DataFrame(), {}
    full_list = list(set(tickers + SA_TOP_10 + BENCHMARKS))
    for t in full_list:
        try:
            time.sleep(0.3)
            tk = yf.Ticker(t)
            df = tk.history(period="2y") # 增加歷史深度修復 $nan
            if not df.empty:
                prices[t] = df['Close']
                curr_p = df['Close'].iloc[-1]
                change = (curr_p / df['Close'].iloc[-2] - 1) * 100
                info[t] = {"price": curr_p, "change": change}
        except: continue
    
    try:
        liq = (fred.get_series('WALCL').iloc[-1] - fred.get_series('WTREGEN').iloc[-1] - fred.get_series('RRPONTSYD').iloc[-1]) / 1000
    except: liq = 0
    return liq, prices, info

# --- 3. 介面渲染：解決數據擠壓 ---
try:
    # 獲取配置與數據
    user_tickers = st.sidebar.multiselect("選擇持倉標的", SA_TOP_10 + BENCHMARKS, default=['MU', 'AMD', '0050.TW', 'BTC-USD'])
    net_liq, prices, market_info = fetch_market_data(user_tickers)
    btc_d, mvrv = fetch_onchain_data()
    vix = prices['^VIX'].iloc[-1]

    # A. 頂部看板：加入 MVRV 與 BTC.D
    st.subheader("🌡️ 週期與情緒審計 (利好出盡偵測器)")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("MVRV 週期溫度", f"{mvrv:.2f}", delta="過熱" if mvrv > 3.0 else "安全")
    m_col2.metric("BTC.D 市佔率", f"{btc_d:.1f}%")
    m_col3.metric("VIX 天氣", f"{vix:.2f}")
    m_col4.metric("淨流動性", f"${net_liq:.2f}B")

    # B. 即時脈搏 (每行 4 檔，解決位數遮斷)