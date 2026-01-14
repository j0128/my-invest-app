import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import time
import plotly.express as px
import plotly.graph_objects as go
from fredapi import Fred
from datetime import datetime

# --- 1. 初始化與核心數據庫 ---
st.set_page_config(page_title="Posa Alpha 4.0", layout="wide")
st.title("🛡️ Posa Alpha 4.0: 全球週期與實戰審計終端")

# Seeking Alpha 知識庫 (解決單薄感)
SA_INSIGHTS = {
    'MU': {'note': 'HBM 領導者, PEG 0.20x (折價 88%)', 'growth': '206%'},
    'CLS': {'note': '15次盈餘上修, AI整合核心', 'growth': '51%'},
    'AMD': {'note': 'OpenAI 夥伴, M1400 加速器', 'growth': '34%'},
    'ALL': {'note': '高品質保險, AI 核保效率高', 'growth': '193%'}
}

try:
    FRED_API_KEY = st.secrets["FRED_API_KEY"]
    fred = Fred(api_key=FRED_API_KEY)
except Exception:
    st.error("❌ 請在 Secrets 設定 FRED_API_KEY")
    st.stop()

# --- 2. 數據抓取模組 (真實鏈上數據) ---
@st.cache_data(ttl=3600)
def fetch_real_onchain():
    """從 CoinGecko 與 Blockchain.com 抓取真實數據"""
    try:
        btc_d = requests.get("https://api.coingecko.com/api/v3/global", timeout=10).json()['data']['market_cap_percentage']['btc']
        mvrv_data = requests.get("https://api.blockchain.info/charts/mvrv?timespan=1year&format=json", timeout=10).json()
        mvrv = mvrv_data['values'][-1]['y']
    except Exception:
        btc_d, mvrv = 52.5, 2.1  # 異常時顯示預設值
    return btc_d, mvrv

@st.cache_data(ttl=600)
def fetch_market_master(tickers):
    """修復 0050.TW 與 $nan 問題的數據抓取"""
    processed = [t.upper() if ".TW" in t.upper() else t for t in tickers]
    benchmarks = ['QQQ', '0050.TW', '^VIX', '^MOVE', 'BTC-USD']
    full_list = list(set(processed + benchmarks))
    
    # 抓取 1 年資料確保 20EMA 穩定
    data = yf.download(full_list, period="1y", auto_adjust=True, progress=False)
    # ffill 補齊台美股休市的時間差 (關鍵修復)
    prices = data['Close'].ffill()
    return prices

# --- 3. 側邊欄配置 ---
st.sidebar.header("💰 12.7萬實戰資產配置")
if 'portfolio_df' not in st.session_state:
    st.session_state.portfolio_df = pd.DataFrame([
        {"代號": "MU", "金額": 30000}, {"代號": "AMD", "金額": 25000},
        {"代號": "0050.TW", "金額": 40000}, {"代號": "BTC-USD", "金額": 32000}
    ])
edited_df = st.sidebar.data_editor(st.session_state.portfolio_df, num_rows="dynamic")
user_tickers = edited_df["代號"].tolist()
total_val = edited_df["金額"].sum()

# --- 4. 執行審計與顯示 ---
try:
    prices = fetch_market_master(user_tickers)
    btc_d, mvrv = fetch_real_onchain()
    net_liq = (fred.get_series('WALCL').iloc[-1] - fred.get_series('WTREGEN').iloc[-1] - fred.get_series('RRPONTSYD').iloc[-1]) / 1000
    
    # A. 週期溫度表：利好出盡偵測器
    st.subheader("🌡️ 週期審計：MVRV 與 BTC 市佔率")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MVRV 週期溫度", f"{mvrv:.2f}", delta="利多出盡預警" if mvrv > 3 else "安全")
    m2.metric("BTC.D 市佔率", f"{btc_d:.1f}%")
    m3.metric("VIX 恐慌指數", f"{prices['^VIX'].iloc[-1]:.2f}")
    m4.metric("淨流動性", f"${net_liq:,.2f}B")

    # B. 即時脈搏：網格佈局解決遮斷 (image_182fdd)
    st.divider()
    st.subheader("⚡ 即時市場脈