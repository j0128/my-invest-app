import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import time
import plotly.express as px
import plotly.graph_objects as go
from fredapi import Fred
from datetime import datetime

# --- 1. 系統初始化與 Seeking Alpha 數據庫 ---
st.set_page_config(page_title="Posa Alpha 3.9.1", layout="wide")
st.title("🛡️ Posa Alpha 3.9.1: 全功能終極審計中心")

# SA 知識庫
SA_INSIGHTS = {
    'MU': {'note': 'HBM 領先, PEG 0.20x', 'growth': '206%'},
    'CLS': {'note': '15次盈餘上修, AI核心', 'growth': '51%'},
    'AMD': {'note': 'OpenAI 夥伴, M1400 加速器', 'growth': '34%'},
    'ALL': {'note': '連續 32 年配息, 高品質保險', 'growth': '193%'},
    'GOLD': {'note': '金+銅 雙避險, 能源轉型受益', 'growth': '58%'}
}

try:
    FRED_API_KEY = st.secrets["FRED_API_KEY"]
    fred = Fred(api_key=FRED_API_KEY)
except Exception:
    st.error("❌ 請在 Secrets 設定 FRED_API_KEY")
    st.stop()

# --- 2. 數據抓取模組 (真實鏈上與市場數據) ---
@st.cache_data(ttl=3600)
def fetch_onchain_metrics():
    try:
        # BTC.D (CoinGecko)
        btc_d = requests.get("https://api.coingecko.com/api/v3/global", timeout=10).json()['data']['market_cap_percentage']['btc']
        # MVRV (Blockchain.com)
        mvrv_data = requests.get("https://api.blockchain.info/charts/mvrv?timespan=1year&format=json", timeout=10).json()
        current_mvrv = mvrv_data['values'][-1]['y']
    except Exception:
        btc_d, current_mvrv = 52.5, 2.1
    return btc_d, current_mvrv

@st.cache_data(ttl=600)
def fetch_master_data(tickers):
    processed = [t.upper() if ".TW" in t.upper() else t for t in tickers]
    benchmarks = ['QQQ', '0050.TW', '^VIX', '^MOVE', 'BTC-USD']
    full_list = list(set(processed + benchmarks))
    
    # 抓取 1 年資料確保 EMA 穩定
    data = yf.download(full_list, period="1y", interval="1d", auto_adjust=True, progress=False)
    prices = data['Close'].ffill()
    
    earnings = {}
    for t in processed:
        if "-" not in t and ".TW" not in t:
            try:
                cal = yf.Ticker(t).calendar
                if cal is not None and not cal.empty:
                    earnings[t] = cal.loc['Earnings Date'].iloc[0].strftime('%Y-%m-%d')
            except Exception: pass
    return prices, earnings

# --- 3. 側邊欄設定 ---
st.sidebar.header("💰 12.7萬實戰配置")
if 'portfolio_df' not in st.session_state:
    st.session_state.portfolio_df = pd.DataFrame([
        {"代號": "MU", "金額": 30000}, {"代號": "AMD", "金額": 25000},
        {"代號": "0050.TW", "金額": 40000}, {"代號": "BTC-USD", "金額": 32000}
    ])
edited_df = st.sidebar.data_editor(st.session_state.portfolio_df, num_rows="dynamic")
user_tickers = edited_df["代號"].tolist()
total_val = edited_df["金額"].sum()

# --- 4. 執行與渲染 ---
try:
    prices, earnings_dates = fetch_master_data(user_tickers)
    btc_d, mvrv = fetch_onchain_metrics()
    net_liq = (fred.get_series('WALCL').iloc[-1] - fred.get_series('WTREGEN').iloc[-1] - fred.get_series('RRPONTSYD').iloc[-1]) / 1000
    
    # A. 宏觀看板 (修正 Gauge 顯示)
    st.subheader("🌐 全球週期與地基審計")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MVRV (週期溫度)", f"{mvrv:.2f}", delta="過熱" if mvrv > 3 else "安全")
    m2.metric("BTC.D (資金羅盤)", f"{btc_d:.1f}%")
    m3.metric("VIX (股市天氣)", f"{prices['^VIX'].iloc[-1]:.2f}")
    m4.metric("淨流動性", f"${net_liq:.2f}B")

    # B. 即時脈搏 (每行 4 檔解決位數遮斷)
    st.divider()
    st.subheader("⚡ 即時市場脈搏")
    for i in range(0, len(user_tickers), 4):
        cols = st.columns(4)
        for j, t in enumerate(user_tickers[i:i+4]):
            if t in prices.columns:
                curr = prices[t].iloc[-1]
                chg = (prices[t].iloc[-1]/prices[t].iloc[-2]-1)*100
                cols[j].metric(t, f"${curr:,.2f}", f"{chg:.2f}%")

    # C. 深度審計表 (整合所有指標與走勢預判)
    st.subheader("📋 跨市場深度審計與走勢預判")
    audit_data = []
    today = datetime.now().date()
    for t in user_tickers:
        if t not in prices.columns or t in ['^VIX', '^MOVE', 'QQQ']: continue
        curr = prices[t].iloc[-1]
        ema20 = prices[t].ewm(span=20).mean().iloc[-1]
        
        # 贏過 QQQ & 0050
        win_qqq = (prices[t]/prices['QQQ']).iloc[-1] > (prices[t]/prices['QQQ']).rolling(20).mean().iloc[-1]
        win_0050 = (prices[t]/prices['0050.TW']).iloc[-1] > (prices[t]/prices['0050.TW']).rolling(20).mean().iloc[-1] if '0050.TW' in prices.columns else False
        
        # 凱利勝率
        rets = prices[t].shift(-5) / prices[t] - 1
        sig = (prices[t] > prices[t].ewm(span=20).mean())
        v_rets = rets[sig].dropna() # 修正了截圖中的點語法錯誤