import streamlit as st
import yfinance as yf
import pandas as pd
from fredapi import Fred
import plotly.express as px
import time
import random

# --- 1. 2026 語法修正與初始化 ---
st.set_page_config(page_title="Posa x biibo Alpha 2.2", layout="wide")
st.title("📈 Posa x biibo 投資風險審計儀表板")

# 從 Secrets 讀取 API Key
try:
    FRED_API_KEY = st.secrets["FRED_API_KEY"]
    fred = Fred(api_key=FRED_API_KEY)
except Exception:
    st.error("❌ 找不到 FRED_API_KEY！請檢查 Streamlit Secrets 設定。")
    st.stop()

# --- 2. 側邊欄 ---
st.sidebar.header("⚙️ 參數設定")
ticker_input = st.sidebar.text_input("輸入個股代號 (空白分隔)", "AMD CLS URA VRTX").upper()
user_tickers = list(set(ticker_input.split()))

# --- 3. 強化版數據抓取 (增加延遲躲避 Rate Limit) ---
def fetch_with_retry(ticker):
    # 隨機延遲 1-3 秒，避免被 Yahoo 偵測為機器人
    time.sleep(random.uniform(1, 3))
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False)
        if df.empty: return None, None
        # 修正 2026 年的欄位處理
        close = df['Close'].iloc[:, 0] if isinstance(df['Close'], pd.DataFrame) else df['Close']
        vol = df['Volume'].iloc[:, 0] if isinstance(df['Volume'], pd.DataFrame) else df['Volume']
        return close, vol
    except Exception as e:
        st.warning(f"⚠️ {ticker} 抓取超時，重試中... ({e})")
        return None, None

@st.cache_data(ttl=3600)
def fetch_all_data(tickers):
    # A. 抓取宏觀流動性 (FRED)
    try:
        walcl = fred.get_series('WALCL').iloc[-1]
        tga = fred.get_series('WTREGEN').iloc[-1]
        rrp = fred.get_series('RRPONTSYD').iloc[-1]
        net_liq = (walcl - tga - rrp) / 1000 
    except: net_liq = 0

    # B. 逐一抓取
    prices = pd.DataFrame()
    volumes = pd.DataFrame()
    core_symbols = ['QQQ', '^VIX', '^MOVE']
    for t in list(set(core_symbols + tickers)):
        p, v = fetch_with_retry(t)
        if p is not None:
            prices[t] = p
            volumes[t] = v
    return net_liq, prices, volumes

# --- 4. 執行與顯示 ---
try:
    with st.spinner('biibo 正在進行數據審計...'):
        net_liq, prices, volumes = fetch_all_data(user_tickers)

    if prices.empty:
        st.error("❌ Yahoo Finance 目前拒絕連線，請 5 分鐘後重新整理頁面。")
        st.stop()

    # 指標看板
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("美元淨流動性", f"${net_liq:.2f}B" if net_liq > 0 else "抓取中")
    
    if '^VIX' in prices.columns:
        v = prices['^VIX'].dropna().iloc[-1]
        m2.metric("VIX 恐慌指數", f"{v:.2f}", delta="危險" if v > 22 else "安全", delta_color="inverse")
    
    if '^MOVE' in prices.columns:
        m = prices['^MOVE'].dropna().iloc[-1]
        m3.metric("MOVE 債券預警", f"{m:.2f}")
    
    if 'QQQ' in prices.columns:
        q = prices['QQQ'].dropna().iloc[-1]
        q_ema = prices['QQQ'].ewm(span=20).mean().iloc[-1]
        m4.metric("QQQ 狀態", f"${q:.1f}", delta=f"{((q/q_ema)-1)*100:.2f}% (vs EMA20)")

    st.divider()

    # 安全審計清單
    st.subheader("🔍 個股安全性審計")
    audit_list = []
    for t in user_tickers:
        if t not in prices.columns or t in ['QQQ', '^VIX', '^MOVE']: continue
        ema = prices[t].ewm(span=20).mean().iloc[-1]
        rs = prices[t] / prices['QQQ']
        rs_trend = "↗️ 強勢" if rs.iloc[-1] > rs.rolling(20).mean().iloc[-1] else "↘️ 弱勢"
        score = 0
        if prices[t].iloc[-1] > ema: score += 4
        if "↗️" in rs_trend: score += 3
        if '^VIX' in prices.columns and prices['^VIX'].iloc[-1] < 18: score += 3
        
        audit_list.append({
            "標的": t, "安全得分": f"{score}/10", 
            "20EMA": "🟢 站穩" if prices[t].iloc[-1] > ema else "🔴 跌破",
            "相對強度(RS)": rs_trend, "現價": f"${prices[t].iloc[-1]:.2f}"
        })
    st.table(pd.DataFrame(audit_list))

    # 圖表修正 (使用 2026 年新語法 width="stretch")
    st.subheader("📊 趨勢分析")
    target = st.selectbox("選擇查看標的", [t for t in user_tickers if t in prices.columns])
    fig = px.line(prices[target] / prices['QQQ'], title=f"{target} 相對 QQQ 強度")
    st.plotly_chart(fig, width="stretch") # 修正過期語法

except Exception as e:
    st.error(f"系統檢查中：{e}")