import streamlit as st
import yfinance as yf
import pandas as pd
from fredapi import Fred
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. 設定頁面與 API ---
st.set_page_config(page_title="Posa x biibo Alpha 2.0 Risk Dashboard", layout="wide")
st.title("📈 Posa x biibo Alpha 2.0 投資風險審計儀表板")

# 這裡請填入你的 FRED API Key
FRED_API_KEY = "你的 FRED API Key"
fred = Fred(api_key=FRED_API_KEY)

# --- 2. 側邊欄：手動輸入區 ---
st.sidebar.header("⚙️ 參數設定")
user_tickers = st.sidebar.text_input("輸入要審計的個股代號 (以空白分隔)", "AMD CLS URA VRTX").upper().split()
st.sidebar.info("biibo 提醒：天氣不好(VIX>22)時，請執行降檔指令。")

# --- 3. 數據抓取：宏觀流動性 (FRED) ---
@st.cache_data(ttl=3600)
def get_net_liquidity():
    # 抓取 Fed 資產負債表, TGA 帳戶, 逆回購
    walcl = fred.get_series('WALCL', limit=1).iloc[-1]
    tga = fred.get_series('WTREGEN', limit=1).iloc[-1]
    rrp = fred.get_series('RRPONTSYD', limit=1).iloc[-1]
    net_liq = (walcl - tga - rrp) / 1000  # 單位換算成 B (十億)
    return net_liq

# --- 4. 數據抓取：市場天氣 (Yahoo Finance) ---
@st.cache_data(ttl=300)
def get_market_data(tickers):
    all_tickers = list(set(['QQQ', '^VIX', '^MOVE'] + tickers))
    data = yf.download(all_tickers, period="2y", interval="1d")
    return data['Close'], data['Volume']

try:
    net_liq = get_net_liquidity()
    prices, volumes = get_market_data(user_tickers)

    # --- 5. 儀表板第一層：宏觀天氣狀況 ---
    vix = prices['^VIX'].iloc[-1]
    move = prices['^MOVE'].iloc[-1]
    qqq_20ema = prices['QQQ'].ewm(span=20, adjust=False).mean().iloc[-1]
    qqq_current = prices['QQQ'].iloc[-1]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("VIX (股市恐慌)", f"{vix:.2f}", delta_color="inverse")
        status = "🟢 A+ 級" if vix < 18 else ("🟡 B 級" if vix < 22 else "🔴 暴風雨")
        st.write(f"天氣判定：{status}")
    
    with col2:
        st.metric("MOVE (債券預警)", f"{move:.2f}")
        st.write("biibo 指標：>100 代表大資產在逃離")

    with col3:
        diff = ((qqq_current - qqq_20ema) / qqq_20ema) * 100
        st.metric("QQQ 偏離度 (20 EMA)", f"{diff:.2f}%")
        st.write("路況判定：" + ("🟢 平整" if diff > 0 else "🔴 坑洞/回檔"))

    with col4:
        st.metric("美元淨流動性", f"${net_liq:.2f}B")
        st.write("地基判定：流動性為市場真理")

    st.divider()

    # --- 6. 儀表板第二層：個股審計分析 ---
    st.subheader("🔍 即時個股安全範圍審計")
    stock_cols = st.columns(len(user_tickers))

    for i, ticker in enumerate(user_tickers):
        with stock_cols[i]:
            # 計算相對強度 RS (Stock/QQQ)
            rs_series = prices[ticker] / prices['QQQ']
            rs_trend = "↗️" if rs_series.iloc[-1] > rs_series.iloc[-20] else "↘️"
            
            # 計算流動量審計 (RVOL)
            avg_vol = volumes[ticker].rolling(20).mean().iloc[-1]
            curr_vol = volumes[ticker].iloc[-1]
            rvol = curr_vol / avg_vol
            
            # 安全性總分 (1-10)
            score = 0
            if prices[ticker].iloc[-1] > prices[ticker].ewm(span=20).mean().iloc[-1]: score += 3
            if rs_trend == "↗️": score += 3
            if rvol < 1.5: score += 2 # 避免放量重摔
            if vix < 18: score += 2
            
            color = "green" if score >= 7 else ("orange" if score >= 5 else "red")
            st.markdown(f"### :{color}[{ticker}]")
            st.write(f"**安全評分：{score}/10**")
            st.write(f"相對強度：{rs_trend}")
            st.write(f"成交量比：{rvol:.2f}x")
            
    # --- 7. 相關性矩陣 (biibo 隱藏審計) ---
    st.divider()
    st.subheader("🤝 持倉相關性審計 (避免假分散)")
    corr = prices[user_tickers].corr()
    fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title="各標的相關性 (越紅代表風險越集中)")
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"數據抓取失敗，請檢查 API Key 或 代號是否正確。錯誤訊息: {e}")