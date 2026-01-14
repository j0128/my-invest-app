import streamlit as st
import yfinance as yf
import pandas as pd
from fredapi import Fred
import plotly.express as px
import plotly.graph_objects as go
import time
import random
from datetime import datetime, timedelta

# --- 1. 初始化與核心清單 ---
st.set_page_config(page_title="Posa Alpha 2.4 (Audit Edition)", layout="wide")
st.title("🛡️ Alpha 2.4 專業投資審計操作台")

# Seeking Alpha 2026 十大金股
TOP_10_2026 = ['MU', 'AMD', 'CLS', 'COHR', 'CIEN', 'WLDN', 'ATI', 'GOLD', 'ALL', 'INCY']

# 讀取 Secrets
try:
    FRED_API_KEY = st.secrets["FRED_API_KEY"]
    fred = Fred(api_key=FRED_API_KEY)
except Exception:
    st.error("❌ 找不到 FRED_API_KEY！請檢查 Streamlit Secrets 設定。")
    st.stop()

# --- 2. 側邊欄參數設定 ---
st.sidebar.header("⚙️ 審計參數設定")
custom_input = st.sidebar.text_input("輸入自定義持倉 (空白分隔)", "VRTX QQQ").upper()
TRAILING_PCT = st.sidebar.slider("移動止損趴數 (%)", 5, 15, 7) / 100
user_tickers = list(set(custom_input.split() + TOP_10_2026))

# --- 3. 數據抓取與計算 ---
@st.cache_data(ttl=3600)
def fetch_and_audit(tickers):
    # A. 宏觀流動性 (FRED)
    try:
        net_liq = (fred.get_series('WALCL').iloc[-1] - fred.get_series('WTREGEN').iloc[-1] - fred.get_series('RRPONTSYD').iloc[-1]) / 1000
    except: net_liq = 0

    # B. 抓取標的數據
    prices, volumes = pd.DataFrame(), pd.DataFrame()
    earnings_info = {}
    
    all_symbols = list(set(tickers + ['QQQ', '^VIX', '^MOVE']))
    for t in all_symbols:
        time.sleep(random.uniform(0.1, 0.5))
        try:
            ticker_obj = yf.Ticker(t)
            df = ticker_obj.history(period="1y")
            if not df.empty:
                prices[t] = df['Close']
                volumes[t] = df['Volume']
                # 抓取財報日 (功能 1)
                cal = ticker_obj.calendar
                if cal is not None and 'Earnings Date' in cal:
                    earnings_info[t] = cal['Earnings Date'][0].strftime('%Y-%m-%d')
        except: continue
            
    return net_liq, prices, volumes, earnings_info

# --- 4. 執行邏輯 ---
try:
    with st.spinner('正在進行 2026 年度數據與風險審計...'):
        net_liq, prices, volumes, earnings_dates = fetch_and_audit(user_tickers)

    # A. 頂部狀態看板
    vix = prices['^VIX'].iloc[-1] if '^VIX' in prices.columns else 20
    move = prices['^MOVE'].iloc[-1] if '^MOVE' in prices.columns else 0
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("美元淨流動性", f"${net_liq:.2f}B")
    c2.metric("VIX 恐慌指數", f"{vix:.2f}", delta="警報" if vix > 22 else "安全", delta_color="inverse")
    c3.metric("MOVE 指數", f"{move:.2f}")
    c4.metric("QQQ 20EMA 偏離", f"{((prices['QQQ'].iloc[-1]/prices['QQQ'].ewm(span=20).mean().iloc[-1])-1)*100:.2f}%")

    # B. biibo 換檔決策建議
    st.divider()
    if vix < 18:
        st.success("🔥 進攻模式：建議 80% 十大金股 + 20% QQQ。目前路況極佳，全速前進。")
    elif vix < 22:
        st.warning("🛡️ 平衡模式：建議 30% 個股 + 70% QQQ。適度減碼，保護獲利。")
    else:
        st.error("🛑 避險模式：建議 100% 現金。暴風雨來襲，執行會計師強制限價。")

    # C. 功能整合：安全性審計清單 (含移動止損 & 財報預警)
    st.subheader("🔍 實戰審計與避雷清單")
    audit_data = []
    today = datetime.now().date()
    
    for t in user_tickers:
        if t not in prices.columns or t in ['QQQ', '^VIX', '^MOVE']: continue
        
        curr_p = prices[t].iloc[-1]
        peak_p = prices[t].max() # 一年最高價
        stop_p = peak_p * (1 - TRAILING_PCT) # 移動止損價
        
        # 財報警示 (功能 1)
        e_date = earnings_dates.get(t, "未知")
        e_alert = "⚠️ 7天內" if e_date != "未知" and (datetime.strptime(e_date, '%Y-%m-%d').date() - today).days <= 7 else "✅ 安全"
        
        # 評分邏輯
        ema20 = prices[t].ewm(span=20).mean().iloc[-1]
        rs = (prices[t] / prices['QQQ']).iloc[-1] > (prices[t] / prices['QQQ']).rolling(20).mean().iloc[-1]
        score = 0
        if curr_p > ema20: score += 4
        if rs: score += 3
        if vix < 18: score += 3
        
        audit_data.append({
            "標的": t, "安全得分": f"{score}/10",
            "財報