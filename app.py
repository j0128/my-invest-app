import streamlit as st
import yfinance as yf
import pandas as pd
from fredapi import Fred
import plotly.express as px
import time
import random

# --- 1. 初始化設定 ---
st.set_page_config(page_title="Posa x biibo Alpha 2.3", layout="wide")
st.title("📈 Posa Alpha 2.3 (2026 十大金股監控版)")

# Seeking Alpha 2026 十大金股清單 (核心輻射源)
TOP_10_2026 = ['MU', 'AMD', 'CLS', 'COHR', 'CIEN', 'WLDN', 'ATI', 'GOLD', 'ALL', 'INCY']

# 從 Secrets 讀取 API Key
try:
    FRED_API_KEY = st.secrets["FRED_API_KEY"]
    fred = Fred(api_key=FRED_API_KEY)
except Exception:
    st.error("❌ 找不到 FRED_API_KEY！請檢查 Streamlit Secrets 設定。")
    st.stop()

# --- 2. 側邊欄：自定義持倉 ---
st.sidebar.header("📋 我的自定義持倉")
custom_input = st.sidebar.text_input("輸入你想額外審計的代號 (如: VRTX QQQ)", "VRTX QQQ").upper()
custom_tickers = list(set(custom_input.split()))

# --- 3. 強化數據抓取函數 ---
def fetch_with_retry(ticker):
    time.sleep(random.uniform(0.5, 1.5)) # 降低延遲，提升效率
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False)
        if df.empty: return None, None
        close = df['Close'].iloc[:, 0] if isinstance(df['Close'], pd.DataFrame) else df['Close']
        vol = df['Volume'].iloc[:, 0] if isinstance(df['Volume'], pd.DataFrame) else df['Volume']
        return close, vol
    except:
        return None, None

@st.cache_data(ttl=3600)
def fetch_system_data(user_tickers):
    # A. 抓取宏觀流動性 (FRED)
    try:
        walcl = fred.get_series('WALCL').iloc[-1]
        tga = fred.get_series('WTREGEN').iloc[-1]
        rrp = fred.get_series('RRPONTSYD').iloc[-1]
        net_liq = (walcl - tga - rrp) / 1000 
    except: net_liq = 0

    # B. 抓取所有標的 (Top 10 + Custom + QQQ + VIX + MOVE)
    prices, volumes = pd.DataFrame(), pd.DataFrame()
    all_needed = list(set(TOP_10_2026 + user_tickers + ['QQQ', '^VIX', '^MOVE']))
    
    for t in all_needed:
        p, v = fetch_with_retry(t)
        if p is not None:
            prices[t], volumes[t] = p, v
    return net_liq, prices, volumes

# --- 4. 執行與顯示介面 ---
try:
    with st.spinner('正在同步 2026 十大金股數據...'):
        net_liq, prices, volumes = fetch_system_data(custom_tickers)

    # 第一層：宏觀指標
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("美元淨流動性", f"${net_liq:.2f}B")
    
    vix = prices['^VIX'].dropna().iloc[-1] if '^VIX' in prices.columns else 20
    m2.metric("VIX 指數", f"{vix:.2f}", delta="危險" if vix > 22 else "安全", delta_color="inverse")
    
    move = prices['^MOVE'].dropna().iloc[-1] if '^MOVE' in prices.columns else 0
    m3.metric("MOVE 指數", f"{move:.2f}")

    q_now = prices['QQQ'].dropna().iloc[-1]
    q_ema = prices['QQQ'].ewm(span=20).mean().iloc[-1]
    m4.metric("QQQ 狀態", f"${q_now:.1f}", delta=f"{((q_now/q_ema)-1)*100:.2f}%")

    # 第二層：biibo 決策大腦
    st.divider()
    if vix < 18:
        mode, color, strategy = "🔥 進攻模式", "green", "建議配置：80% 十大金股 + 20% QQQ。利用高 RS 標的擴大獲利。"
    elif vix < 22:
        mode, color, strategy = "🛡️ 平衡模式", "orange", "建議配置：30% 十大金股 + 70% QQQ。適度收縮，回防母艦。"
    else:
        mode, color, strategy = "🛑 避險模式", "red", "建議配置：100% 現金或 TLT。避開崩盤風險。"
    
    st.subheader(f"🎯 當前操作指令：:{color}[{mode}]")
    st.info(f"**戰略指引**：{strategy}")

    # 第三層：數據自動掃描區 (Top 10)
    st.subheader("🚀 2026 十大金股：即時安全審計")
    
    def get_audit_row(t):
        if t not in prices.columns: return None
        ema = prices[t].ewm(span=20).mean().iloc[-1]
        rs = (prices[t] / prices['QQQ'])
        rs_trend = "↗️ 強" if rs.iloc[-1] > rs.rolling(20).mean().iloc[-1] else "↘️ 弱"
        score = 0
        if prices[t].iloc[-1] > ema: score += 4
        if rs_trend == "↗️ 強": score += 3
        if vix < 18: score += 3
        return {"標的": t, "評分": f"{score}/10", "20EMA": "🟢 站上" if prices[t].iloc[-1] > ema else "🔴 跌破", "相對強度": rs_trend, "現價": f"${prices[t].iloc[-1]:.2f}"}

    top_10_audit = [get_audit_row(t) for t in TOP_10_2026 if get_audit_row(t) is not None]
    st.table(pd.DataFrame(top_10_audit))

    # 第四層：自定義持倉審計
    st.subheader("📋 我的持倉審計 (自定義)")
    custom_audit = [get_audit_row(t) for t in custom_tickers if get_audit_row(t) is not None]
    if custom_audit:
        st.table(pd.DataFrame(custom_audit))

    # 第五層：視覺化分析
    st.subheader("📊 相對強度 (RS) 輻射圖")
    target = st.selectbox("選擇要分析的標的", TOP_10_2026 + custom_tickers)
    if target in prices.columns:
        fig = px.line(prices[target] / prices['QQQ'], title=f"{target} vs QQQ (曲線向上代表跑贏大盤)")
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"系統檢查中：{e}")