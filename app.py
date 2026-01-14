import streamlit as st
import yfinance as yf
import pandas as pd
from fredapi import Fred
import plotly.express as px
from datetime import datetime, timedelta

# --- 1. 初始化與安全設定 ---
st.set_page_config(page_title="Posa x biibo Alpha 2.0", layout="wide")
st.title("📈 Posa x biibo 投資風險審計儀表板")

# 從 Streamlit Secrets 讀取 API Key
try:
    FRED_API_KEY = st.secrets["FRED_API_KEY"]
    fred = Fred(api_key=FRED_API_KEY)
except Exception:
    st.error("❌ 找不到 FRED_API_KEY！請在 Streamlit 控制台的 Secrets 設定中添加。")
    st.stop()

# --- 2. 側邊欄輸入 ---
st.sidebar.header("⚙️ 參數設定")
ticker_input = st.sidebar.text_input("輸入個股代號 (空白分隔)", "AMD CLS URA VRTX QQQ").upper()
user_tickers = list(set(ticker_input.split()))

# --- 3. 數據抓取函數 ---
@st.cache_data(ttl=3600)
def get_data(tickers):
    # 抓取宏觀流動性 (FRED)
    # WALCL (資產), WTREGEN (TGA), RRPONTSYD (逆回購)
    walcl = fred.get_series('WALCL').iloc[-1]
    tga = fred.get_series('WTREGEN').iloc[-1]
    rrp = fred.get_series('RRPONTSYD').iloc[-1]
    net_liq = (walcl - tga - rrp) / 1000 # B (十億)

    # 抓取市場數據 (Yahoo Finance)
    all_symbols = tickers + ['^VIX', '^MOVE', 'QQQ']
    df = yf.download(all_symbols, period="1y", interval="1d")
    
    # 處理 yfinance 可能產生的 MultiIndex 欄位
    if isinstance(df.columns, pd.MultiIndex):
        close_prices = df['Close']
        volumes = df['Volume']
    else:
        close_prices = df[['Close']] # 這裡需要更細緻處理，通常多代號必為 MultiIndex
        
    return net_liq, close_prices, volumes

# --- 4. 執行與顯示 ---
try:
    with st.spinner('正在同步數據中...'):
        net_liq, prices, volumes = get_data(user_tickers)

    # A. 頂部核心指標
    vix = prices['^VIX'].iloc[-1] if '^VIX' in prices.columns else 0
    move = prices['^MOVE'].iloc[-1] if '^MOVE' in prices.columns else 0
    qqq_now = prices['QQQ'].iloc[-1]
    qqq_20ema = prices['QQQ'].ewm(span=20).mean().iloc[-1]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("淨流動性 (USD)", f"${net_liq:.2f}B")
    m2.metric("VIX 恐慌指數", f"{vix:.2f}", delta="危險" if vix > 22 else "安全", delta_color="inverse")
    m3.metric("MOVE 債券預警", f"{move:.2f}" if move > 0 else "N/A")
    m4.metric("QQQ 狀態", f"{qqq_now:.1f}", delta=f"{((qqq_now/qqq_20ema)-1)*100:.2f}% (vs EMA20)")

    st.divider()

    # B. 個股審計表
    st.subheader("🔍 個股安全性審計清單")
    audit_data = []
    for t in user_tickers:
        if t in ['^VIX', '^MOVE', 'QQQ']: continue
        
        # 1. 計算相對強度 (RS)
        rs = (prices[t] / prices['QQQ'])
        rs_trend = "↗️ 強勢" if rs.iloc[-1] > rs.rolling(20).mean().iloc[-1] else "↘️ 弱勢"
        
        # 2. 均線位置
        ema20 = prices[t].ewm(span=20).mean().iloc[-1]
        price_status = "🟢 站穩" if prices[t].iloc[-1] > ema20 else "🔴 跌破"
        
        # 3. 成交量審計 (RVOL)
        rvol = volumes[t].iloc[-1] / volumes[t].rolling(20).mean().iloc[-1]
        vol_status = "⚠️ 放量" if rvol > 1.5 else "✅ 正常"
        
        # 4. 綜合評分
        score = 0
        if prices[t].iloc[-1] > ema20: score += 4
        if "↗️" in rs_trend: score += 3
        if vix < 18: score += 3
        
        audit_data.append({
            "代號": t,
            "安全性評分": f"{score}/10",
            "均線狀態": price_status,
            "相對大盤 (RS)": rs_trend,
            "當前量能比": f"{rvol:.2f}x ({vol_status})",
            "現價": f"${prices[t].iloc[-1]:.2f}"
        })
    
    st.table(pd.DataFrame(audit_data))

    # C. 圖表可視化
    st.subheader("📈 價格與相對強度走勢")
    selected_stock = st.selectbox("選擇要查看圖表的個股", [t for t in user_tickers if t != 'QQQ'])
    fig = px.line(prices[selected_stock] / prices['QQQ'], title=f"{selected_stock} 相對強度曲線 (向上代表贏過大盤)")
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"發生預期外錯誤: {e}")
    st.info("提示：請確認代號是否正確，且 FRED API Key 是否有效。")