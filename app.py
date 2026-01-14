import streamlit as st
import yfinance as yf
import pandas as pd
from fredapi import Fred
from datetime import datetime

# 網頁標題與設定
st.set_page_config(page_title="biibo v6.0 投資駕駛艙", layout="wide")
st.title("📊 biibo v6.0 投資框架：自動化換檔系統")

# ==========================================
# 1. 側邊欄設定 (互動輸入)
# ==========================================
st.sidebar.header("🔧 參數設定")
fred_key = st.sidebar.text_input("輸入 FRED API Key", value="9382c202c6133484efb2c1cb571495af")
current_pe = st.sidebar.number_input("當前 NDX Forward P/E", value=25.3, step=0.1)

# biibo 框架的狀態閥值
pe_mean = 22.0
pe_std = 2.0

# ==========================================
# 2. 數據抓取
# ==========================================
@st.cache_data(ttl=3600) # 快取數據一小時，避免過度請求
def get_data(api_key):
    fred = Fred(api_key=api_key)
    # 抓取市場數據
    prices = yf.download(['QQQ', '^VIX', 'SOXX'], period='2y', interval='1d')['Adj Close']
    # 抓取利差與利率
    spread = fred.get_series('BAMLH0A0HYM2').iloc[-1]
    fed_rate = fred.get_series('FEDFUNDS')
    return prices, spread, fed_rate

try:
    prices, spread, fed_rate = get_data(fred_key)
    
    # 指標計算
    qqq = prices['QQQ']
    current_price = qqq.iloc[-1]
    ma250 = qqq.rolling(window=250).mean().iloc[-1]
    ema10 = qqq.ewm(span=10, adjust=False).mean().iloc[-1]
    ema20 = qqq.ewm(span=20, adjust=False).mean().iloc[-1]
    vix = prices['^VIX'].iloc[-1]

    # ==========================================
    # 3. 邏輯判定 (biibo 核心)
    # ==========================================
    # Layer 0: 估值
    if current_pe < (pe_mean - 0.5 * pe_std):
        v_status, v_color = "① 偏低 (低估)", "green"
    elif current_pe > (pe_mean + 0.5 * pe_std):
        v_status, v_color = "③ 偏高 (昂貴)", "red"
    else:
        v_status, v_color = "② 合理 (標準)", "blue"

    # 換檔邏輯
    if current_price < ma250:
        recommend = "🔴 強制清倉/QQQ (跌破50週均線)"
    elif vix > 22:
        recommend = "🟡 QQQ (1檔) - 風暴預警"
    elif vix < 18 and current_price > ema10:
        recommend = "🟢 QLD (2檔) - 天氣 A+" if v_status != "③ 偏高 (昂貴)" else "🟡 QQQ (1檔) - 估值過高禁止升檔"
    else:
        recommend = "🟢 QQQ (1檔) - 穩健行駛"

    # ==========================================
    # 4. 網頁 UI 呈現
    # ==========================================
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("QQQ 現價", f"${current_price:.2f}")
    col2.metric("VIX 指數", f"{vix:.2f}", delta_color="inverse")
    col3.metric("信用利差", f"{spread:.2f}%")
    col4.metric("50週均線", f"${ma250:.2f}")

    st.markdown("---")
    
    st.subheader(f"🎯 建議行動指令： {recommend}")
    
    # 詳細審計報告
    with st.expander("查看 biibo v6.0 詳細審計清單"):
        st.write(f"**第 0 層 (估值)：** :{v_color}[{v_status}]")
        st.write(f"**第 2 層 (趨勢)：** {'🟢 在50週均線之上' if current_price > ma250 else '🔴 跌破50週均線'}")
        st.write(f"**第 5 層 (路況)：** {'🟢 站穩20日線' if current_price > ema20 else '🟡 跌破20日線'}")
        st.write(f"**Alpha 3.0 預警：** {'⚠️ 利差過高' if spread > 1.2 else '✅ 信用環境穩定'}")

    # 歷史走勢圖
    st.line_chart(prices[['QQQ', 'SOXX']])

except Exception as e:
    st.error(f"數據抓取失敗，請檢查 API Key 是否正確。錯誤訊息: {e}")