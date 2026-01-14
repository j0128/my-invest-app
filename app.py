import streamlit as st
import yfinance as yf
import pandas as pd
from fredapi import Fred
import plotly.express as px
import plotly.graph_objects as go
import time
import random
from datetime import datetime

# --- 1. 初始化與核心標的 ---
st.set_page_config(page_title="Posa Alpha 3.4", layout="wide")
st.title("🛡️ Posa Alpha 3.4: 雙指標對比與即時趨勢決策中心")

SA_TOP_10 = ['MU', 'AMD', 'CLS', 'CIEN', 'COHR', 'ALL', 'INCY', 'GOLD', 'WLDN', 'ATI']
BENCHMARKS = ['QQQ', '0050.TW', '^VIX', 'BTC-USD']

try:
    FRED_API_KEY = st.secrets["FRED_API_KEY"]
    fred = Fred(api_key=FRED_API_KEY)
except:
    st.error("❌ 請設定 FRED_API_KEY")
    st.stop()

# --- 2. 側邊欄：持倉設定 ---
st.sidebar.header("💰 實戰資產配置")
if 'portfolio_df' not in st.session_state:
    st.session_state.portfolio_df = pd.DataFrame([
        {"代號": "MU", "金額": 30000},
        {"代號": "AMD", "金額": 25000},
        {"代號": "0050.TW", "金額": 50000},
        {"代號": "SOL-USD", "金額": 15000}
    ])
edited_df = st.sidebar.data_editor(st.session_state.portfolio_df, num_rows="dynamic")
user_tickers = edited_df["代號"].tolist()

# --- 3. 數據抓取 (含即時價格) ---
@st.cache_data(ttl=300) # 每 5 分鐘更新一次
def fetch_realtime_data(tickers):
    prices = pd.DataFrame()
    info_box = {}
    full_list = list(set(tickers + SA_TOP_10 + BENCHMARKS))
    
    for t in full_list:
        try:
            time.sleep(0.2)
            tk = yf.Ticker(t)
            # 抓取歷史與最新價格
            df = tk.history(period="1y")
            if not df.empty:
                prices[t] = df['Close']
                # 計算即時漲跌 (最後兩筆)
                change = (df['Close'].iloc[-1] / df['Close'].iloc[-2] - 1) * 100
                info_box[t] = {"price": df['Close'].iloc[-1], "change": change}
        except: continue
    
    try:
        liq = (fred.get_series('WALCL').iloc[-1] - fred.get_series('WTREGEN').iloc[-1] - fred.get_series('RRPONTSYD').iloc[-1]) / 1000
    except: liq = 0
    return liq, prices, info_box

# --- 4. 渲染頁面 ---
try:
    net_liq, prices, info_box = fetch_realtime_data(user_tickers)
    
    # A. 即時行情走馬燈 (解決單薄感)
    st.subheader("⚡ 即時市場脈搏")
    cols = st.columns(len(user_tickers))
    for i, t in enumerate(user_tickers):
        if t in info_box:
            cols[i].metric(t, f"${info_box[t]['price']:.2f}", f"{info_box[t]['change']:.2f}%")

    st.divider()

    # B. 雙指標相對強度分析 (判斷未來走勢)
    st.subheader("🎯 相對強度雷達：誰才是真正的領跑者？")
    target = st.selectbox("選擇分析標的", [t for t in user_tickers if t not in BENCHMARKS])
    
    if target in prices.columns:
        c1, c2 = st.columns(2)
        with c1:
            # 標的 vs QQQ
            rs_qqq = prices[target] / prices['QQQ']
            fig_qqq = px.line(rs_qqq, title=f"{target} / QQQ (向上=贏過美股大盤)")
            st.plotly_chart(fig_qqq, use_container_width=True)
        with c2:
            # 標的 vs 0050
            rs_0050 = prices[target] / prices['0050.TW']
            fig_0050 = px.line(rs_0050, title=f"{target} / 0050 (向上=贏過台股地基)")
            st.plotly_chart(fig_0050, use_container_width=True)

    # C. 趨勢預判表格
    st.subheader("🔍 趨勢健康度審計")
    audit_results = []
    for t in user_tickers:
        if t not in prices.columns or t in ['^VIX']: continue
        curr_p = prices[t].iloc[-1]
        ema20 = prices[t].ewm(span=20).mean().iloc[-1]
        
        # 預判指標：RS 斜率 (過去 5 天)
        rs_trend = (prices[t]/prices['QQQ']).iloc[-5:].pct_change().sum()
        status = "🔥 加速" if (curr_p > ema20 and rs_trend > 0) else "⚠️ 弱化" if (curr_p < ema20) else "🛡️ 盤整"
        
        audit_results.append({
            "標的": t, "目前價格": f"${curr_p:.2f}",
            "20EMA 狀態": "🟢 站穩" if curr_p > ema20 else "🔴 跌破",
            "相對 QQQ 趨勢": "↗️ 增強" if rs_trend > 0 else "↘️ 轉弱",
            "未來走勢預判": status
        })
    st.table(pd.DataFrame(audit_results))

    # D. 智慧會計師報告 (最底端)
    st.divider()
    st.subheader("🖋️ Alpha 3.4 決策修正建議")
    with st.container(border=True):
        vix = prices['^VIX'].iloc[-1]
        if vix > 18: st.warning(f"⚠️ VIX ({vix:.2f}) 突破警戒線，即便今晚大漲，也應視為反彈減碼點。")
        
        for t in user_tickers:
            if t in prices.columns and prices[t].iloc[-1] < prices[t].ewm(span=20).mean().iloc[-1]:
                st.write(f"🛑 **指令：** {t} 趨勢已跌破 20EMA，且相對於 QQQ 轉弱。建議將資金移往更強勢的標的或 0050。")
        
        if info_box.get('BTC-USD', {}).get('change', 0) > 2:
            st.info("💡 **觀察：** 幣圈動能強於美股，符合您觀察到的『資金溢出』，可適度維持幣圈權重。")

except Exception as e:
    st.error(f"系統運行中: {e}")