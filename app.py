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
st.set_page_config(page_title="Posa Alpha 3.5", layout="wide")
st.title("🛡️ Posa Alpha 3.5: 終極穩定與雙指標預判中心")

# 核心清單：美股金股 + 幣圈 + 台股
CORE_LIST = ['MU', 'AMD', 'CLS', 'COHR', 'URA', 'VRTX', '0050.TW', 'BTC-USD', 'SOL-USD', 'ETH-USD', 'TLT']
SA_DATA = {'MU': 'PEG 0.20x (極度折價)', 'CLS': '15次盈餘上修', 'AMD': 'AI 動能領先'}

try:
    FRED_API_KEY = st.secrets["FRED_API_KEY"]
    fred = Fred(api_key=FRED_API_KEY)
except:
    st.error("❌ 請設定 FRED_API_KEY")
    st.stop()

# --- 2. 側邊欄：資產配置 ---
st.sidebar.header("💰 實戰資產配置")
if 'portfolio_df' not in st.session_state:
    st.session_state.portfolio_df = pd.DataFrame([
        {"代號": "MU", "金額": 30000},
        {"代號": "AMD", "金額": 36000},
        {"代號": "0050.TW", "金額": 70000},
        {"代號": "BTC-USD", "金額": 100000}
    ])
edited_df = st.sidebar.data_editor(st.session_state.portfolio_df, num_rows="dynamic")
user_tickers = edited_df["代號"].tolist()
total_val = edited_df["金額"].sum()

# --- 3. 強化數據抓取 (修復 $nan 問題) ---
@st.cache_data(ttl=300)
def fetch_fix_data(tickers):
    prices = pd.DataFrame()
    info = {}
    full_list = list(set(tickers + CORE_LIST + ['QQQ', '^VIX']))
    
    for t in full_list:
        try:
            # 針對不同標的調整抓取策略
            tk = yf.Ticker(t)
            df = tk.history(period="2y") # 抓兩年確保 EMA20 不會變 nan
            if not df.empty:
                prices[t] = df['Close']
                # 即時價格與漲跌
                curr_p = df['Close'].iloc[-1]
                prev_p = df['Close'].iloc[-2]
                change = (curr_p / prev_p - 1) * 100
                info[t] = {"price": curr_p, "change": change}
        except: continue
    
    try:
        liq = (fred.get_series('WALCL').iloc[-1] - fred.get_series('WTREGEN').iloc[-1] - fred.get_series('RRPONTSYD').iloc[-1]) / 1000
    except: liq = 0
    return liq, prices, info

# --- 4. 渲染頁面：解決顯示擠壓 ---
try:
    net_liq, prices, info = fetch_fix_data(user_tickers)
    vix = prices['^VIX'].iloc[-1]

    # A. 即時行情 (改用網格佈局解決位數遮斷)
    st.subheader("⚡ 即時市場脈搏")
    rows = [user_tickers[i:i + 4] for i in range(0, len(user_tickers), 4)]
    for row in rows:
        cols = st.columns(4)
        for i, t in enumerate(row):
            if t in info:
                cols[i].metric(t, f"${info[t]['price']:,.2f}", f"{info[t]['change']:.2f}%")

    # B. 雙指標趨勢審計 (QQQ & 0050)
    st.divider()
    st.subheader("🎯 趨勢健康度與預判")
    audit_data = []
    for t in user_tickers:
        if t not in prices.columns or t in ['^VIX', 'QQQ', '0050.TW']: continue
        
        curr_p = prices[t].iloc[-1]
        ema20 = prices[t].ewm(span=20).mean().iloc[-1]
        
        # 相對強度 (RS)
        rs_qqq = (prices[t]/prices['QQQ']).iloc[-1] > (prices[t]/prices['QQQ']).rolling(20).mean().iloc[-1]
        rs_tw = (prices[t]/prices['0050.TW']).iloc[-1] > (prices[t]/prices['0050.TW']).rolling(20).mean().iloc[-1] if '0050.TW' in prices.columns else False
        
        # 未來走勢預判邏輯
        if curr_p > ema20 and rs_qqq:
            trend = "🔥 加速上升"
        elif curr_p < ema20 and not rs_qqq:
            trend = "🛑 趨勢反轉"
        else:
            trend = "🛡️ 盤整測試"

        audit_data.append({
            "標的": t, "20EMA": "🟢 站穩" if curr_p > ema20 else "🔴 跌破",
            "贏過 QQQ": "✅" if rs_qqq else "❌",
            "贏過 0050": "✅" if rs_tw else "❌",
            "未來預判": trend,
            "止損位": f"${prices[t].max()*0.93:.2f}"
        })
    st.table(pd.DataFrame(audit_data))

    # C. 視覺化對比：RS 曲線
    st.subheader("📊 未來動力分析：相對強度曲線")
    target = st.selectbox("選擇要分析的標的", [t for t in user_tickers if t not in ['QQQ', '0050.TW']])
    if target in prices.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prices.index, y=prices[target]/prices['QQQ'], name="vs QQQ (美股基準)"))
        fig.add_trace(go.Scatter(x=prices.index, y=prices[target]/prices['0050.TW'], name="vs 0050 (台股基準)"))
        st.plotly_chart(fig, use_container_width=True)

    # D. biibo 盲點補強：相關性審計
    st.divider()
    st.subheader("🤝 板塊相關性矩陣 (biibo 盲點：過度集中風險)")
    st.plotly_chart(px.imshow(prices[user_tickers].corr(), text_auto=".2f", color_continuous_scale='RdBu_r'))

    # E. 終極智慧修正意見
    st.subheader("🖋️ Posa 實戰決策報告")
    with st.container(border=True):
        if vix > 18: st.warning(f"⚠️ VIX 目前 {vix:.2f}，市場保險變貴。即便標的上漲，凱利公式也建議保持 20% 現金。")
        for _, row in pd.DataFrame(audit_data).iterrows():
            if row['20EMA'] == "🔴 跌破":
                st.write(f"🛑 **強制指令：** {row['標的']} 跌破生命線。這不是『便宜』，這是『變質』，請執行減碼。")
        if info.get('BTC-USD', {}).get('change', 0) > info.get('QQQ', {}).get('change', 0):
            st.info("💡 **資金流向觀察：** 幣圈動能強於美股 QQQ，確認資金溢出效應，SOL/BTC 權重可維持。")

except Exception as e:
    st.error(f"系統運行中：{e}")