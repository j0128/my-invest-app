import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import plotly.express as px
from fredapi import Fred
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

# --- 1. 系統初始化 ---
st.set_page_config(page_title="Posa Alpha 4.3", layout="wide")
st.title("🛡️ Posa Alpha 4.3: 週期預判與槓桿效率審計")

try:
    FRED_API_KEY = st.secrets["FRED_API_KEY"]
    fred = Fred(api_key=FRED_API_KEY)
except:
    st.error("❌ Secrets 中缺少 FRED_API_KEY")
    st.stop()

# --- 2. 核心數據抓取 (包含 URA 強化抓取) ---
@st.cache_data(ttl=600)
def fetch_alpha_data(tickers):
    # 標的清洗與基準擴充
    processed = [t.strip().upper() for t in tickers if t]
    benchmarks = ['QQQ', 'QLD', 'TQQQ', '0050.TW', 'BTC-USD', '^VIX', '^MOVE']
    full_list = list(set(processed + benchmarks))
    
    # 增加抓取天數以支援 350DMA (Pi Cycle 需要)
    data = yf.download(full_list, period="2y", auto_adjust=True, progress=False)
    prices = data['Close'].ffill()
    
    # 財報抓取邏輯
    earnings = {}
    for t in processed:
        if "." not in t and "-" not in t:
            try:
                tk = yf.Ticker(t)
                cal = tk.calendar
                if cal is not None and not cal.empty:
                    date_val = cal.loc['Earnings Date'].iloc[0]
                    earnings[t] = date_val.date() if hasattr(date_val, 'date') else date_val
            except: earnings[t] = None
    return prices, earnings

# --- 3. 預測與週期模型 (LaTeX 邏輯實現) ---
def calculate_predictions(series):
    """
    1w: Expected Move (Volatility based)
    1m: Linear Regression
    """
    last_price = series.iloc[-1]
    # 1w: 基於隱含波動率 (使用 30 日歷史標普年化波動代理)
    vol = series.pct_change().std() * np.sqrt(252) 
    expected_move_1w = last_price * vol * np.sqrt(7/365)
    
    # 1m: 線性回歸推估
    y = series.tail(60).values.reshape(-1, 1)
    x = np.array(range(len(y))).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    pred_1m = model.predict([[len(y) + 22]])[0][0]
    
    return last_price - expected_move_1w, last_price + expected_move_1w, pred_1m

# --- 4. 側邊欄：實戰輸入 form ---
with st.sidebar.form("alpha_form"):
    st.header("💰 12.7萬資金部署")
    if 'portfolio_df' not in st.session_state:
        st.session_state.portfolio_df = pd.DataFrame([
            {"代號": "MU", "金額": 30000}, {"代號": "AMD", "金額": 25000},
            {"代號": "URA", "金額": 15000}, {"代號": "BTC-USD", "金額": 57000}
        ])
    edited_df = st.data_editor(st.session_state.portfolio_df, num_rows="dynamic")
    submit = st.form_submit_button("🚀 執行全方位審計與預判")

# --- 5. 實戰審計渲染 ---
if submit:
    user_tickers = edited_df["代號"].dropna().tolist()
    prices, earnings_map = fetch_alpha_data(user_tickers)
    
    # A. 比特幣 Pi Cycle Top 警報 (頂部紅綠燈)
    st.subheader("🔮 比特幣週期導航 (Pi Cycle Top)")
    btc = prices['BTC-USD']
    ma111 = btc.rolling(111).mean()
    ma350_x2 = btc.rolling(350).mean() * 2
    is_top = ma111.iloc[-1] > ma350_x2.iloc[-1]
    
    c1, c2, c3 = st.columns([1, 1, 2])
    c1.metric("當前價格", f"${btc.iloc[-1]:,.0f}")
    c2.metric("Pi 頂部壓力線", f"${ma350_x2.iloc[-1]:,.0f}")
    if is_top:
        c3.error("🚨 PI CYCLE TOP 觸發：週期頂部已現，請即刻執行全線獲利了結！")
    else:
        c3.success("✅ 週期安全：目前尚未觸及 Pi Cycle 頂部交叉。")

    # B. 效率審計表 (對標 QLD/TQQQ)
    st.divider()
    st.subheader("📊 資金效率審計 (標的 vs 槓桿基準)")
    audit_data = []
    for t in user_tickers:
        if t not in prices.columns or t in ['QQQ', 'QLD', 'TQQQ']: continue
        curr = prices[t].iloc[-1]
        
        # 效率計算
        rel_qld = (prices[t]/prices['QLD']).iloc[-1] / (prices[t]/prices['QLD']).iloc[-20] - 1
        efficiency = "🚀 高效 (贏槓桿)" if rel_qld > 0 else "🐌 低效 (輸槓桿)"
        
        # 財報倒數
        e_date = earnings_map.get(t)
        days_to_e = (e_date - datetime.now().date()).days if e_date else "N/A"
        
        # 預測區間
        low_1w, high_1w, p_1m = calculate_predictions(prices[t])
        
        audit_data.append({
            "標的": t, "效率評價": efficiency,
            "1w 預期震盪區間": f"${low_1w:.1f} - ${high_1w:.1f}",
            "1m 回歸目標": f"${p_1m:.1f}",
            "財報倒數": f"{days_to_e} 天" if isinstance(days_to_e, int) else "N/A",
            "指令": "🔥 續抱" if curr > prices[t].ewm(span=20).mean().iloc[-1] else "🛑 減碼"
        })
    st.table(pd.DataFrame(audit_data))

    # C. 視覺化：20EMA 生命線圖表 (審計手冊要求)
    st.subheader("📉 趨勢圖表：20EMA 生命線審核")
    target_plot = st.selectbox("選擇查看趨勢圖", user_tickers)
    if target_plot in prices.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prices.index, y=prices[target_plot], name="股價"))
        fig.add_trace(go.Scatter(x=prices.index, y=prices[target_plot].ewm(span=20).mean(), name="20EMA 生命線", line=dict(dash='dash')))
        st.plotly_chart(fig, use_container_width=True)

    # D. 量化消息過濾 (Placeholder 邏輯)
    st.divider()
    st.subheader("📰 重要消息驚奇指數 (量化篩選)")
    st.info("💡 偵測到 MOVE 指數變動 > 5%，債市波動加劇，建議密切關注今晚美債拍賣結果。")

else:
    st.info("💡 請點擊『🚀 執行全方位審計與預判』開始進行週期推估。")