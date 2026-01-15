import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from fredapi import Fred
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

# --- 數據緩衝與修復模組 ---
@st.cache_data(ttl=600)
def fetch_comprehensive_data(tickers):
    # 確保 URA 等標的不會掉隊，強制補齊 2 年歷史
    processed = [t.strip().upper() for t in tickers if t]
    benchmarks = ['QQQ', 'QLD', 'TQQQ', '0050.TW', 'BTC-USD', '^VIX', '^MOVE']
    full_list = list(set(processed + benchmarks))
    
    data = yf.download(full_list, period="2y", auto_adjust=True, progress=False)
    prices = data['Close'].ffill().bfill() # 雙向填充修復 $nan
    
    # 財報日期抓取與 999 修正
    earnings = {}
    for t in processed:
        if "." not in t and "-" not in t:
            try:
                tk = yf.Ticker(t)
                cal = tk.calendar
                if cal is not None and not cal.empty:
                    d = cal.loc['Earnings Date'].iloc[0]
                    earnings[t] = d.date() if hasattr(d, 'date') else d
            except: earnings[t] = None
    return prices, earnings

def calculate_audit_metrics(series, qld_prices):
    """計算動態凱利與預測值"""
    curr = series.iloc[-1]
    # 動態凱利：過去 120 天勝率
    rets = series.pct_change().shift(-5) # 5日持倉期望
    sig = series > series.ewm(span=20).mean()
    v_rets = rets[sig].dropna()
    
    if len(v_rets) > 10:
        win_p = (v_rets > 0).mean()
        r_ratio = v_rets[v_rets > 0].mean() / abs(v_rets[v_rets < 0].mean())
        k = max(0, (win_p - (1 - win_p) / r_ratio) * 0.5)
    else:
        win_p, k = 0.5, 0.0

    # 1w IV 區間
    vol = series.pct_change().std() * np.sqrt(252)
    move = curr * vol * np.sqrt(7/365)
    
    # 1m 回歸
    y = series.tail(60).values.reshape(-1, 1)
    x = np.array(range(len(y))).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    p_1m = model.predict([[len(y) + 22]])[0][0]
    
    return k, (curr-move, curr+move), p_1m, series.tail(252).max()*0.93

# --- UI 渲染與邏輯 ---
with st.sidebar.form("alpha_v4"):
    st.header("💰 12.7萬實戰資產輸入")
    default_df = pd.DataFrame([
        {"代號": "MU", "金額": 30000}, {"代號": "AMD", "金額": 25000},
        {"代號": "URA", "金額": 15000}, {"代號": "0050.TW", "金額": 40000},
        {"代號": "BTC-USD", "金額": 57000}
    ])
    edited_df = st.data_editor(default_df, num_rows="dynamic")
    submitted = st.form_submit_button("🚀 啟動全方位審計")

if submitted or 'prices' in st.session_state:
    if submitted:
        # 初次點擊，將數據存入 session_state 防止圖表切換時當機
        user_tickers = edited_df["代號"].dropna().tolist()
        prices, earnings = fetch_comprehensive_data(user_tickers)
        st.session_state.prices = prices
        st.session_state.earnings = earnings
        st.session_state.tickers = user_tickers

    p = st.session_state.prices
    e = st.session_state.earnings
    ts = st.session_state.tickers

    # A. 淨流動性修正 (功能 1)
    try:
        fred = Fred(api_key=st.secrets["FRED_API_KEY"])
        liq = (fred.get_series('WALCL').iloc[-1] - fred.get_series('WTREGEN').iloc[-1] - fred.get_series('RRPONTSYD').iloc[-1]) / 1000
        st.metric("淨流動性 (實質購買力)", f"${liq:,.2f}B", help="總資產 - TGA帳戶 - 逆回購")
    except: st.warning("FRED 數據連結中...")

    # B. 深度審計表 (修正 999 財報問題)
    audit_results = []
    for t in ts:
        if t in p.columns:
            k, (l, h), p1m, tstop = calculate_audit_metrics(p[t], p['QLD'])
            edate = e.get(t)
            days = (edate - datetime.now().date()).days if edate else "無資料"
            
            audit_results.append({
                "標的": t, "凱利權重": f"{k*100:.1f}%", 
                "1w 區間": f"{l:.1f}-{h:.1f}", "1m 目標": f"{p1m:.1f}",
                "移動止損": f"${tstop:.1f}", "財報倒數": f"{days}d" if isinstance(days, int) else days
            })
    st.table(pd.DataFrame(audit_results))

    # C. 20EMA 穩定圖表 (防止切換當機)
    st.subheader("📉 趨勢生命線審計")
    pick = st.selectbox("選擇審查標的", ts, key="plot_select")
    if pick in p.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=p.index, y=p[pick], name="現價"))
        fig.add_trace(go.Scatter(x=p.index, y=p[pick].ewm(span=20).mean(), name="20EMA", line=dict(dash='dash')))
        st.plotly_chart(fig, use_container_width=True)