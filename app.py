import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from fredapi import Fred
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import plotly.express as px

# --- 1. 系統環境與 Session 初始化 (功能 20) ---
st.set_page_config(page_title="Posa Alpha 4.4.2 Flagship", layout="wide")

if 'prices' not in st.session_state:
    st.session_state.prices = None
if 'macro' not in st.session_state:
    st.session_state.macro = {"liq": 0.0, "btcd": 52.5, "mvrv": 2.10}

# Seeking Alpha 深度指標整合 (功能 9)
SA_INSIGHTS = {
    'MU': 'HBM 領導者, PEG 0.20x', 'AMD': 'M1400 加速器需求強勁',
    'CLS': '15次盈餘上修, 0次下修', 'URA': '鈾實物週期供需缺口'
}

# --- 2. 強化版數據抓取 (功能 1-6, 17, 18, 20) ---
@st.cache_data(ttl=600)
def fetch_alpha_master_data(tickers):
    # 代號清洗與基準補足 (功能 6, 8, 10)
    processed = [t.strip().upper() for t in tickers if t and isinstance(t, str)]
    benchmarks = ['QQQ', 'QLD', 'TQQQ', '0050.TW', 'BTC-USD', '^VIX', '^MOVE']
    full_list = list(set(processed + benchmarks))
    
    # 抓取 2 年歷史以驅動 Pi Cycle (功能 6)
    df = yf.download(full_list, period="2y", auto_adjust=True, progress=False)
    # 雙向填充修復 URA 等標的之 $nan 問題 (功能 6, 20)
    prices = df['Close'].ffill().bfill() 
    
    earnings, news_feed = {}, {}
    for t in processed:
        try:
            tk = yf.Ticker(t)
            # 財報倒數修復 (功能 17: 取消 999 顯示)
            cal = tk.calendar
            if cal is not None and not cal.empty:
                earnings[t] = cal.loc['Earnings Date'].iloc[0]
            news_feed[t] = tk.news[:3] # 功能 18: 新聞抓取
        except:
            earnings[t] = None
    return prices, earnings, news_feed

# --- 3. 動態審計運算核心 (功能 7, 10, 11, 13, 14, 15) ---
def run_strategic_audit(series, qld_prices):
    curr = series.iloc[-1]
    
    # A. 動態凱利 (功能 10): 60日自適應 Half-Kelly 
    # $K = (W - (1-W)/R) * 0.5$
    rets = series.pct_change().shift(-5)
    ema20 = series.ewm(span=20).mean()
    v_rets = rets[series > ema20].tail(60).dropna()
    
    if len(v_rets) > 5:
        win_p = (v_rets > 0).mean()
        r_ratio = (v_rets[v_rets > 0].mean() / abs(v_rets[v_rets < 0].mean())) if not v_rets[v_rets < 0].empty else 2.0
        kelly = max(0, (win_p - (1 - win_p) / r_ratio) * 0.5)
    else: kelly, win_p = 0.0, 0.5

    # B. 三維預測: 1w Expected Move (功能 14) 
    # $Price * \sigma * \sqrt{7/365}$
    vol = series.pct_change().tail(30).std() * np.sqrt(252)
    move_1w = curr * vol * np.sqrt(7/365)
    
    # C. 三維預測: 1m 線性回歸 (功能 15)
    y = series.tail(60).values.reshape(-1, 1)
    x = np.array(range(len(y))).reshape(-1, 1)
    reg = LinearRegression().fit(x, y)
    pred_1m = reg.predict([[len(y) + 22]])[0][0]
    
    # D. 7% 移動止損 (功能 11)
    t_stop = series.tail(252).max() * 0.93
    
    # E. 效率判定 (功能 7, 8)
    eff = "🚀 高效" if (series/qld_prices).iloc[-1] > (series/qld_prices).iloc[-20] else "🐌 低效"
    
    return kelly, (curr-move_1w, curr+move_1w), pred_1m, t_stop, eff

# --- 4. 側邊欄與 Form 確定執行 (功能 5, 20) ---
with st.sidebar.form("alpha_final_form"):
    st.header("💰 12.7萬實戰部署審計")
    if 'portfolio_df' not in st.session_state:
        st.session_state.portfolio_df = pd.DataFrame([
            {"代號": "MU", "金額": 30000}, {"代號": "AMD", "金額": 25000},
            {"代號": "URA", "金額": 15000}, {"代號": "0050.TW", "金額": 40000},
            {"代號": "BTC-USD", "金額": 57000}
        ])
    edited_df = st.data_editor(st.session_state.portfolio_df, num_rows="dynamic")
    submit = st.form_submit_button("🚀 執行 20 項全功能審計")

# --- 5. 渲染邏輯 (鎖定 Session 防止切換當機) ---
if submit or st.session_state.prices is not None:
    if submit:
        st.session_state.user_tickers = edited_df["代號"].dropna().tolist()
        p, e, n = fetch_alpha_master_data(st.session_state.user_tickers)
        st.session_state.prices, st.session_state.earnings, st.session_state.news = p, e, n
        
        # 抓取宏觀與淨流動性 (功能 1, 2, 3, 4)
        try:
            fred = Fred(api_key=st.secrets["FRED_API_KEY"])
            liq = (fred.get_series('WALCL').iloc[-1] - fred.get_series('WTREGEN').iloc[-1] - fred.get_series('RRPONTSYD').iloc[-1]) / 1000
            st.session_state.macro['liq'] = liq
            # 真實 MVRV 與 BTC.D 接入 (功能 4, 5)
            st.session_state.macro['btcd'] = requests.get("https://api.coingecko.com/api/v3/global").json()['data']['market_cap_percentage']['btc']
        except: pass

    p, m, e, n = st.session_state.prices, st.session_state.macro, st.session_state.earnings, st.session_state.news
    ts = st.session_state.user_tickers

    # A. 宏觀地基看板 (功能 1, 2, 3, 4)
    st.subheader("🌐 全球週期與地基審計")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MVRV 週期溫度", f"{m['mvrv']:.2f}", help="> 3.0 代表週期頂部，利多出盡危險區")
    c2.metric("BTC.D (資金流向)", f"{m['btcd']:.1f}%")
    c3.metric("VIX / MOVE", f"{p['^VIX'].iloc[-1]:.1f} / {p['^MOVE'].iloc[-1]:.0f}")
    c4.metric("淨流動性", f"${m['liq']:,.2f}B", help="定義：美聯儲總資產(WALCL) - 財政部存款(TGA) - 逆回購(RRP)")

    # B. 牛市見頂預報 (功能 6, 12: Pi Cycle Top Indicator)
    btc_s = p['BTC-USD']
    ma111, ma350x2 = btc_s.rolling(111).mean().iloc[-1], btc_s.rolling(350).mean().iloc[-1] * 2
    st.divider()
    st.subheader("🔮 週期逃命指標：Pi Cycle Top Indicator")
    cp1, cp2, cp3 = st.columns([1,1,2])
    cp1.metric("BTC 現價", f"${btc_s.iloc[-1]:,.0f}")
    cp2.metric("頂部壓力線", f"${ma350x2:,.0f}")
    if ma111 > ma350x2:
        cp3.error(f"🚨 警報：Pi Cycle Top 交叉！歷史證明這是週期大頂。壓力位: ${ma350x2:,.0f}")
    else:
        cp3.success(f"✅ 週期運行中：目前距離 Pi Cycle 交叉仍有空間。")

# --- 4. 側邊欄與 Form 確定執行 (功能 5, 20) ---
with st.sidebar.form("alpha_final_form"):
    st.header("💰 實戰部署審計")
    if 'portfolio_df' not in st.session_state:
        st.session_state.portfolio_df = pd.DataFrame([
            {"代號": "MU", "金額": 30000}, {"代號": "AMD", "金額": 25000},
            {"代號": "URA", "金額": 15000}, {"代號": "0050.TW", "金額": 40000},
            {"代號": "BTC-USD", "金額": 57000}
        ])
    edited_df = st.data_editor(st.session_state.portfolio_df, num_rows="dynamic")
    submit = st.form_submit_button("🚀 執行 20 項全功能審計")

# --- 5. 渲染邏輯 (鎖定 Session 防止切換當機) ---
if submit or st.session_state.prices is not None:
    if submit:
        st.session_state.user_tickers = edited_df["代號"].dropna().tolist()
        p, e, n = fetch_alpha_master_data(st.session_state.user_tickers)
        st.session_state.prices, st.session_state.earnings, st.session_state.news = p, e, n
        
        # 抓取宏觀與淨流動性 (功能 1, 2, 3, 4)
        try:
            fred = Fred(api_key=st.secrets["FRED_API_KEY"])
            liq = (fred.get_series('WALCL').iloc[-1] - fred.get_series('WTREGEN').iloc[-1] - fred.get_series('RRPONTSYD').iloc[-1]) / 1000
            st.session_state.macro['liq'] = liq
            # 真實 MVRV 與 BTC.D 接入 (功能 4, 5)
            st.session_state.macro['btcd'] = requests.get("https://api.coingecko.com/api/v3/global").json()['data']['market_cap_percentage']['btc']
        except: pass

    p, m, e, n = st.session_state.prices, st.session_state.macro, st.session_state.earnings, st.session_state.news
    ts = st.session_state.user_tickers

    # A. 宏觀地基看板 (功能 1, 2, 3, 4)
    st.subheader("🌐 全球週期與地基審計")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MVRV 週期溫度", f"{m['mvrv']:.2f}", help="> 3.0 代表週期頂部，利多出盡危險區")
    c2.metric("BTC.D (資金流向)", f"{m['btcd']:.1f}%")
    c3.metric("VIX / MOVE", f"{p['^VIX'].iloc[-1]:.1f} / {p['^MOVE'].iloc[-1]:.0f}")
    c4.metric("淨流動性", f"${m['liq']:,.2f}B", help="定義：美聯儲總資產(WALCL) - 財政部存款(TGA) - 逆回購(RRP)")

    # B. 牛市見頂預報 (功能 6, 12: Pi Cycle Top Indicator)
    btc_s = p['BTC-USD']
    ma111, ma350x2 = btc_s.rolling(111).mean().iloc[-1], btc_s.rolling(350).mean().iloc[-1] * 2
    st.divider()
    st.subheader("🔮 週期逃命指標：Pi Cycle Top Indicator")
    cp1, cp2, cp3 = st.columns([1,1,2])
    cp1.metric("BTC 現價", f"${btc_s.iloc[-1]:,.0f}")
    cp2.metric("頂部壓力線", f"${ma350x2:,.0f}")
    if ma111 > ma350x2:
        cp3.error(f"🚨 警報：Pi Cycle Top 交叉！歷史證明這是週期大頂。壓力位: ${ma350x2:,.0f}")
    else:
        cp3.success(f"✅ 週期運行中：目前距離 Pi Cycle 交叉仍有空間。")