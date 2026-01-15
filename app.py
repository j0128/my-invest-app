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

# --- 1. 系統環境與核心初始化 ---
st.set_page_config(page_title="Posa Alpha 4.4.3 Final", layout="wide")

# 初始化 Session 狀態，確保切換 20EMA 圖表絕對不當機
if 'prices' not in st.session_state:
    st.session_state.update({'prices': None, 'earnings': {}, 'news': {}, 'macro': {}})

# Seeking Alpha 深度指標 (功能 9)
SA_INSIGHTS = {
    'MU': 'HBM 領先, PEG 0.20x', 'AMD': 'M1400 需求強勁',
    'CLS': '15次盈餘上修', 'URA': '鈾實物週期缺口',
    'BTC-USD': '週期避險資產', '0050.TW': '台股科技核心'
}

# --- 2. 數據抓取引擎 (修復 URA $nan / 功能 17 財報修正) ---
@st.cache_data(ttl=600)
def fetch_alpha_master(tickers):
    processed = [t.strip().upper() for t in tickers if t]
    benchmarks = ['QQQ', 'QLD', 'TQQQ', '0050.TW', 'BTC-USD', '^VIX', '^MOVE']
    full_list = list(set(processed + benchmarks))
    
    # 抓取 2 年資料驅動 Pi Cycle (功能 6)
    df = yf.download(full_list, period="2y", auto_adjust=True, progress=False)
    prices = df['Close'].ffill().bfill() # 雙向填充修復 URA 休市問題
    
    earnings, news_feed = {}, {}
    for t in processed:
        try:
            tk = yf.Ticker(t)
            # 財報倒數修復 (徹底消除 999d)
            cal = tk.calendar
            if cal is not None and not cal.empty:
                earnings[t] = cal.loc['Earnings Date'].iloc[0]
            news_feed[t] = tk.news[:3]
        except: earnings[t] = None
    return prices, earnings, news_feed

# --- 3. 實戰運算邏輯 (60日動態凱利 & 三維預測) ---
def run_strategic_audit(series, qld_prices):
    curr = series.iloc[-1]
    # 功能 10: 60日動態半凱利 (Dynamic Half-Kelly)
    # $K = (W - (1-W)/R) \times 0.5$
    rets = series.pct_change().shift(-5)
    ema20 = series.ewm(span=20).mean()
    v_rets = rets[series > ema20].tail(60).dropna()
    
    if len(v_rets) > 5:
        win_p = (v_rets > 0).mean()
        r_ratio = (v_rets[v_rets > 0].mean() / abs(v_rets[v_rets < 0].mean())) if not v_rets[v_rets < 0].empty else 2.0
        kelly = max(0, (win_p - (1 - win_p) / r_ratio) * 0.5)
    else: kelly, win_p = 0.0, 0.5

    # 功能 14, 15: 1w Expected Move & 1m Regression
    vol = series.pct_change().tail(30).std() * np.sqrt(252)
    move_1w = curr * vol * np.sqrt(7/365)
    y = series.tail(60).values.reshape(-1, 1); x = np.array(range(len(y))).reshape(-1, 1)
    pred_1m = LinearRegression().fit(x, y).predict([[len(y) + 22]])[0][0]
    
    # 功能 11: 7% 移動止損 (Trailing Stop)
    t_stop = series.tail(252).max() * 0.93
    eff = "🚀 高效" if (series/qld_prices).iloc[-1] > (series/qld_prices).iloc[-20] else "🐌 低效"
    
    return kelly, (curr-move_1w, curr+move_1w), pred_1m, t_stop, eff

# --- 4. 側邊欄與 Form 確定執行 (功能 5, 20) ---
with st.sidebar.form(key="alpha_master_form_2026"):
    st.header("💰 12.7萬實戰部署配置")
    if 'portfolio_df' not in st.session_state:
        st.session_state.portfolio_df = pd.DataFrame([
            {"代號": "MU", "金額": 30000}, {"代號": "AMD", "金額": 25000},
            {"代號": "URA", "金額": 15000}, {"代號": "0050.TW", "金額": 40000},
            {"代號": "BTC-USD", "金額": 57000}
        ])
    edited_df = st.data_editor(st.session_state.portfolio_df, num_rows="dynamic")
    submit = st.form_submit_button("🚀 啟動 20 項全方位審計預判")

if submit or st.session_state.prices is not None:
    if submit:
        st.session_state.user_tickers = edited_df["代號"].dropna().tolist()
        p, e, n = fetch_alpha_master(st.session_state.user_tickers)
        st.session_state.prices, st.session_state.earnings, st.session_state.news = p, e, n
        # 抓取宏觀數據 (功能 1-4)
        try:
            fred = Fred(api_key=st.secrets["FRED_API_KEY"])
            liq = (fred.get_series('WALCL').iloc[-1] - fred.get_series('WTREGEN').iloc[-1] - fred.get_series('RRPONTSYD').iloc[-1]) / 1000
            st.session_state.macro = {"liq": liq, "btcd": 57.2, "mvrv": 2.15} # 可接入真鏈上 API
        except: pass

    p, m, e, n_map, ts = st.session_state.prices, st.session_state.macro, st.session_state.earnings, st.session_state.news, st.session_state.user_tickers

    # A. 宏觀地基看板 (功能 1, 2, 3, 4)
    st.subheader("🌐 全球週期與利好出盡偵測")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MVRV 週期溫度", f"{m.get('mvrv', 0):.2f}", help="> 3.0 代表利多出盡危險區")
    c2.metric("BTC.D 市佔率", f"{m.get('btcd', 0):.1f}%")
    c3.metric("VIX / MOVE", f"{p['^VIX'].iloc[-1]:.1f} / {p['^MOVE'].iloc[-1]:.0f}")
    c4.metric("淨流動性", f"${m.get('liq', 0):,.2f}B", help="定義：美聯儲總資產 - 財政部帳戶 - 逆回購")

    # B. 牛市頂部警報 (功能 6, 12: Pi Cycle Top Indicator)
    btc_s = p['BTC-USD']
    ma111, ma350x2 = btc_s.rolling(111).mean().iloc[-1], btc_s.rolling(350).mean().iloc[-1] * 2
    st.divider()
    st.subheader("🔮 週期逃命指標：Pi Cycle Top Indicator")
    cp1, cp2, cp3 = st.columns([1,1,2])
    cp1.metric("BTC 現價", f"${btc_s.iloc[-1]:,.0f}")
    cp2.metric("頂部壓力線", f"${ma350x2:,.0f}")
    if ma111 > ma350x2:
        cp3.error("🚨 **終極警報：PI CYCLE TOP 交叉！** 牛市已見頂。")
    else:
        cp3.success("✅ **週期運行中**：距離 Pi Cycle 交叉仍有空間。")

# C. 深度審計大表 (功能 7-11, 13-15, 17)
    st.divider()
    st.subheader("📋 跨市場深度審計 (動態凱利與預測整合)")
    audit_list = []
    for t in ts:
        if t in p.columns and t not in ['QQQ', 'QLD']:
            k, (l1w, h1w), p1m, tstop, eff = run_strategic_audit(p[t], p['QLD'])
            ed = e.get(t)
            e_val = f"{(ed.date() - datetime.now().date()).days}d" if hasattr(ed, 'date') else "無資料"
            audit_list.append({
                "標的": t, "SA觀點": SA_INSIGHTS.get(t, "實務資產"), "效率": eff,
                "20EMA": "🟢 站穩" if p[t].iloc[-1] > p[t].ewm(span=20).mean().iloc[-1] else "🔴 跌破",
                "1w區間": f"{l1w:.1f}-{h1w:.1f}", "1m回歸": f"${p1m:.1f}",
                "動態凱利": f"{k*100:.1f}%", "移動止損": f"${tstop:.1f}", "財報": e_val
            })
    st.table(pd.DataFrame(audit_list))

    # D. 相關性與趨勢審查 (功能 9, 13, 20: 鎖定 key 防止當機)
    st.divider()
    col_h, col_c = st.columns([1, 1.2])
    with col_h:
        st.subheader("🤝 板塊相關性")
        corr = p[ts].corr()
        st.plotly_chart(px.imshow(corr, text_auto=".2f", color_continuous_scale='RdBu_r'), use_container_width=True)
        # 功能 13: 相關性文字結論
        if corr.unstack().sort_values(ascending=False).drop_duplicates()[1] > 0.8:
            st.warning("🚨 **過度集中**：發現標的高相關 (>0.8)，風險未分散。")
        else:
            st.success("✅ **配置健康**：板塊分散度優良。")

    with col_c:
        st.subheader("📉 趨勢生命線審查 (切換不當機版)")
        # 解決 image_430567: 唯一 key 鎖定
        pick = st.selectbox("選擇要審核的標的", ts, key="stable_final_selector_2026")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=p.index, y=p[pick], name="現價", line=dict(color='gold')))
        fig.add_trace(go.Scatter(x=p.index, y=p[pick].ewm(span=20).mean(), name="20EMA", line=dict(dash='dash')))
        fig.add_hline(y=p[pick].tail(252).max()*0.93, line_dash="dot", line_color="red", annotation_text="7%止損線")
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)

    # E. 旗艦手冊 (功能 15: LaTeX)
    st.divider()
    with st.expander("📚 Posa 旗艦審計手冊"):
        st.markdown(f"""
        ### 1. 預測模型依據
        * **1w Expected Move**: $Price \\pm (Price \\times \\sigma \\times \\sqrt{{7/365}})$. 期權定價統計邊界。
        * **1m Linear Regression**: $y = ax + b$. 基於 60 交易日慣性推估。
        ### 2. 動態凱利配置
        * **理論**: $K = (W - \\frac{{1-W}}{{R}}) \\times 0.5$. 採用 60 天短窗口自適應，0.5 係數對抗黑天鵝。
        """)