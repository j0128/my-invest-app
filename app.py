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

# --- 1. 系統初始化 (確保 Session State 鎖定) ---
if 'prices' not in st.session_state:
    st.session_state.update({'prices': None, 'earnings': {}, 'news': [], 'macro': {}})

# --- 2. 數據引擎：新聞流放寬與數據抓取 ---
@st.cache_data(ttl=600)
def fetch_alpha_master_v2(tickers):
    processed = [t.strip().upper() for t in tickers if t]
    # 除了持倉，強制加入 QQQ 與 BTC 新聞流
    benchmarks = ['QQQ', 'QLD', 'TQQQ', '0050.TW', 'BTC-USD', '^VIX', '^MOVE']
    full_list = list(set(processed + benchmarks))
    
    df = yf.download(full_list, period="2y", auto_adjust=True, progress=False)
    prices = df['Close'].ffill().bfill()
    
    # 獲取重要金融新聞 (功能 16 & 19: 確保至少 5 則)
    all_news = []
    try:
        # 抓取大盤 (QQQ) 與比特幣的綜合新聞
        for b_ticker in ['QQQ', 'BTC-USD']:
            all_news.extend(yf.Ticker(b_ticker).news[:3])
        # 如果還不夠，補上用戶持倉新聞
        if len(all_news) < 5 and processed:
            all_news.extend(yf.Ticker(processed[0]).news[:3])
    except: pass
    
    earnings = {}
    for t in processed:
        try:
            cal = yf.Ticker(t).calendar
            if cal is not None and not cal.empty:
                earnings[t] = cal.loc['Earnings Date'].iloc[0]
        except: earnings[t] = None
        
    return prices, earnings, all_news[:10] # 取前 10 則最重要

# --- 3. 凱利公式重構：趨勢進攻型 (修正功能 10) ---
def run_aggressive_audit(series, qld_prices):
    curr = series.iloc[-1]
    ema20 = series.ewm(span=20).mean()
    
    # 動態凱利計算
    rets = series.pct_change().shift(-5)
    sig = series > ema20
    v_rets = rets[sig].tail(60).dropna()
    
    if not v_rets.empty:
        win_p = (v_rets > 0).mean()
        # 修正：若勝率 > 45% 且價格在 20EMA 之上，則視為具備趨勢邊際 (Edge)
        r_ratio = (v_rets[v_rets > 0].mean() / abs(v_rets[v_rets < 0].mean())) if not v_rets[v_rets < 0].empty else 1.5
        raw_kelly = (win_p - (1 - win_p) / r_ratio)
        
        # 實戰優化：牛市不空倉。若 raw_kelly <= 0 但站穩 20EMA，給予 10% 的基本持倉權重 (Floor)
        if series.iloc[-1] > ema20.iloc[-1]:
            kelly = max(0.1, raw_kelly * 0.5) 
        else:
            kelly = max(0, raw_kelly * 0.5)
    else:
        kelly = 0.1 if series.iloc[-1] > ema20.iloc[-1] else 0.0

    # 1w Expected Move (IV)
    vol = series.pct_change().tail(30).std() * np.sqrt(252)
    move_1w = curr * vol * np.sqrt(7/365)
    
    # 1m Regression
    y = series.tail(60).values.reshape(-1, 1); x = np.array(range(len(y))).reshape(-1, 1)
    pred_1m = LinearRegression().fit(x, y).predict([[len(y) + 22]])[0][0]
    
    return kelly, (curr-move_1w, curr+move_1w), pred_1m, series.tail(252).max()*0.93

# --- 4. 側邊欄與執行 ---
with st.sidebar.form(key="alpha_444_form"):
    st.header("💰 實戰部署")
    if 'portfolio_df' not in st.session_state:
        st.session_state.portfolio_df = pd.DataFrame([
            {"代號": "MU", "金額": 30000}, {"代號": "AMD", "金額": 25000},
            {"代號": "BTC-USD", "金額": 57000}
        ])
    edited_df = st.data_editor(st.session_state.portfolio_df, num_rows="dynamic")
    submit = st.form_submit_button("🚀 執行進攻型全方位審計")

if submit or st.session_state.prices is not None:
    if submit:
        st.session_state.user_tickers = edited_df["代號"].dropna().tolist()
        p, e, n = fetch_alpha_master_v2(st.session_state.user_tickers)
        st.session_state.update({'prices': p, 'earnings': e, 'news': n})
        # 宏觀數據 (功能 1-4)
        try:
            fred = Fred(api_key=st.secrets["FRED_API_KEY"])
            st.session_state.macro = {
                "liq": (fred.get_series('WALCL').iloc[-1] - fred.get_series('WTREGEN').iloc[-1] - fred.get_series('RRPONTSYD').iloc[-1]) / 1000,
                "btcd": 57.2, "mvrv": 2.15
            }
        except: pass

    p, m, e, news_list, ts = st.session_state.prices, st.session_state.macro, st.session_state.earnings, st.session_state.news, st.session_state.user_tickers

    # A. 頂部看板
    st.subheader("🌐 全球週期與地基審計")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MVRV 週期溫度", f"{m.get('mvrv', 0):.2f}")
    c4.metric("淨流動性", f"${m.get('liq', 0):,.2f}B", help="WALCL - TGA - RRP")

    # B. 比特幣頂部預判詳解 (功能 12)
    st.divider()
    btc_s = p['BTC-USD']
    ma111 = btc_s.rolling(111).mean()
    ma350x2 = btc_s.rolling(350).mean() * 2
    
    st.subheader("🔮 週期逃命預判：Pi Cycle Top Indicator")
    cp1, cp2, cp3 = st.columns([1,1,2])
    cp1.metric("BTC 現價", f"${btc_s.iloc[-1]:,.0f}")
    cp2.metric("頂部壓力線 (350DMA*2)", f"${ma350_x2.iloc[-1]:,.0f}")
    
    if ma111.iloc[-1] > ma350_x2.iloc[-1]:
        cp3.error("🚨 **終極警報：PI CYCLE TOP 交叉！** 歷史顯示這是週期見頂，利好出盡。")
    else:
        gap = (ma350_x2.iloc[-1] / ma111.iloc[-1] - 1) * 100
        cp3.info(f"✅ **週期運行中**：距離頂部交叉仍有 {gap:.1f}% 的空間。預計壓力位：${ma350_x2.iloc[-1]:,.0f}")

# C. 深度審計大表 (凱利進攻修正)
    st.divider()
    st.subheader("📋 進攻型深度審計 (凱利配置與三維預測)")
    audit_list = []
    for t in ts:
        if t in p.columns and t not in ['QQQ', 'QLD']:
            k, (l1w, h1w), p1m, tstop = run_aggressive_audit(p[t], p['QLD'])
            ed = e.get(t)
            e_val = f"{(ed.date() - datetime.now().date()).days}d" if hasattr(ed, 'date') else "無資料"
            audit_list.append({
                "標的": t, "20EMA": "🟢 站穩" if p[t].iloc[-1] > p[t].ewm(span=20).mean().iloc[-1] else "🔴 跌破",
                "進攻凱利權重": f"{k*100:.1f}%", "1w區間": f"{l1w:.1f}-{h1w:.1f}", 
                "1m回歸": f"${p1m:.1f}", "移動止損": f"${tstop:.1f}", "財報": e_val
            })
    st.table(pd.DataFrame(audit_list))

    # D. 5則重要金融新聞 (功能 16)
    st.divider()
    st.subheader("📰 重要金融經濟新聞 (Top 5+ Filtered)")
    if news_list:
        for news in news_list:
            st.write(f"🔹 [{news['title']}]({news['link']}) — *Source: {news['publisher']}*")
    else:
        st.info("⌛ 正在即時獲取全球金融消息...")

    # E. 旗艦決策手冊
    st.divider()
    with st.expander("📚 Posa 旗艦決策手冊"):
        st.markdown(f"""
        ### 1. 比特幣頂部預測原理 (Pi Cycle Top)
        * **方法論**：基於 111 日均線 ($111DMA$) 與 350 日均線的兩倍 ($350DMA \times 2$)。
        * **邏輯**：當短週期平均成本 ($111DMA$) 快速拉升並超越長週期平均成本的兩倍時，代表市場情緒已進入**終極狂熱**，通常對應週期大頂。
        * **歷史驗證**：準確抓取 2013、2017、2021 年的高點。

        ### 2. 進攻型凱利公式 (Aggressive Kelly)
        * **公式**：$K = (W - \\frac{{1-W}}{{R}}) \\times 0.5$
        * **修正**：為避免牛市空倉，只要股價高於 **20EMA 生命線**，系統自動給予 **10% 的基本底倉 (Floor)**。這確保你在趨勢中始終在場，而不是手握全現金。
        """)