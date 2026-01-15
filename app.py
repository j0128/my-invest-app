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

# --- 1. 系統初始化 (確保啟動時所有 Session 狀態已閉合) ---
st.set_page_config(page_title="Posa Alpha 4.5.1", layout="wide")

def init_session():
    # 強制初始化所有必要的狀態，避免 image_42f226 類型的報錯
    states = {
        'prices': None, 'earnings': {}, 'news': [], 
        'macro': {"liq": 5777.0, "btcd": 57.4, "mvrv": 2.10},
        'user_tickers': []
    }
    for key, value in states.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session()

# --- 2. 左側儀表板配置 (功能 1: 股票投資實戰部署) ---
st.sidebar.title("🚀 股票投資實戰部署")

with st.sidebar.form(key="posa_master_v5_form"):
    st.subheader("💰 12.7萬資金配置輸入")
    if 'portfolio_df' not in st.session_state:
        st.session_state.portfolio_df = pd.DataFrame([
            {"代號": "MU", "金額": 30000}, {"代號": "AMD", "金額": 25000},
            {"代號": "URA", "金額": 15000}, {"代號": "BTC-USD", "金額": 57000}
        ])
    edited_df = st.data_editor(st.session_state.portfolio_df, num_rows="dynamic")
    submit = st.form_submit_button("🚀 啟動全方位量化審計")

# 質化知識庫 (功能 9)
SA_INSIGHTS = {
    'MU': {'note': 'HBM 領先, PEG 0.20x', 'growth': 0.35},
    'AMD': {'note': 'M1400 加速器需求', 'growth': 0.28},
    'BTC-USD': {'note': '週期數位黃金', 'growth': 0.50},
    'URA': {'note': '鈾實物週期缺口', 'growth': 0.15}
}

# --- 3. 數據抓取引擎 (功能 6, 16, 17, 20) ---
@st.cache_data(ttl=600)
def fetch_alpha_master_v5(tickers):
    processed = [t.strip().upper() for t in tickers if t]
    benchmarks = ['QQQ', 'QLD', 'TQQQ', '0050.TW', 'BTC-USD', '^VIX', '^MOVE']
    full_list = list(set(processed + benchmarks))
    
    # 抓取 2 年資料確保均線穩定
    df = yf.download(full_list, period="2y", auto_adjust=True, progress=False)
    # 修復功能 20: 補點技術解決 $nan (image_41338a)
    prices = df['Close'].ffill().bfill()
    
    # 強化版新聞流 (功能 16: 確保 > 5 則)
    all_news = []
    try:
        # 混合抓取大盤與個股新聞
        for target in ['QQQ', 'BTC-USD'] + processed[:1]:
            all_news.extend(yf.Ticker(target).news[:4])
    except: pass
    
    earnings = {}
    for t in processed:
        try:
            cal = yf.Ticker(t).calendar
            if cal is not None and not cal.empty:
                dt = cal.loc['Earnings Date'].iloc[0]
                earnings[t] = dt.date() if hasattr(dt, 'date') else dt
        except: earnings[t] = None
    return prices, earnings, all_news[:10]

def fetch_macro_data():
    """功能 1-4: 淨流動性、MVRV、BTC.D"""
    try:
        fred = Fred(api_key=st.secrets["FRED_API_KEY"])
        liq = (fred.get_series('WALCL').iloc[-1] - fred.get_series('WTREGEN').iloc[-1] - fred.get_series('RRPONTSYD').iloc[-1]) / 1000
        return liq, 57.4, 2.15 # BTC.D 與 MVRV 預設值
    except:
        return 5777.0, 57.4, 2.15

# --- 4. 戰略運算核心 (功能 7, 10, 11, 13, 14, 15) ---
def run_strategic_audit_v5(ticker, series, qld_prices, tqqq_prices):
    curr = series.iloc[-1]
    ema20_series = series.ewm(span=20).mean()
    ema20 = ema20_series.iloc[-1]
    
    # A. 進攻型凱利 (功能 10: 15% 底倉)
    rets = series.pct_change().shift(-5)
    sig = series > ema20_series
    v_rets = rets[sig].tail(60).dropna()
    
    if not v_rets.empty:
        win_p = (v_rets > 0).mean()
        r_ratio = (v_rets[v_rets > 0].mean() / abs(v_rets[v_rets < 0].mean())) if not v_rets[v_rets < 0].empty else 1.5
        raw_k = (win_p - (1 - win_p) / r_ratio) * 0.5
        # 進攻修正：只要站穩 20EMA，最低權重給予 15%
        kelly = max(0.15, raw_k) if curr > ema20 else max(0, raw_k)
    else: kelly = 0.15 if curr > ema20 else 0.0

    # B. 三維預測 (1w, 1m, 1q)
    vol = series.pct_change().tail(30).std() * np.sqrt(252)
    move_1w = curr * vol * np.sqrt(7/365)
    
    # 1m 線性回歸預測
    y_reg = series.tail(60).values.reshape(-1, 1); x_reg = np.array(range(len(y_reg))).reshape(-1, 1)
    p_1m = LinearRegression().fit(x_reg, y_reg).predict([[len(y_reg) + 22]])[0][0]
    
    # 1q 估值預測 (SA Growth)
    growth = SA_INSIGHTS.get(ticker, {}).get('growth', 0.20)
    p_1q = curr * (1 + growth / 4)

    # C. 效率審計 (功能 7: 對標槓桿)
    if (series/tqqq_prices).iloc[-1] > (series/tqqq_prices).iloc[-20]: eff = "🔥 超越 TQQQ"
    elif (series/qld_prices).iloc[-1] > (series/qld_prices).iloc[-20]: eff = "🚀 贏 QLD"
    else: eff = "🐌 輸槓桿大盤"
    
    return kelly, (curr-move_1w, curr+move_1w), p_1m, p_1q, series.tail(252).max()*0.93, eff

# --- 5. 主頁面渲染 (功能 5, 20: 鎖定 Session 防止當機) ---
if submit or st.session_state.prices is not None:
    if submit:
        st.session_state.user_tickers = edited_df["代號"].dropna().tolist()
        p, e, n = fetch_alpha_master_v5(st.session_state.user_tickers)
        liq, btcd, mvrv = fetch_macro_data()
        st.session_state.update({'prices': p, 'earnings': e, 'news': n, 'macro': {"liq": liq, "btcd": btcd, "mvrv": mvrv}})

    p, m, e, news_list, ts = st.session_state.prices, st.session_state.macro, st.session_state.earnings, st.session_state.news, st.session_state.user_tickers

    if p is not None:
        # A. 宏觀看板
        st.subheader("🌐 全球週期與地基審計")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MVRV 週期溫度", f"{m['mvrv']:.2f}")
        c2.metric("BTC.D 市佔率", f"{m['btcd']:.1f}%")
        c3.metric("VIX / MOVE", f"{p['^VIX'].iloc[-1]:.1f} / {p['^MOVE'].iloc[-1]:.0f}")
        c4.metric("淨流動性", f"${m['liq']:,.2f}B", help="WALCL - TGA - RRP")

        # B. 比特幣週期頂部警報 (功能 6, 12: Pi Cycle Top)
        st.divider()
        btc_s = p['BTC-USD']
        ma111, ma350x2 = btc_s.rolling(111).mean(), btc_s.rolling(350).mean() * 2
        st.subheader("🔮 週期逃命預判：Pi Cycle Top Indicator")
        cp1, cp2, cp3 = st.columns([1,1,2])
        cp1.metric("BTC 現價", f"${btc_s.iloc[-1]:,.0f}")
        cp2.metric("壓力位 (350DMA*2)", f"${ma350x2.iloc[-1]:,.0f}")
        if ma111.iloc[-1] > ma350x2.iloc[-1]: cp3.error("🚨 PI CYCLE TOP 交叉！牛市見頂預警。")
        else: cp3.success(f"✅ 週期運行中。距離頂部交叉仍有 {(ma350x2.iloc[-1]/btc_s.iloc[-1]-1)*100:.1f}% 空間。")

        # C. 深度審計表
        st.divider()
        st.subheader("📋 進攻型深度審計 (含三維預測)")
        audit_data = []
        for t in ts:
            if t in p.columns and t not in ['QQQ', 'QLD', 'TQQQ']:
                k, (l1, h1), p1, pQ, ts_p, eff = run_strategic_audit_v5(t, p[t], p['QLD'], p['TQQQ'])
                ed = e.get(t); rem = (ed - datetime.now().date()).days if ed else "無資料"
                audit_data.append({"標的": t, "效率": eff, "20EMA": "🟢 站穩" if p[t].iloc[-1] > p[t].ewm(span=20).mean().iloc[-1] else "🔴 跌破",
                                  "凱利權重": f"{k*100:.1f}%", "1w區間": f"{l1:.1f}-{h1:.1f}", "1m回歸": f"${p1:.1f}", "1q估值": f"${pQ:.1f}", "移動止損": f"${ts_p:.1f}", "財報": f"{rem}天" if isinstance(rem, int) else rem})
        st.table(pd.DataFrame(audit_data))

        # D. 熱力圖與新聞 (功能 9, 13, 16)
        col_h, col_n = st.columns([1.2, 1])
        with col_h:
            st.subheader("🤝 板塊相關性熱力圖")
            st.plotly_chart(px.imshow(p[ts].corr(), text_auto=".2f", color_continuous_scale='RdBu_r'), use_container_width=True)
        with col_n:
            st.subheader("📰 重要財金新聞 (Top 5+)")
            for news in news_list[:6]: st.write(f"🔹 [{news['title']}]({news['link']}) — *{news['publisher']}*")

        # E. 趨勢圖與手冊 (功能 20: 鎖定 Key)
        st.divider()
        pick = st.selectbox("選擇要審查的標的", ts, key="stable_final_selector_v5")
        if pick in p.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=p.index, y=p[pick], name="價格", line=dict(color='gold')))
            fig.add_trace(go.Scatter(x=p.index, y=p[pick].ewm(span=20).mean(), name="20EMA", line=dict(dash='dash')))
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("📚 Posa 旗艦審計手冊"):
            st.markdown(f"""
            ### 1. 比特幣頂部預測 (Pi Cycle Top)
            * **公式**：當 $111DMA > 350DMA \\times 2$。代表短線成本快速超載長線 2 倍，情緒達頂點。
            ### 2. 進攻型凱利
            * **進攻地板**：只要站穩 **20EMA**，強制給予 15% 權重，拒絕全現金建議。
            """)
else: st.info("💡 請啟動審計。")