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

# --- 1. 系統初始化與 Seeking Alpha 數據庫 ---
st.set_page_config(page_title="Posa Alpha 4.3 Flagship", layout="wide")
st.title("🛡️ Posa Alpha 4.3: 全球週期與 16 項全功能審計終端")

# SA 質化指標 (功能 10：標的觀點整合)
SA_DATA = {
    'MU': 'HBM 領導, PEG 0.20x', 'CLS': '15次盈餘上修',
    'AMD': 'M1400 加速器', 'URA': '鈾實物需求週期',
    'GOLD': '金銅雙週期', 'SOL-USD': '鏈上活動溢價'
}

try:
    FRED_API_KEY = st.secrets["FRED_API_KEY"]
    fred = Fred(api_key=FRED_API_KEY)
except:
    st.error("❌ 請檢查 Secrets 中的 FRED_API_KEY")
    st.stop()

# --- 2. 數據抓取引擎 (功能 1, 2, 3, 4, 6, 16) ---
@st.cache_data(ttl=3600)
def fetch_macro_onchain():
    """功能 1, 2, 3: 抓取宏觀與真實鏈上數據"""
    try:
        # 淨流動性
        liq = (fred.get_series('WALCL').iloc[-1] - fred.get_series('WTREGEN').iloc[-1] - fred.get_series('RRPONTSYD').iloc[-1]) / 1000
        # BTC.D (CoinGecko)
        btc_d = requests.get("https://api.coingecko.com/api/v3/global", timeout=10).json()['data']['market_cap_percentage']['btc']
        # MVRV (Blockchain.com)
        mvrv = requests.get("https://api.blockchain.info/charts/mvrv?timespan=1year&format=json", timeout=10).json()['values'][-1]['y']
    except: liq, btc_d, mvrv = 0, 52.5, 2.1
    return liq, btc_d, mvrv

@st.cache_data(ttl=600)
def fetch_market_data(tickers):
    """功能 6, 10, 11, 14: 修復 URA/0050 並抓取市場數據"""
    processed = [t.strip().upper() for t in tickers if t]
    benchmarks = ['QQQ', 'QLD', 'TQQQ', '0050.TW', 'BTC-USD', '^VIX', '^MOVE']
    full_list = list(set(processed + benchmarks))
    
    # 抓取 2 年以支援 Pi Cycle (功能 12)
    data = yf.download(full_list, period="2y", auto_adjust=True, progress=False)
    prices = data['Close'].ffill() # 解決台美股休市斷層
    
    earnings, news_feed = {}, {}
    for t in processed:
        try:
            tk = yf.Ticker(t)
            # 抓取財報 (功能 14)
            cal = tk.calendar
            if cal is not None and not cal.empty:
                earnings[t] = cal.loc['Earnings Date'].iloc[0].date()
            # 抓取新聞 (功能 16)
            news_feed[t] = tk.news[:3]
        except: pass
    return prices, earnings, news_feed

# --- 3. 核心邏輯計算 (功能 7, 8, 12, 13) ---
def run_audit_logic(t_prices, q_prices, qld_prices):
    """功能 7, 8, 13: 凱利、止損與預測"""
    last_p = t_prices.iloc[-1]
    # 1. 凱利勝率與盈虧比 (過去 120 天)
    rets = t_prices.pct_change().shift(-5)
    ema20 = t_prices.ewm(span=20).mean()
    sig = t_prices > ema20
    v_rets = rets[sig].dropna()
    win_p = (v_rets > 0).mean() if not v_rets.empty else 0.52
    odds = v_rets[v_rets > 0].mean() / abs(v_rets[v_rets < 0].mean()) if not v_rets.empty else 2.0
    kelly = max(0, (win_p - (1 - win_p) / odds) * 0.5)

    # 2. 1w Expected Move (功能 13)
    vol = t_prices.pct_change().std() * np.sqrt(252)
    move_1w = last_p * vol * np.sqrt(7/365)
    
    # 3. 1m Regression (功能 13)
    y = t_prices.tail(60).values.reshape(-1, 1)
    x = np.array(range(len(y))).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    pred_1m = model.predict([[len(y) + 22]])[0][0]
    
    # 4. 移動止損位 (功能 8)
    trailing_stop = t_prices.tail(252).max() * 0.93
    
    # 5. 效率 (vs QLD)
    efficiency = "🚀 高效" if (t_prices/qld_prices).iloc[-1] > (t_prices/qld_prices).iloc[-20] else "🐌 低效"
    
    return kelly, (last_p - move_1w, last_p + move_1w), pred_1m, trailing_stop, efficiency, win_p

# --- 4. 側邊欄：實戰輸入 Form (功能 5) ---
with st.sidebar.form("alpha_form"):
    st.header("💰 12.7萬資金部署審計")
    if 'portfolio_df' not in st.session_state:
        st.session_state.portfolio_df = pd.DataFrame([
            {"代號": "MU", "金額": 30000}, {"代號": "AMD", "金額": 25000},
            {"代號": "URA", "金額": 15000}, {"代號": "0050.TW", "金額": 40000},
            {"代號": "BTC-USD", "金額": 57000}
        ])
    edited_df = st.data_editor(st.session_state.portfolio_df, num_rows="dynamic")
    submit = st.form_submit_button("🚀 執行 16 項全方位審計")

# --- 5. 實戰渲染邏輯 ---
if submit:
    # 執行數據抓取
    user_tickers = edited_df["代號"].dropna().tolist()
    total_val = edited_df["金額"].sum()
    prices, earnings_map, news_map = fetch_market_data(user_tickers)
    liq, btc_d, mvrv = fetch_macro_onchain()
    
    # A. 宏觀地基與週期 (功能 1, 2, 3, 4)
    st.subheader("🌐 全球週期與利好出盡偵測")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MVRV 週期溫度", f"{mvrv:.2f}", delta="利多出盡危險" if mvrv > 3.0 else "週期安全")
    m2.metric("BTC.D 市佔率", f"{btc_d:.1f}%")
    m3.metric("VIX/MOVE (股債天氣)", f"{prices['^VIX'].iloc[-1]:.2f} / {prices['^MOVE'].iloc[-1]:.0f}")
    m4.metric("淨流動性", f"${liq:,.2f}B")

    # B. 比特幣 Pi Cycle Top 警報 (功能 12)
    st.divider()
    btc = prices['BTC-USD']
    ma111 = btc.rolling(111).mean()
    ma350_x2 = btc.rolling(350).mean() * 2
    
    st.subheader("🔮 週期逃命指標：Pi Cycle Top")
    if ma111.iloc[-1] > ma350_x2.iloc[-1]:
        st.error(f"🚨 **PI CYCLE TOP 觸發**：比特幣目前價格 ${btc.iloc[-1]:,.0f} 已進入週期頂部交叉！")
    else:
        st.success(f"✅ 週期安全：Pi Cycle 尚未交叉（壓力位：${ma350_x2.iloc[-1]:,.0f}）")
    
    

    # C. 即時脈搏：網格佈局 (功能 6, 11)
    st.subheader("⚡ 即時市場脈搏")
    display_tickers = [t for t in user_tickers if t in prices.columns]
    for i in range(0, len(display_tickers), 4):
        cols = st.columns(4)
        for j, t in enumerate(display_tickers[i:i+4]):
            curr_p = prices[t].iloc[-1]
            chg = (prices[t].iloc[-1]/prices[t].iloc[-2]-1)*100
            cols[j].metric(t, f"${curr_p:,.2f}", f"{chg:.2f}%")

    # D. 深度審計表 (功能 7, 8, 10, 13, 14)
    st.divider()
    st.subheader("📋 跨市場深度審計 (含凱利與三維預測)")
    audit_list = []
    today = datetime.now().date()
    for t in user_tickers:
        if t not in prices.columns or t in ['QQQ', 'QLD', 'TQQQ']: continue
        
        # 執行 Part 1 的計算邏輯
        k_w, range_1w, p_1m, t_stop, eff, win_p = run_audit_logic(prices[t], prices['QQQ'], prices['QLD'])
        
        # 財報 (功能 14)
        e_date = earnings_map.get(t)
        days_to_e = (e_date - today).days if e_date else 999
        e_alert = f"⚠️ {days_to_e}d" if days_to_e <= 7 else f"{days_to_e}d"
        
        audit_list.append({
            "標的": t, "效率審計": eff, 
            "20EMA": "🟢" if prices[t].iloc[-1] > prices[t].ewm(span=20).mean().iloc[-1] else "🔴",
            "1w 預期震盪": f"${range_1w[0]:.1f} - ${range_1w[1]:.1f}",
            "1m 回歸目標": f"${p_1m:.1f}",
            "凱利權重": f"{k_w*100:.1f}%",
            "移動止損": f"${t_stop:.1f}",
            "財報倒數": e_alert
        })
    st.table(pd.DataFrame(audit_list))

    

    # E. 熱力圖與趨勢視覺化 (功能 9, 15)
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("🤝 板塊相關性審計")
        st.plotly_chart(px.imshow(prices[user_tickers].corr(), text_auto=".2f", color_continuous_scale='RdBu_r'), use_container_width=True)
    with col_right:
        st.subheader("📈 20EMA 生命線審核")
        t_plot = st.selectbox("選擇標的", user_tickers)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prices.index, y=prices[t_plot], name="價格"))
        fig.add_trace(go.Scatter(x=prices.index, y=prices[t_plot].ewm(span=20).mean(), name="20EMA", line=dict(dash='dash')))
        st.plotly_chart(fig, use_container_width=True)

    # F. 量化新聞驚奇 (功能 16)
    st.divider()
    st.subheader("📰 重要消息與驚奇指數")
    for t in user_tickers:
        if t in news_map:
            with st.expander(f"{t} 核心消息庫"):
                for n in news_map[t]:
                    st.write(f"🔗 [{n['title']}]({n['link']})")

    # G. 判斷依準手冊 (功能 15)
    st.divider()
    st.subheader("📚 Posa 旗艦審計決策手冊")
    with st.expander("查看所有量化判斷依準"):
        st.markdown(f"""
        ### 1. 未來預測模型 (LaTeX 依據)
        * **1w Expected Move**: 基於 Black-Scholes 波動率投射：$Price \\pm (Price \\times \\sigma \\times \\sqrt{{7/365}})$.
        * **1m Regression**: 利用過去 60 交易日進行線性回歸 $y = ax + b$，推估慣性目標。
        
        ### 2. 凱利配置 (Kelly Criterion)
        * 實戰公式：$K = W - \\frac{{1-W}}{{R}}$ (其中 $W$ 為勝率，$R$ 為盈虧比)。
        * **縮放係數**：系統自動採用 0.5 縮放以對抗黑天鵝。

        ### 3. 效率與止損
        * **🚀 高效**: 代表該標的跑贏 **QLD (2x 槓桿納指)**。
        * **移動止損**: 取過去一年最高點之 93%（即 7% 回撤止損）。
        """)

else:
    st.info("💡 請在左方填寫持倉後，點擊『🚀 執行 16 項全方位審計』開始決策分析。")