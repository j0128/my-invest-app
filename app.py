import streamlit as st
import yfinance as yf
import pandas as pd
from fredapi import Fred
import plotly.express as px
import plotly.graph_objects as go
import time
import random
import requests
from datetime import datetime

# --- 1. 系統設定 ---
st.set_page_config(page_title="Posa Alpha 3.8.1", layout="wide")
st.title("🛡️ Posa Alpha 3.8.1: 跨市場週期與鏈上審計中心")

# 核心標的：2026 金股 + 幣圈 + 台股基準
SA_TOP_10 = ['MU', 'AMD', 'CLS', 'CIEN', 'COHR', 'ALL', 'INCY', 'GOLD', 'WLDN', 'ATI']
CRYPTO_BENCH = ['BTC-USD', 'SOL-USD', 'ETH-USD']
BENCHMARKS = ['QQQ', '0050.TW', '^VIX']

try:
    FRED_API_KEY = st.secrets["FRED_API_KEY"]
    fred = Fred(api_key=FRED_API_KEY)
except:
    st.error("❌ 請在 Secrets 設定 FRED_API_KEY")
    st.stop()

# --- 2. 真實數據抓取 (BTC.D & MVRV) ---
@st.cache_data(ttl=3600)
def fetch_onchain_metrics():
    try:
        # 1. BTC.D (CoinGecko)
        global_resp = requests.get("https://api.coingecko.com/api/v3/global", timeout=10).json()
        btc_d = global_resp['data']['market_cap_percentage']['btc']
        
        # 2. MVRV (Blockchain.com)
        mvrv_resp = requests.get("https://api.blockchain.info/charts/mvrv?timespan=2years&format=json", timeout=10).json()
        current_mvrv = mvrv_resp['values'][-1]['y']
    except:
        btc_d, current_mvrv = 52.5, 2.1  # 異常時顯示中性預設值
    return btc_d, current_mvrv

@st.cache_data(ttl=600)
def fetch_master_data(tickers):
    prices, info = pd.DataFrame(), {}
    all_needed = list(set(tickers + SA_TOP_10 + CRYPTO_BENCH + BENCHMARKS))
    for t in all_needed:
        try:
            time.sleep(0.3)
            tk = yf.Ticker(t)
            df = tk.history(period="2y") # 抓取 2 年解決 $nan 問題
            if not df.empty:
                # 處理 yfinance 多索引問題
                close_series = df['Close']
                if isinstance(close_series, pd.DataFrame):
                    close_series = close_series.iloc[:, 0]
                prices[t] = close_series
                info[t] = {
                    "price": close_series.iloc[-1],
                    "change": (close_series.iloc[-1] / close_series.iloc[-2] - 1) * 100
                }
        except: continue
    
    try:
        liq = (fred.get_series('WALCL').iloc[-1] - fred.get_series('WTREGEN').iloc[-1] - fred.get_series('RRPONTSYD').iloc[-1]) / 1000
    except: liq = 0
    return liq, prices, info

# --- 3. 邏輯運算 ---
def get_kelly_stats(t_prices, q_prices):
    try:
        ema20 = t_prices.ewm(span=20).mean()
        rs = t_prices / q_prices
        sig = (t_prices > ema20) & (rs > rs.rolling(20).mean())
        rets = t_prices.shift(-5) / t_prices - 1
        v_rets = rets[sig].dropna() # 修復語法點錯誤
        if len(v_rets) < 5: return 0.52, 2.0
        return (v_rets > 0).mean(), (v_rets[v_rets > 0].mean() / abs(v_rets[v_rets < 0].mean()))
    except: return 0.5, 1.5

# --- 4. 頁面渲染 ---
try:
    st.sidebar.header("💰 實戰持倉設定")
    if 'portfolio_df' not in st.session_state:
        st.session_state.portfolio_df = pd.DataFrame([
            {"代號": "MU", "金額": 30000}, {"代號": "AMD", "金額": 36000},
            {"代號": "0050.TW", "金額": 70000}, {"代號": "SOL-USD", "金額": 100000}
        ])
    edited_df = st.sidebar.data_editor(st.session_state.portfolio_df, num_rows="dynamic")
    user_tickers = edited_df["代號"].tolist()
    total_val = edited_df["金額"].sum()

    liq, prices, market_info = fetch_master_data(user_tickers)
    btc_d, mvrv = fetch_onchain_metrics()
    vix = prices['^VIX'].iloc[-1]

    # A. 週期溫度看板
    st.subheader("🌡️ 週期與情緒審計 (利好出盡偵測)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MVRV 週期溫度", f"{mvrv:.2f}", delta="過熱" if mvrv > 3 else "安全")
    m2.metric("BTC.D 市佔率", f"{btc_d:.1f}%")
    m3.metric("VIX 天氣", f"{vix:.2f}")
    m4.metric("淨流動性", f"${liq:.2f}B")

    # B. 即時脈搏 (每行 4 檔，解決位數遮斷)
    st.divider()
    st.subheader("⚡ 即時市場脈搏")
    rows = [user_tickers[i:i + 4] for i in range(0, len(user_tickers), 4)]
    for row in rows:
        cols = st.columns(4)
        for i, t in enumerate(row):
            if t in market_info:
                cols[i].metric(t, f"${market_info[t]['price']:,.2f}", f"{market_info[t]['change']:.2f}%")

    # C. 趨勢與雙指標預判
    st.divider()
    st.subheader("🎯 趨勢健康度與未來走勢預判")
    audit_data = []
    for t in user_tickers:
        if t not in prices.columns or t in BENCHMARKS: continue
        curr_p = prices[t].iloc[-1]
        ema20 = prices[t].ewm(span=20).mean().iloc[-1]
        rs_qqq = (prices[t]/prices['QQQ']).iloc[-1] > (prices[t]/prices['QQQ']).rolling(20).mean().iloc[-1]
        rs_tw = (prices[t]/prices['0050.TW']).iloc[-1] > (prices[t]/prices['0050.TW']).rolling(20).mean().iloc[-1] if '0050.TW' in prices.columns else False
        
        status = "🔥 加速" if (curr_p > ema20 and rs_qqq) else "🛑 轉弱" if (curr_p < ema20) else "🛡️ 盤整"
        
        audit_data.append({
            "標的": t, "20EMA": "🟢 站穩" if curr_p > ema20 else "🔴 跌破",
            "贏過 QQQ": "✅" if rs_qqq else "❌",
            "贏過 0050": "✅" if rs_tw else "❌",
            "未來走勢": status, "止損位": f"${prices[t].max()*0.93:.2f}"
        })
    st.table(pd.DataFrame(audit_data))

    # D. 跨市場比較分析
    st.subheader("📊 未來動力：相對強度曲線 (vs QQQ & 0050)")
    target = st.selectbox("選擇要深度對比的標的", [t for t in user_tickers if t not in BENCHMARKS])
    if target in prices.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prices.index, y=prices[target]/prices['QQQ'], name="vs QQQ (美股)"))
        fig.add_trace(go.Scatter(x=prices.index, y=prices[target]/prices['0050.TW'], name="vs 0050 (台股)"))
        st.plotly_chart(fig, use_container_width=True)

    # E. 文字報告
    st.divider()
    st.subheader("🖋️ Alpha 3.8.1 會計師審計報告")
    with st.container(border=True):
        if mvrv > 3.0: st.error("🚨 **週期性利好出盡警告**：MVRV 超過 3.0。這已不是震盪，而是週期性頂部，強烈建議撤出大部分倉位。")
        if vix > 18: st.warning("⚠️ **天氣惡化**：VIX 升高，應嚴守止損，切勿在此刻加碼。")
        for t in user_tickers:
            if t in prices.columns and prices[t].iloc[-1] < prices[t].ewm(span=20).mean().iloc[-1]:
                st.write(f"🛑 **指令：** {t} 已跌破生命線 (20EMA)。任何大漲皆視為『逃命反彈』，請執行減碼。")

except Exception as e:
    st.error(f"系統自動審計中發生異常: {e}")