import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import time
import random
import plotly.express as px
import plotly.graph_objects as go
from fredapi import Fred
from datetime import datetime

# --- 1. 系統設定與 Seeking Alpha 數據庫 ---
st.set_page_config(page_title="Posa Alpha 3.9 Final", layout="wide")
st.title("🛡️ Posa Alpha 3.9: 全功能跨市場終極審計系統")

# SA 2026 十大金股與深度數據
SA_INSIGHTS = {
    'MU': {'note': 'HBM 領先, PEG 0.20x', 'growth': '206%'},
    'CLS': {'note': '15次盈餘上修, AI整合核心', 'growth': '51%'},
    'AMD': {'note': 'OpenAI 夥伴, M1400 加速器', 'growth': '34%'},
    'ALL': {'note': '連續 32 年配息, 高品質保險', 'growth': '193%'},
    'GOLD': {'note': '金+銅 雙避險, 能源轉型受益', 'growth': '58%'}
}

try:
    FRED_API_KEY = st.secrets["FRED_API_KEY"]
    fred = Fred(api_key=FRED_API_KEY)
except:
    st.error("❌ 請在 Secrets 設定 FRED_API_KEY")
    st.stop()

# --- 2. 核心數據抓取 (含鏈上真實值) ---
@st.cache_data(ttl=3600)
def fetch_onchain_metrics():
    try:
        # BTC.D (CoinGecko)
        btc_d = requests.get("https://api.coingecko.com/api/v3/global", timeout=10).json()['data']['market_cap_percentage']['btc']
        # MVRV (Blockchain.com)
        mvrv_data = requests.get("https://api.blockchain.info/charts/mvrv?timespan=1year&format=json", timeout=10).json()
        current_mvrv = mvrv_data['values'][-1]['y']
    except:
        btc_d, current_mvrv = 52.5, 2.1
    return btc_d, current_mvrv

@st.cache_data(ttl=600)
def fetch_master_data(tickers):
    # 台股代碼自動校正
    processed = [t.upper() if ".TW" in t.upper() else t for t in tickers]
    benchmarks = ['QQQ', '0050.TW', '^VIX', '^MOVE', 'BTC-USD']
    full_list = list(set(processed + benchmarks))
    
    # 抓取 1 年資料確保 EMA 穩定，使用 auto_adjust 修復台股
    data = yf.download(full_list, period="1y", interval="1d", auto_adjust=True, progress=False)
    prices = data['Close'].ffill()
    
    # 抓取財報日
    earnings = {}
    for t in processed:
        if "-" not in t and ".TW" not in t:
            try:
                cal = yf.Ticker(t).calendar
                if cal is not None and not cal.empty:
                    earnings[t] = cal.loc['Earnings Date'].iloc[0].strftime('%Y-%m-%d')
            except: pass
    return prices, earnings

# --- 3. 側邊欄設定 ---
st.sidebar.header("💰 12.7萬實戰資產配置")
if 'portfolio_df' not in st.session_state:
    st.session_state.portfolio_df = pd.DataFrame([
        {"代號": "MU", "金額": 30000},
        {"代號": "AMD", "金額": 25000},
        {"代號": "0050.TW", "金額": 40000},
        {"代號": "BTC-USD", "金額": 32000}
    ])
edited_df = st.sidebar.data_editor(st.session_state.portfolio_df, num_rows="dynamic")
user_tickers = edited_df["代號"].tolist()
total_val = edited_df["金額"].sum()

# --- 4. 執行與渲染 ---
try:
    prices, earnings_dates = fetch_master_data(user_tickers)
    btc_d, mvrv = fetch_onchain_metrics()
    net_liq = (fred.get_series('WALCL').iloc[-1] - fred.get_series('WTREGEN').iloc[-1] - fred.get_series('RRPONTSYD').iloc[-1]) / 1000
    
    # A. 宏觀與週期儀表盤
    st.subheader("🌐 全球週期與地基審計")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MVRV (週期溫度)", f"{mvrv:.2f}", delta="過熱" if mvrv > 3 else "安全")
    m2.metric("BTC.D (資金羅盤)", f"{btc_d:.1f}%", delta="山寨季偵測" if btc_d < 45 else None)
    m3.metric("VIX (股市天氣)", f"{prices['^VIX'].iloc[-1]:.2f}")
    m4.metric("淨流動性", f"${net_liq:.2f}B")

    # B. 即時脈搏 (解決位數遮斷)
    st.divider()
    st.subheader("⚡ 即時市場脈搏")
    rows = [user_tickers[i:i + 4] for i in range(0, len(user_tickers), 4)]
    for row in rows:
        cols = st.columns(4)
        for i, t in enumerate(row):
            if t in prices.columns:
                curr = prices[t].iloc[-1]
                chg = (prices[t].iloc[-1]/prices[t].iloc[-2]-1)*100
                cols[i].metric(t, f"${curr:,.2f}", f"{chg:.2f}%")

    # C. 深度審計表 (整合所有指標)
    st.subheader("📋 跨市場深度審計與走勢預判")
    audit_data = []
    today = datetime.now().date()
    for t in user_tickers:
        if t not in prices.columns or t in ['^VIX', '^MOVE', 'QQQ']: continue
        curr = prices[t].iloc[-1]
        ema20 = prices[t].ewm(span=20).mean().iloc[-1]
        
        # 贏過 QQQ & 0050
        win_qqq = (prices[t]/prices['QQQ']).iloc[-1] > (prices[t]/prices['QQQ']).rolling(20).mean().iloc[-1]
        win_0050 = (prices[t]/prices['0050.TW']).iloc[-1] > (prices[t]/prices['0050.TW']).rolling(20).mean().iloc[-1]
        
        # 凱利勝率回測 (過去90天)
        rets = prices[t].shift(-5) / prices[t] - 1
        win_p = (rets.tail(90) > 0).mean()
        
        # 財報預警
        e_date = earnings_dates.get(t, "N/A")
        e_alert = "⚠️ 7天內" if e_date != "N/A" and (datetime.strptime(e_date, '%Y-%m-%d').date() - today).days <= 7 else "✅"
        
        audit_data.append({
            "標的": t, "SA 觀點": SA_INSIGHTS.get(t, {}).get('note', '-'),
            "20EMA": "🟢 站穩" if curr > ema20 else "🔴 跌破",
            "勝過QQQ": "✅" if win_qqq else "❌",
            "勝過0050": "✅" if win_0050 else "❌",
            "回測勝率": f"{win_p*100:.0f}%",
            "財報風險": e_alert,
            "止損價位": f"${prices[t].max()*0.93:,.2f}"
        })
    st.table(pd.DataFrame(audit_data))

    # D. 相關性矩陣 (biibo 沒看到的指標)
    st.subheader("🤝 板塊集中度 (相關性) 審計")
    st.plotly_chart(px.imshow(prices[user_tickers].corr(), text_auto=".2f", color_continuous_scale='RdBu_r'), use_container_width=True)

    # E. 終極文字報告 (決策大腦)
    st.divider()
    st.