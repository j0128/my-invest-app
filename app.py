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

# --- 1. 系統初始化與 Seeking Alpha 數據庫 ---
st.set_page_config(page_title="Posa Alpha 4.0", layout="wide")
st.title("🛡️ Posa Alpha 4.0: 全球週期與實戰審計終端")

# SA 知識庫 (整合 biibo 沒看到的質化指標)
SA_INSIGHTS = {
    'MU': {'note': 'HBM 領導者, PEG 0.20x (折價 88%)', 'growth': '206%'},
    'CLS': {'note': '15次盈餘上修, 0次下修', 'growth': '51%'},
    'AMD': {'note': 'OpenAI 夥伴, M1400 加速器', 'growth': '34%'},
    'ALL': {'note': '連續 32 年配息, 高品質保險', 'growth': '193%'}
}

try:
    FRED_API_KEY = st.secrets["FRED_API_KEY"]
    fred = Fred(api_key=FRED_API_KEY)
except Exception:
    st.error("❌ 請在 Secrets 設定 FRED_API_KEY")
    st.stop()

# --- 2. 數據抓取模組 (真實鏈上與市場數據) ---
@st.cache_data(ttl=3600)
def fetch_real_onchain():
    """從 CoinGecko 與 Blockchain.com 抓取真實數據"""
    try:
        # BTC.D
        btc_d = requests.get("https://api.coingecko.com/api/v3/global", timeout=10).json()['data']['market_cap_percentage']['btc']
        # MVRV
        mvrv_data = requests.get("https://api.blockchain.info/charts/mvrv?timespan=1year&format=json", timeout=10).json()
        mvrv = mvrv_data['values'][-1]['y']
    except Exception:
        btc_d, mvrv = 52.5, 2.1  # 異常時顯示預設值
    return btc_d, mvrv

@st.cache_data(ttl=600)
def fetch_market_master(tickers):
    """修復 0050.TW 與 $nan 問題的數據抓取"""
    # 台股代碼自動校正 (處理大小寫與點號)
    processed = [t.upper() if ".TW" in t.upper() else t for t in tickers]
    benchmarks = ['QQQ', '0050.TW', '^VIX', '^MOVE', 'BTC-USD']
    full_list = list(set(processed + benchmarks))
    
    # 抓取 1 年資料確保 20EMA 穩定，使用 auto_adjust 處理台股復權
    data = yf.download(full_list, period="1y", auto_adjust=True, progress=False)
    # ffill 補齊台美股休市的時間差，解決 $nan 關鍵
    prices = data['Close'].ffill()
    
    # 抓取財報日
    earnings = {}
    for t in processed:
        if "-" not in t and ".TW" not in t:
            try:
                cal = yf.Ticker(t).calendar
                if cal is not None and not cal.empty:
                    earnings[t] = cal.loc['Earnings Date'].iloc[0].strftime('%Y-%m-%d')
            except Exception: pass
    return prices, earnings

# --- 3. 側邊欄配置 ---
st.sidebar.header("💰 12.7萬實戰資產配置")
if 'portfolio_df' not in st.session_state:
    st.session_state.portfolio_df = pd.DataFrame([
        {"代號": "MU", "金額": 30000}, {"代號": "AMD", "金額": 25000},
        {"代號": "0050.TW", "金額": 40000}, {"代號": "BTC-USD", "金額": 32000}
    ])
edited_df = st.sidebar.data_editor(st.session_state.portfolio_df, num_rows="dynamic")
user_tickers = edited_df["代號"].tolist()
total_val = edited_df["金額"].sum()

# --- 4. 執行審計與顯示 ---
try:
    prices, earnings_dates = fetch_market_master(user_tickers)
    btc_d, mvrv = fetch_real_onchain()
    liq = (fred.get_series('WALCL').iloc[-1] - fred.get_series('WTREGEN').iloc[-1] - fred.get_series('RRPONTSYD').iloc[-1]) / 1000
    
    # A. 頂部看板：週期審計
    st.subheader("🌡️ 週期與情緒審計 (利好出盡偵測)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MVRV 週期溫度", f"{mvrv:.2f}", delta="利多出盡風險" if mvrv > 3.0 else "安全")
    m2.metric("BTC.D 市佔率", f"{btc_d:.1f}%")
    m3.metric("VIX 天氣", f"{prices['^VIX'].iloc[-1]:.2f}")
    m4.metric("淨流動性", f"${liq:,.2f}B")

    # B. 即時脈搏：網格佈局解決位數遮斷 (image_182fdd)
    st.divider()
    st.subheader("⚡ 即時市場脈搏")
    rows = [user_tickers[i:i + 4] for i in range(0, len(user_tickers), 4)]
    for row in rows:
        cols = st.columns(4)
        for i, t in enumerate(row):
            if t in prices.columns:
                curr_p = prices[t].iloc[-1]
                chg = (prices[t].iloc[-1]/prices[t].iloc[-2]-1)*100
                cols[i].metric(t, f"${curr_p:,.2f}", f"{chg:.2f}%")

    # C. 深度審計表 (修復括號與 $nan 問題)
    st.divider()
    st.subheader("📋 跨市場深度審計與走勢預判")
    audit_data = []
    today = datetime.now().date()
    for t in user_tickers:
        if t not in prices.columns or t in ['^VIX', '^MOVE', 'QQQ']: continue
        
        curr = prices[t].iloc[-1]
        ema20 = prices[t].ewm(span=20).mean().iloc[-1]
        
        # 相對強度預判 (修復括號閉合)
        win_qqq = (prices[t]/prices['QQQ']).iloc[-1] > (prices[t]/prices['QQQ']).rolling(20).mean().iloc[-1]
        win_0050 = (prices[t]/prices['0050.TW']).iloc[-1] > (prices[t]/prices['0050.TW']).rolling(20).mean().iloc[-1] if '0050.TW' in prices.columns else False
        
        # 財報預警
        e_date = earnings_dates.get(t, "N/A")
        e_alert = "⚠️ 7天內" if e_date != "N/A" and (datetime.strptime(e_date, '%Y-%m-%d').date() - today).days <= 7 else "✅"
        
        audit_data.append({
            "標的": t, "SA 觀點": SA_INSIGHTS.get(t, {}).get('note', '-'),
            "20EMA 狀態": "🟢 站穩" if curr > ema20 else "🔴 跌破",
            "勝過 QQQ": "✅" if win_qqq else "❌",
            "勝過 0050": "✅" if win_0050 else "❌",
            "止損位": f"${prices[t].max()*0.93:,.2f}",
            "財報": e_alert
        })
    st.table(pd.DataFrame(audit_data))

    # D. 板塊相關性矩陣
    st.subheader("🤝 板塊集中度 (相關性) 審計")
    st.plotly_chart(px.imshow(prices[user_tickers].corr(), text_auto=".2f", color_continuous_scale='RdBu_r'), use_container_width=True)

    # E. 文字報告 (最底端修正建議)
    st.divider()
    st.subheader("🖋️ Alpha 4.0 會計師修正意見報告")
    with st.container(border=True):
        if mvrv > 3.0: st.error("🚨 **週期警報：** MVRV 指數過熱，這就是你要找的『利好出盡』，建議執行減碼。")
        if prices['^MOVE'].iloc[-1] > 110: st.warning("⚠️ **債市風險：** MOVE 指數過高，美債波動將壓抑科技股溢價。")
        for t in user_tickers:
            if t in prices.columns and prices[t].iloc[-1] < prices[t].ewm(span=20).mean().iloc[-1]:
                st.write(f"🛑 **指令：** {t} 已跌破生命線，反彈皆視為減碼機會。")

except Exception as e:
    st.error(f"系統自動審計中發生異常，請檢查資料連結: {e}")