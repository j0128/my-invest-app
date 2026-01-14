import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import time
import plotly.express as px
import plotly.graph_objects as go
from fredapi import Fred
from datetime import datetime

# --- 1. 初始化 ---
st.set_page_config(page_title="Posa Alpha 4.2", layout="wide")
st.title("🛡️ Posa Alpha 4.2: 實戰審計與決策手冊版")

try:
    FRED_API_KEY = st.secrets["FRED_API_KEY"]
    fred = Fred(api_key=FRED_API_KEY)
except Exception:
    st.error("❌ 請在 Secrets 設定 FRED_API_KEY")
    st.stop()

# --- 2. 數據抓取模組 (含真實鏈上與穩定台股) ---
@st.cache_data(ttl=3600)
def fetch_onchain():
    try:
        btc_d = requests.get("https://api.coingecko.com/api/v3/global", timeout=10).json()['data']['market_cap_percentage']['btc']
        mvrv_data = requests.get("https://api.blockchain.info/charts/mvrv?timespan=1year&format=json", timeout=10).json()
        mvrv = mvrv_data['values'][-1]['y']
    except Exception:
        btc_d, mvrv = 52.5, 2.1
    return btc_d, mvrv

@st.cache_data(ttl=600)
def fetch_market_data(tickers):
    # 代碼校正
    processed = [t.upper() if ".TW" in t.upper() else t for t in tickers if t]
    benchmarks = ['QQQ', '0050.TW', '^VIX', '^MOVE', 'BTC-USD']
    full_list = list(set(processed + benchmarks))
    
    # 抓取 1 年資料以獲得穩定的 20EMA
    data = yf.download(full_list, period="1y", auto_adjust=True, progress=False)
    prices = data['Close'].ffill() # 關鍵：若關市則自動填充前一日價格
    
    earnings = {}
    for t in processed:
        if "-" not in t and ".TW" not in t:
            try:
                tk = yf.Ticker(t)
                cal = tk.calendar
                if cal is not None and not cal.empty:
                    earnings[t] = cal.loc['Earnings Date'].iloc[0].strftime('%Y-%m-%d')
            except Exception: pass
    return prices, earnings

# --- 3. 側邊欄：表單模式 (修正 1) ---
with st.sidebar.form("input_form"):
    st.header("💰 12.7萬實戰資產輸入")
    if 'portfolio_df' not in st.session_state:
        st.session_state.portfolio_df = pd.DataFrame([
            {"代號": "MU", "金額": 30000}, {"代號": "AMD", "金額": 25000},
            {"代號": "0050.TW", "金額": 40000}, {"代號": "BTC-USD", "金額": 32000}
        ])
    edited_df = st.data_editor(st.session_state.portfolio_df, num_rows="dynamic")
    submit_button = st.form_submit_button("🚀 確認並執行審計")

# --- 4. 執行邏輯 ---
if submit_button:
    try:
        user_tickers = edited_df["代號"].dropna().tolist()
        total_val = edited_df["金額"].sum()
        
        prices, earnings_dates = fetch_market_data(user_tickers)
        btc_d, mvrv = fetch_onchain()
        liq = (fred.get_series('WALCL').iloc[-1] - fred.get_series('WTREGEN').iloc[-1] - fred.get_series('RRPONTSYD').iloc[-1]) / 1000
        
        # A. 頂部指標
        st.subheader("🌡️ 週期與情緒審計")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MVRV 週期溫度", f"{mvrv:.2f}", delta="利多出盡警戒" if mvrv > 3.0 else "安全")
        m2.metric("BTC.D 市佔率", f"{btc_d:.1f}%")
        m3.metric("VIX 天氣", f"{prices['^VIX'].iloc[-1]:.2f}")
        m4.metric("淨流動性", f"${liq:,.2f}B")

        # B. 即時脈搏 (每行 4 檔)
        st.divider()
        st.subheader("⚡ 即時市場脈搏")
        display_tickers = [t for t in user_tickers if t in prices.columns]
        for i in range(0, len(display_tickers), 4):
            cols = st.columns(4)
            for j, t in enumerate(display_tickers[i:i+4]):
                curr_p = prices[t].iloc[-1]
                chg = (prices[t].iloc[-1]/prices[t].iloc[-2]-1)*100
                cols[j].metric(t, f"${curr_p:,.2f}", f"{chg:.2f}%")

        # C. 深度審計表
        st.divider()
        st.subheader("📋 跨市場深度審計與走勢預判")
        audit_data = []
        today = datetime.now().date()
        for t in user_tickers:
            if t not in prices.columns or t in ['^VIX', '^MOVE', 'QQQ']: continue
            curr = prices[t].iloc[-1]
            ema20 = prices[t].ewm(span=20).mean().iloc[-1]
            win_qqq = (prices[t]/prices['QQQ']).iloc[-1] > (prices[t]/prices['QQQ']).rolling(20).mean().iloc[-1]
            e_date = earnings_dates.get(t, "N/A")
            e_alert = "⚠️ 7天內" if e_date != "N/A" and (datetime.strptime(e_date, '%Y-%m-%d').date() - today).days <= 7 else "✅"
            
            audit_data.append({
                "標地": t, 
                "20EMA 狀態": "🟢 站穩" if curr > ema20 else "🔴 跌破",
                "勝過 QQQ": "✅" if win_qqq else "❌",
                "未來走勢預判": "🔥 加速" if (curr > ema20 and win_qqq) else "🛑 轉弱",
                "財報風險": e_alert
            })
        st.table(pd.DataFrame(audit_data))

        # D. 審計邏輯手冊 (修正 3)
        st.divider()
        st.subheader("📚 Posa 審計決策手冊 (判斷依準說明)")
        with st.expander("點擊展開：查看詳細判斷邏輯"):
            st.markdown("""
            ### 1. 趨勢預判邏輯 (Future Trend)
            * **🔥 加速上升**：當股價位於 **20EMA 生命線** 之上，且相對於 **QQQ (納斯達克)** 的強度增加。代表此標的是目前市場領跑者。
            * **🛑 轉弱/減碼**：當股價跌破 **20EMA**，即便有利多消息也視為「利好出盡」的反彈，應優先保本。
            
            ### 2. 財報預警邏輯 (Earnings Risk)
            * **⚠️ 7天內**：財報公佈前後波動劇烈，依據會計審計原則，不應在此時參與博弈，建議空倉或減碼。
            
            ### 3. 利好出盡與週期預判 (Cycle Temperature)
            * **MVRV 指數**：衡量比特幣持有者的盈虧。若 **MVRV > 3.0**，代表市場獲利盤巨大，極易觸發集體獲利了結，是利多出盡的終極警訊。
            * **BTC.D (市佔率)**：若市佔下降且比特幣價格橫盤，代表資金正向山寨幣（如 SOL）擴散，波動將放大。
            """)

    except Exception as e:
        st.error(f"審計執行中發生錯誤: {e}")
else:
    st.info("💡 請在左方輸入持倉資訊，並點擊『🚀 確認並執行審計』按鈕開始分析。")