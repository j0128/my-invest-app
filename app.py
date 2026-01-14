import streamlit as st
import yfinance as yf
import pandas as pd
from fredapi import Fred
import plotly.express as px
import plotly.graph_objects as go
import time
import random
from datetime import datetime, timedelta

# --- 1. 系統初始化 ---
st.set_page_config(page_title="Posa Alpha 2.8 (Ultimate Audit)", layout="wide")
st.title("🛡️ Posa x biibo Alpha 2.8 終極審計與自適應配置系統")

# 2026 金股清單
TOP_10_2026 = ['MU', 'AMD', 'CLS', 'COHR', 'CIEN', 'WLDN', 'ATI', 'GOLD', 'ALL', 'INCY']

try:
    FRED_API_KEY = st.secrets["FRED_API_KEY"]
    fred = Fred(api_key=FRED_API_KEY)
except:
    st.error("❌ Secrets 設定錯誤，請確認 FRED_API_KEY。")
    st.stop()

# --- 2. 側邊欄：資產與風險設定 ---
st.sidebar.header("💰 實戰資產配置")
init_data = [
    {"代號": "MU", "金額": 30000},
    {"代號": "AMD", "金額": 25000},
    {"代號": "QQQ", "金額": 40000},
    {"代號": "TLT", "金額": 32000}
]
edited_df = st.sidebar.data_editor(pd.DataFrame(init_data), num_rows="dynamic")
user_tickers = edited_df["代號"].tolist()
total_val = edited_df["金額"].sum()

TRAILING_PCT = st.sidebar.slider("移動止損 (%)", 5, 15, 7) / 100
KELLY_SCALE = st.sidebar.slider("凱利係數 (建議 0.5)", 0.1, 1.0, 0.5)

# --- 3. 核心函數：自適應勝率回測 ---
def get_adaptive_stats(ticker_prices, qqq_prices):
    """計算過去一年，在 biibo 分數滿分時買入，5天後的勝率"""
    try:
        ema20 = ticker_prices.ewm(span=20).mean()
        rs = ticker_prices / qqq_prices
        rs_ema = rs.rolling(20).mean()
        # 信號：價 > EMA20 且 RS > 均值 (biibo 強勢區)
        signals = (ticker_prices > ema20) & (rs > rs_ema)
        returns = ticker_prices.shift(-5) / ticker_prices - 1
        valid_returns = returns[signals].dropna()
        if len(valid_returns) < 5: return 0.52 # 樣本太少給保守值
        win_rate = (valid_returns > 0).mean()
        avg_win = valid_returns[valid_returns > 0].mean() if any(valid_returns > 0) else 0.05
        avg_loss = abs(valid_returns[valid_returns < 0].mean()) if any(valid_returns < 0) else 0.05
        odds = avg_win / avg_loss if avg_loss != 0 else 2.0
        return win_rate, odds
    except:
        return 0.50, 1.0

# --- 4. 數據抓取 (防封鎖版) ---
@st.cache_data(ttl=3600)
def fetch_all_audit_data(tickers):
    prices, volumes, earnings_info = pd.DataFrame(), pd.DataFrame(), {}
    all_symbols = list(set(tickers + ['QQQ', '^VIX', '^MOVE'] + TOP_10_2026))
    
    for t in all_symbols:
        time.sleep(random.uniform(0.5, 1.0)) # 隨機延遲防 Ban
        try:
            tk = yf.Ticker(t)
            df = tk.history(period="1y")
            if not df.empty:
                # 處理 yfinance 多索引問題
                prices[t] = df['Close']
                volumes[t] = df['Volume']
                cal = tk.calendar
                if cal is not None and not cal.empty:
                    earnings_info[t] = cal.loc['Earnings Date'].iloc[0].strftime('%Y-%m-%d')
        except: continue
        
    try:
        net_liq = (fred.get_series('WALCL').iloc[-1] - fred.get_series('WTREGEN').iloc[-1] - fred.get_series('RRPONTSYD').iloc[-1]) / 1000
    except: net_liq = 0
    return net_liq, prices, volumes, earnings_info

# --- 5. 介面與審計邏輯 ---
try:
    net_liq, prices, volumes, earnings_dates = fetch_all_audit_data(user_tickers)
    vix = prices['^VIX'].iloc[-1] if '^VIX' in prices.columns else 20
    
    # 看板
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("淨流動性", f"${net_liq:.2f}B")
    m2.metric("VIX", f"{vix:.2f}", delta="危險" if vix > 22 else "安全", delta_color="inverse")
    m3.metric("組合總值", f"${total_val:,.0f}")
    m4.metric("QQQ 狀態", "🟢 站穩" if prices['QQQ'].iloc[-1] > prices['QQQ'].ewm(span=20).mean().iloc[-1] else "🔴 跌破")

    # 凱利配置與風險審計表
    st.subheader("📋 終極配置審計與凱利建議")
    audit_list = []
    today = datetime.now().date()
    
    for t in user_tickers:
        if t not in prices.columns or t in ['^VIX', '^MOVE', 'QQQ']: continue
        
        # 自適應凱利計算
        win_rate, odds = get_adaptive_stats(prices[t], prices['QQQ'])
        kelly_f = (win_rate - (1 - win_rate) / odds) * KELLY_SCALE
        
        # 財報預警
        e_date_str = earnings_dates.get(t, "未知")
        e_alert = "⚠️ 7天內" if e_date_str != "未知" and (datetime.strptime(e_date_str, '%Y-%m-%d').date() - today).days <= 7 else "✅ 安全"
        
        # 移動止損
        stop_p = prices[t].max() * (1 - TRAILING_PCT)
        curr_p = prices[t].iloc[-1]
        
        # 實際權重
        amt = edited_df.loc[edited_df['代號']==t, '金額'].values[0]
        actual_w = amt / total_val if total_val > 0 else 0
        
        audit_list.append({
            "標的": t,
            "回測勝率": f"{win_rate*100:.1f}%",
            "凱利建議權重": f"{max(0, kelly_f*100):.1f}%",
            "實際權重": f"{actual_w*100:.1f}%",
            "財報風險": e_alert,
            "狀態": "🟢 持有" if curr_p > stop_p else "❌ 止損",
            "現價": f"${curr_p:.2f}"
        })
    
    st.table(pd.DataFrame(audit_list))

    # 集中度矩陣
    st.divider()
    st.subheader("🤝 板塊集中度 (相關性) 審計")
    st.plotly_chart(px.imshow(prices[user_tickers].corr(), text_auto=True, color_continuous_scale='RdBu_r'), use_container_width=True)

except Exception as e:
    st.error(f"系統運行中：{e}")