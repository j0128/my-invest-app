import streamlit as st
import yfinance as yf
import pandas as pd
from fredapi import Fred
import plotly.express as px
import time
import random
from datetime import datetime

# --- 1. 系統初始化 (加入幣圈指標) ---
st.set_page_config(page_title="Posa Alpha 3.1 (Global Flow)", layout="wide")
st.title("🌐 Posa Alpha 3.1: 跨市場資金流與幣圈動能審計")

# 核心監控名單
SA_TOP_10 = ['MU', 'CIEN', 'GOLD', 'CLS', 'INCY', 'ALL', 'WLDN', 'AMD', 'COHR', 'ATI']
CRYPTO_SEEDS = ['BTC-USD', 'ETH-USD', 'SOL-USD'] # 幣圈核心
POTENTIAL_SEEDS = ['VRT', 'PLTR', 'NVDA'] # 美股輻射

try:
    FRED_API_KEY = st.secrets["FRED_API_KEY"]
    fred = Fred(api_key=FRED_API_KEY)
except:
    st.error("❌ 請檢查 Secrets 中的 FRED_API_KEY")
    st.stop()

# --- 2. 側邊欄設定 ---
st.sidebar.header("💰 實戰資產配置")
# 這裡你可以手動輸入你的美股或幣圈持倉金額
if 'portfolio_df' not in st.session_state:
    st.session_state.portfolio_df = pd.DataFrame([
        {"代號": "MU", "金額": 30000},
        {"代號": "AMD", "金額": 25000},
        {"代號": "QQQ", "金額": 40000},
        {"代號": "BTC-USD", "金額": 10000} # 加入預設幣圈持倉
    ])
edited_df = st.sidebar.data_editor(st.session_state.portfolio_df, num_rows="dynamic")
user_tickers = edited_df["代號"].tolist()
total_val = edited_df["金額"].sum()

# --- 3. 自適應勝率計算 (修正對幣圈的高波動處理) ---
def get_adaptive_stats(ticker_prices, qqq_prices):
    try:
        ema20 = ticker_prices.ewm(span=20).mean()
        rs = ticker_prices / qqq_prices
        signals = (ticker_prices > ema20) & (rs > rs.rolling(20).mean())
        # 幣圈改看未來 3 天，因為節奏較快
        returns = ticker_prices.shift(-3) / ticker_prices - 1
        valid_rets = returns[signals].dropna()
        if len(valid_rets) < 5: return 0.52, 2.0
        win_p = (valid_rets > 0).mean()
        # 凱利公式中的賠率計算
        avg_w = valid_rets[valid_rets > 0].mean()
        avg_l = abs(valid_rets[valid_rets < 0].mean())
        return win_p, (avg_w / avg_l if avg_l > 0 else 2.0)
    except: return 0.5, 1.5

# --- 4. 數據抓取 ---
@st.cache_data(ttl=3600)
def fetch_global_data(tickers):
    prices, earnings = pd.DataFrame(), {}
    full_list = list(set(tickers + SA_TOP_10 + CRYPTO_SEEDS + POTENTIAL_SEEDS + ['QQQ', '^VIX']))
    for t in full_list:
        time.sleep(random.uniform(0.3, 0.8))
        try:
            tk = yf.Ticker(t)
            df = tk.history(period="1y")
            if not df.empty:
                prices[t] = df['Close']
                if "-" not in t: # 幣圈沒財報，過濾掉
                    cal = tk.calendar
                    if cal is not None and not cal.empty:
                        earnings[t] = cal.loc['Earnings Date'].iloc[0].strftime('%Y-%m-%d')
        except: continue
    try:
        liq = (fred.get_series('WALCL').iloc[-1] - fred.get_series('WTREGEN').iloc[-1] - fred.get_series('RRPONTSYD').iloc[-1]) / 1000
    except: liq = 0
    return liq, prices, earnings

# --- 5. 主介面展示 ---
try:
    net_liq, prices, earnings_dates = fetch_global_data(user_tickers)
    vix = prices['^VIX'].iloc[-1] if '^VIX' in prices.columns else 20
    
    # 頂部指標
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("美元淨流動性", f"${net_liq:.2f}B")
    m2.metric("VIX 天氣", f"{vix:.2f}", delta="警報" if vix > 22 else "安全", delta_color="inverse")
    
    # 幣圈動能偵測
    btc_status = "🟢 強勢" if prices['BTC-USD'].iloc[-1] > prices['BTC-USD'].ewm(span=20).mean().iloc[-1] else "🔴 弱勢"
    m3.metric("BTC 趨勢", btc_status)
    m4.metric("總市值", f"${total_val:,.0f}")

    # 審計表格
    st.subheader("📋 跨市場資產審計 (美股 + 幣圈)")
    audit_list = []
    for t in list(set(user_tickers + CRYPTO_SEEDS)):
        if t not in prices.columns or t in ['^VIX', 'QQQ']: continue
        win_p, odds = get_adaptive_stats(prices[t], prices['QQQ'])
        kelly_f = max(0, (win_p - (1 - win_p) / odds) * 0.5)
        
        amt = edited_df.loc[edited_df['代號']==t, '金額'].sum()
        weight = amt / total_val if total_val > 0 else 0
        
        audit_list.append({
            "標的": t, "類型": "幣圈" if "-" in t else "美股",
            "回測勝率": f"{win_p*100:.1f}%", "凱利建議權重": f"{kelly_f*100:.1f}%",
            "實際權重": f"{weight*100:.1f}%", "狀態": "✅" if prices[t].iloc[-1] > prices[t].ewm(span=20).mean().iloc[-1] else "⚠️"
        })
    st.table(pd.DataFrame(audit_list).sort_values(by="回測勝率", ascending=False))

    # --- 6. 鐵血會計師修正意見 (加入幣圈邏輯) ---
    st.divider()
    st.subheader("🖋️ Alpha 3.1 跨市場審計報告")
    
    reports = []
    # A. 溢出效應判定
    if vix > 20 and prices['BTC-USD'].iloc[-1] > prices['BTC-USD'].ewm(span=20).mean().iloc[-1]:
        reports.append("🚀 **資金流向提示：** 目前美股 VIX 升高，但 BTC 依然站穩均線。確認 **『資金溢出效應』** 發生中，建議將美股避險資金轉往高勝率幣圈標的（如 SOL）。")
    
    # B. 凱利偏離警告
    for t in user_tickers:
        if t in prices.columns:
            win_p, odds = get_adaptive_stats(prices[t], prices['QQQ'])
            kelly_f = (win_p - (1 - win_p) / odds) * 0.5
            actual_w = edited_df.loc[edited_df['代號']==t, '金額'].sum() / total_val
            if actual_w > kelly_f + 0.15:
                reports.append(f"🚨 **配置：** 標的 **{t}** 權重過高，凱利建議為 {kelly_f*100:.1f}%。請縮小倉位以防波動。")

    for r in reports:
        st.write(r)

except Exception as e:
    st.error(f"分析中：{e}")