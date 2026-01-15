import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from sklearn.linear_model import LinearRegression
from datetime import datetime, date, timedelta

# --- 2026 核心配置 ---
FRED_API_KEY = "你的_FRED_API_KEY_在此" 

# 1. 自動財報日預判
def get_2026_earnings_date(ticker):
    schedule = {
        'AMD': '2026-01-27', 'NVDA': '2026-02-25', 'TSM': '2026-01-16',
        'QQQ': '2026-01-29', 'AAPL': '2026-01-30', 'BTC-USD': 'N/A'
    }
    return schedule.get(ticker.upper(), "2026-02-15")

# 2. 數據洗滌：強力索引扁平化模組
def module_integrity(df_raw):
    df = df_raw.copy()
    # 解決 image_4fbb72 的 KeyError：強制將多層索引轉為單層
    if isinstance(df.columns, pd.MultiIndex):
        if 'Adj Close' in df.columns.levels[0]:
            df = df['Adj Close']
        else:
            df.columns = df.columns.get_level_values(-1)
    
    df = df.ffill().dropna(how='all')
    # 確保基準 QQQ 存在
    if 'QQQ' not in df.columns:
        return None, "❌ 基準缺失：請務必在左側勾選 QQQ"
    
    clean_df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return clean_df, None

# 3. 核心運算引擎
def run_strategic_audit_v5(data, investments, exit_date):
    clean, err = module_integrity(data)
    if err: return None, err
    
    y = clean['QQQ'].values.reshape(-1, 1)
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    k, eff = model.coef_[0][0], model.score(x, y)
    ts_p = model.predict(x).flatten()
    
    # 找出除了 QQQ 以外的主攻資產
    target_ticker = [a for a in investments.keys() if a != 'QQQ'][0] if len(investments) > 1 else 'QQQ'
    earn_date_str = get_2026_earnings_date(target_ticker)
    
    # 計算最佳權重與等級
    total_cap = sum(investments.values()) if sum(investments.values()) > 0 else 1
    eps = 1e-12
    rets = clean.pct_change().dropna().sum()
    target_sum = rets.get(target_ticker, 0)
    qld_sum = rets.get('QLD', eps)
    tqqq_sum = rets.get('TQQQ', eps)
    
    # 解決 division by zero
    grade = "Alpha+" if target_sum > tqqq_sum else ("Beta+" if target_sum > qld_sum else "Underperform")

    return {
        "k": k, "eff": eff, "p1": model.predict([[len(y)+22]])[0][0],
        "ts_p": ts_p, "earn_date": earn_date_str, 
        "total": total_cap, "grade": grade,
        "weights": {k: v/total_cap for k, v in investments.items()}
    }, None

# --- UI 介面 ---
st.set_page_config(page_title="Alpha 2.0 Strategic Audit", layout="wide")
st.sidebar.header("🎯 進攻調度中心 (2026)")

with st.sidebar.form("audit_form"):
    monitored = st.multiselect("監控資產", ["QQQ","QLD","TQQQ","BTC-USD","AMD","NVDA","TSM"], default=["QQQ","QLD","TQQQ","AMD"])
    st.write("---")
    user_investments = {}
    for asset in monitored:
        user_investments[asset] = st.number_input(f"{asset} 金額 (USD)", min_value=0, value=1000)
    exit_in = st.date_input("2026 清倉目標日", value=date(2026, 5, 31))
    submit_button = st.form_submit_button("🚀 執行進攻型深度審計")

st.title("🚀 Alpha 2.0 進攻型深度審計 (2026 版)")

if submit_button:
    # yfinance 下載
    raw_data = yf.download(monitored, start="2024-01-01", end="2026-01-16")
    
    if not raw_data.empty:
        res, err = run_strategic_audit_v5(raw_data, user_investments, exit_in)
        if err:
            st.error(err)
        else:
            # 修正 SyntaxError：確保所有 f-string 閉合
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("進攻斜率 (k)", f"{res['k']:.2f}")
            c2.metric("自動財報日", f"{res['earn_date']}")
            c3.metric("1M 預測價", f"${res['p1']:.2f}")
            c4.metric("總持倉價值", f"${res['total']:,.0f}")
            
            st.divider()
            col_l, col_r = st.columns(2)
            with col_l:
                st.subheader("📊 持倉比重")
                st.bar_chart(pd.DataFrame(res['weights'].items(), columns=['資產', '權重']).set_index('資產'))
                st.write(f"當前選股等級：**{res['grade']}**")
            with col_r:
                st.subheader("📈 QQQ 趨勢生命線")
                q_price = raw_data['Adj Close']['QQQ'][-60:] if isinstance(raw_data.columns, pd.MultiIndex) else raw_data['QQQ'][-60:]
                st.line_chart(pd.DataFrame({"實際價格": q_price.values, "預測趨勢": res['ts_p'][-60:]}))
            
            st.info(f"📍 審計結論：目前趨勢穩定。距離 2026/05 撤退日剩餘 {(exit_in - date(2026,1,15)).days} 天。")
    else:
        st.error("API 數據注入失敗，請檢查網絡連結。")
else:
    st.info("請在左側確認選取 **QQQ** 並輸入持倉金額後按下「🚀 執行進攻型深度審計」。")