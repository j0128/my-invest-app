import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from sklearn.linear_model import LinearRegression
from datetime import datetime, date, timedelta

# --- 2026 核心參數配置 ---
FRED_API_KEY = "你的_FRED_API_KEY_在此" 

# 1. 財報日自動預測邏輯 (2026 Q1)
def get_2026_earnings_date(ticker):
    schedule = {
        'AMD': '2026-01-27', 'NVDA': '2026-02-25', 'TSM': '2026-01-16',
        'QQQ': '2026-01-29', 'AAPL': '2026-01-30', 'MSFT': '2026-01-27'
    }
    return schedule.get(ticker.upper(), "2026-02-15")

# 2. 數據清洗模組 (解決 KeyError 與 MultiIndex 結構)
def module_integrity(df_raw):
    df = df_raw.copy()
    # 處理 yfinance 多標的下載產生的多層索引
    if isinstance(df.columns, pd.MultiIndex):
        if 'Adj Close' in df.columns.levels[0]:
            df = df['Adj Close']
        else:
            df.columns = df.columns.get_level_values(-1)
    
    df = df.ffill().dropna(how='all')
    if 'QQQ' not in df.columns:
        return None, "❌ 錯誤：請務必在左側監控資產中勾選 QQQ 作為基準。"
    
    clean_df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return clean_df, None

# 3. 進攻型審計核心引擎
def run_strategic_audit_v5(data, investments, exit_date_obj):
    clean, err = module_integrity(data)
    if err: return None, err
    
    # 核心回歸運算
    y = clean['QQQ'].values.reshape(-1, 1)
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    k, eff = model.coef_[0][0], model.score(x, y)
    ts_p = model.predict(x).flatten()
    
    # 財報風險自動監控
    target_ticker = [a for a in investments.keys() if a != 'QQQ'][0] if len(investments) > 1 else 'QQQ'
    earn_date_str = get_2026_earnings_date(target_ticker)
    
    # 資金配比與等級計算 (解決 ValueError: identically-labeled Series)
    total_cap = sum(investments.values()) if sum(investments.values()) > 0 else 1
    weights = {k: v/total_cap for k, v in investments.items()}
    
    # 提取純標量數值進行比較，避免 Series 對齊錯誤
    eps = 1e-12
    rets_series = clean.pct_change().dropna().sum()
    
    val_target = float(rets_series.get(target_ticker, 0))
    val_qld = float(rets_series.get('QLD', eps))
    val_tqqq = float(rets_series.get('TQQQ', eps))
    
    # 級別判定
    if val_target > val_tqqq:
        grade = "Alpha+"
    elif val_target > val_qld:
        grade = "Beta+"
    else:
        grade = "Underperform"

    return {
        "k": k, "eff": eff, "p1": model.predict([[len(y)+22]])[0][0],
        "ts_p": ts_p, "earn_date": earn_date_str, 
        "total": total_cap, "weights": weights, "grade": grade
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
    # 數據抓取
    raw_data = yf.download(monitored, start="2024-01-01", end="2026-01-16")
    
    if not raw_data.empty:
        res, err = run_strategic_audit_v5(raw_data, user_investments, exit_in)
        
        if err:
            st.error(err)
        else:
            # 數據看板
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("進攻斜率 (k)", f"{res['k']:.2f}")
            c2.metric("自動偵測財報日", f"{res['earn_date']}")
            c3.metric("1M 預估價 (QQQ)", f"${res['p1']:.2f}")
            c4.metric("總資產價值", f"${res['total']:,.0f}")
            
            st.divider()
            
            col_l, col_r = st.columns(2)
            with col_l:
                st.subheader("📊 持倉比重分析")
                w_df = pd.DataFrame(res['weights'].items(), columns=['資產', '權重']).set_index('資產')
                st.bar_chart(w_df)
                st.info(f"當前選股等級：**{res['grade']}**")
            with col_r: