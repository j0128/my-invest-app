import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from sklearn.linear_model import LinearRegression
from datetime import datetime, date, timedelta

# --- 2026 核心參數 ---
FRED_API_KEY = "你的_FRED_API_KEY_在此" 

# 1. 財報日自動查詢模組 (2026 Q1 模擬邏輯)
def get_auto_earnings_date(ticker):
    """
    根據 2026 年 1 月時間點，自動預判/抓取下一季財報日
    (實戰中可對接 yfinance 的 info 或專門財報 API)
    """
    # 2026 Q1 主要科技股財報預計日期表
    earnings_map_2026 = {
        'AMD': '2026-01-27', 'NVDA': '2026-02-25', 'TSM': '2026-01-16',
        'QQQ': '2026-01-29', 'TQQQ': '2026-01-29', 'AAPL': '2026-01-30'
    }
    return earnings_map_2026.get(ticker.upper(), "2026-02-15")

# 2. 數據洗滌模組 (處理 QQQ 缺失與 MultiIndex)
def module_integrity(df_raw):
    if isinstance(df_raw.columns, pd.MultiIndex):
        df = df_raw['Adj Close'].copy() if 'Adj Close' in df_raw.columns.levels[0] else df_raw.copy()
    else:
        df = df_raw.copy()
    
    df = df.ffill().dropna(how='all')
    
    # 強制要求 QQQ 作為基準
    if 'QQQ' not in df.columns:
        return None, "❌ 錯誤：QQQ 是審計基準，請在左側監控資產中勾選 QQQ。"
    
    clean_df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return clean_df, None

# 3. 進攻型主引擎
def run_strategic_audit_v5(data, user_investments, exit_date_obj):
    clean, err = module_integrity(data)
    if err: return None, err
    
    # 核心回歸
    y = clean['QQQ'].values.reshape(-1, 1)
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    k, eff = model.coef_[0][0], model.score(x, y)
    
    # 最佳持倉與 Kelly
    total_cap = sum(user_investments.values()) if sum(user_investments.values()) > 0 else 1
    current_weights = {k: v / total_cap for k, v in user_investments.items()}
    
    # 自動抓取選定資產中「最近」的財報風險
    first_asset = [a for a in user_investments.keys() if a != 'QQQ'][0] if len(user_investments) > 1 else 'QQQ'
    auto_earn = get_auto_earnings_date(first_asset)
    
    days_to_earn = (datetime.strptime(auto_earn, "%Y-%m-%d").date() - date(2026, 1, 15)).days
    risk_level = "⚠️ 禁區" if days_to_earn <= 7 else "✅ 安全"
    
    return {
        "k": k, "eff": eff, "p1": model.predict([[len(y)+22]])[0][0],
        "weights": current_weights, "risk": risk_level, "earn_date": auto_earn,
        "ts_p": model.predict(x).flatten(), "total": total_cap
    }, None

# --- UI 介面 ---
st.set_page_config(page_title="Alpha 2.0 Strategic Audit", layout="wide")
st.sidebar.header("🎯 進攻調度中心 (2026)")

with st.sidebar.form("audit_form"):
    # 預設必須包含 QQQ
    monitored = st.multiselect("監控資產", ["QQQ","QLD","TQQQ","BTC-USD","AMD","NVDA","TSM"], default=["QQQ","QLD","TQQQ","AMD"])
    
    st.write("---")
    user_investments = {}
    for asset in monitored:
        user_investments[asset] = st.number_input(f"{asset} 持倉 (USD)", min_value=0, value=1000)
    
    exit_in = st.date_input("2026 清倉目標日", value=date(2026, 5, 31))
    submit_button = st.form_submit_button("🚀 執行進攻型審計")

st.title("🚀 Alpha 2.0 進攻型深度審計 (2026 版)")

if submit_button:
    # yfinance 下載
    raw_data = yf.download(monitored, start="2024-01-01", end="2026-01-16")
    
    res, err = run_strategic_audit_v5(raw_data, user_investments, exit_in)
    
    if err:
        st.error(err)
    else:
        # 第一排：核心看板
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("進攻斜率 (k)", f"{res['k']:.2f}")
        c2.metric("