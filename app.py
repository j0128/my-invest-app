import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from sklearn.linear_model import LinearRegression
from datetime import datetime, date, timedelta

# --- 2026 核心參數配置 ---
FRED_API_KEY = "你的_FRED_API_KEY_在此" 

# 1. 財報日自動預測邏輯 (2026 Q1 模擬數據)
def get_2026_earnings_date(ticker):
    schedule = {
        'AMD': '2026-01-27', 'NVDA': '2026-02-25', 'TSM': '2026-01-16',
        'QQQ': '2026-01-29', 'AAPL': '2026-01-30', 'MSFT': '2026-01-27',
        'BTC-USD': 'N/A'
    }
    return schedule.get(ticker.upper(), "2026-02-15")

# 2. 數據清洗模組 (解決 KeyError 與 MultiIndex 結構)
def module_integrity(df_raw):
    df = df_raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        if 'Adj Close' in df.columns.levels[0]:
            df = df['Adj Close']
        else:
            df.columns = df.columns.get_level_values(0)
    
    df = df.ffill().dropna(how='all')
    if 'QQQ' not in df.columns:
        return None, "❌ 錯誤：QQQ 為量化基準，請務必在左側勾選。"
    
    clean_df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return clean_df, None

# 3. 進攻型審計核心引擎
def run_strategic_audit_v5(data, investments, exit_date_obj):
    clean, err = module_integrity(data)
    if err: return None, err
    
    # 線性回歸趨勢 [1, 2, 3, 4, 5]
    y = clean['QQQ'].values.reshape(-1, 1)
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    k, eff = model.coef_[0][0], model.score(x, y)
    ts_p = model.predict(x).flatten()
    
    # 財報風險自動監控 [B]
    target_ticker = [a for a in investments.keys() if a != 'QQQ'][0] if len(investments) > 1 else 'QQQ'
    earn_date_str = get_2026_earnings_date(target_ticker)
    
    days_to_earn = 999
    risk_tag = "SAFE"
    if earn_date_str != 'N/A':
        days_to_earn = (datetime.strptime(earn_date_str, "%Y-%m-%d").date() - date(2026, 1, 15)).days
        risk_tag = "⚠️ 禁區" if days_to_earn <= 7 else "✅ 安全"

    # 資金配比與 Kelly [13, 21]
    total_cap = sum(investments.values()) if sum(investments.values()) > 0 else 1
    weights = {k: v/total_cap for k, v in investments.items()}
    
    # 選股等級對標 (解決 division by zero)
    eps = 1e-12
    rets = clean.pct_change().dropna().sum()
    qld_sum = rets.get('QLD', eps)
    tqqq_sum = rets.get('TQQQ', eps)
    target_sum = rets.get(target_ticker, 0)
    
    grade = "Alpha+" if target_sum > (tqqq_sum if abs(tqqq_sum) > eps else eps) else "Underperform"
    if target_sum > (qld_sum if abs(qld_sum) > eps else eps) and grade == "Underperform":
        grade = "Beta+"

    return {
        "k": k, "eff": eff, "p1": model.predict([[len(y)+22]])[0][0],
        "ts_p": ts_p, "risk": risk_tag, "earn_date": earn_date_str, 
        "total": total_cap, "weights": weights, "grade": grade
    }, None

# --- UI 介面 ---
st.set_page_config(page_title="Alpha 2.0 Strategic Audit", layout="wide")
st.sidebar.header("🎯 進攻調度中心 (2026)")

# 使用 Sidebar Form 建立確認機制
with st.sidebar.form("audit_form"):
    monitored = st.multiselect("監控資產", ["QQQ","QLD","TQQQ","BTC-USD","AMD","NVDA","TSM"], default=["QQQ","QLD","TQQQ","AMD"])
    
    st.write("---")
    st.write("💰 輸入各標的持倉金額 (USD)")
    user_investments = {}
    for asset in monitored:
        user_investments[asset] = st.number_input(f"{asset} 金額", min_value=0, value=1000)
    
    exit_in = st.date_input("2026 獲利清倉日", value=date(2026, 5, 31))
    submit_button = st.form_submit_button("🚀 執行進攻型深度審計")

st.title("🚀 Alpha 2.0 進攻型深度審計 (2026 版)")

if submit_button:
    # 下載數據
    raw_data = yf.download(monitored, start="2024-01-01", end="2026-01-16")
    
    if not raw_data.empty:
        res, err = run_strategic_audit_v5(raw_data, user_investments, exit_in)
        
        if err:
            st.error(err)
        else:
            # 第一排：核心數據
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("進攻斜率 (k)", f"{res['k']:.2f}")
            c2.metric("自動偵測財報日", f"{res['earn_date']}", delta=res['risk'])
            c3.metric("1M 預測價 (QQQ)", f"${res['p1']:.2f}")
            c4.metric("總持倉價值", f"${res['total']:,.0f}")
            
            st.divider()
            
            # 第二排：圖表
            col_l, col_r = st.columns(2)
            with col_l:
                st.subheader("📊 持倉比重分配")
                weight_df = pd.DataFrame(res['weights'].items(), columns=['資產', '權重']).set_index('資產')
                st.bar_chart(weight_df)
                st.write(f"當前選股等級：**{res['grade']}**")
            with col_r:
                st.subheader("📈 QQQ 趨勢生命線")
                # 解決 MultiIndex 下的圖表繪製
                q_price = raw_data['Adj Close']['QQQ'][-60:] if isinstance(raw_data.columns, pd.MultiIndex) else raw_data['QQQ'][-60:]
                plot_df = pd.DataFrame({"實際價格": q_price.values, "預測趨勢": res['ts_p'][-60:]})
                st.line_chart(plot_df)
            
            st.info(f"🚩 審計結論：當前模型預測 QQQ 維持進攻。距離 2026/05 撤退日剩餘 {(exit_in - date(2026,1,15)).days} 天。")
    else:
        st.error("數據抓取失敗，請檢查網路。")
else:
    st.info("請在左側輸入持倉金額並按下「🚀 執行進攻型深度審計」。")