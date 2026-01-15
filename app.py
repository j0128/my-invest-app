import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from sklearn.linear_model import LinearRegression
from datetime import datetime, date, timedelta

# --- 2026 核心配置 ---
FRED_API_KEY = "你的_FRED_API_KEY_在此" 

# 1. 財報日自動預測邏輯 (2026 Q1 版)
def auto_fetch_earnings(ticker):
    # 2026 年科技股財報預估表
    schedule = {
        'AMD': '2026-01-27', 'NVDA': '2026-02-25', 'TSM': '2026-01-16',
        'QQQ': '2026-01-29', 'AAPL': '2026-01-30', 'MSFT': '2026-01-27'
    }
    return schedule.get(ticker.upper(), "2026-02-15")

# 2. 數據清洗模組 (防禦 MultiIndex 與 KeyError)
def module_integrity(df_raw):
    # 強制扁平化 yfinance 下載的多層索引
    df = df_raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        if 'Adj Close' in df.columns.levels[0]:
            df = df['Adj Close']
        else:
            df.columns = df.columns.get_level_values(0)
    
    df = df.ffill().dropna(how='all')
    if 'QQQ' not in df.columns:
        return None, "❌ 必須包含 QQQ 作為審計基準"
    
    clean_df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return clean_df, None

# 3. 核心運算引擎
def run_strategic_audit_v5(data, investments, exit_date):
    clean, err = module_integrity(data)
    if err: return None, err
    
    # 線性回歸趨勢
    y = clean['QQQ'].values.reshape(-1, 1)
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    k, eff = model.coef_[0][0], model.score(x, y)
    
    # 財報風險自動監控
    target_asset = [a for a in investments.keys() if a != 'QQQ'][0] if len(investments) > 1 else 'QQQ'
    earn_date_str = auto_fetch_earnings(target_asset)
    days_to_earn = (datetime.strptime(earn_date_str, "%Y-%m-%d").date() - date(2026, 1, 15)).days
    risk_tag = "⚠️ 禁區" if days_to_earn <= 7 else "✅ 安全"

    # 資金配比
    total_cap = sum(investments.values()) if sum(investments.values()) > 0 else 1
    
    return {
        "k": k, "eff": eff, "p1": model.predict([[len(y)+22]])[0][0],
        "ts_p": model.predict(x).flatten(), "risk": risk_tag, 
        "earn_date": earn_date_str, "total": total_cap,
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
        user_investments[asset] = st.number_input(f"{asset} 持倉金額 (USD)", min_value=0, value=1000)
    
    exit_in = st.date_input("2026 清倉目標日", value=date(2026, 5, 31))
    # 確認執行鍵
    submit_button = st.form_submit_button("🚀 執行進攻型深度審計")

st.title("🚀 Alpha 2.0 進攻型深度審計 (2026 版)")

if submit_button:
    # 抓取真實數據
    raw_data = yf.download(monitored, start="2024-01-01", end="2026-01-16")
    
    if not raw_data.empty:
        res, err = run_strategic_audit_v5(raw_data, user_investments, exit_in)
        
        if err:
            st.error(err)
        else:
            # 第一排：核心數據
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("進攻斜率 (k)", f"{res['k']:.2f}")
            c2.metric("自動偵測財報日", res['earn_date'], delta=res['risk'])
            c3.metric("1M 預測價 (QQQ)", f"${res['p1']:.2f}")
            c4.metric("總曝險金額", f"${res['total']:,.0f}")
            
            st.divider()
            
            # 第二排：圖表
            col_l, col_r = st.columns(2)
            with col_l:
                st.subheader("📊 持倉權重分配")
                st.bar_chart(pd.DataFrame(res['weights'].items(), columns=['資產', '權重']).set_index('資產'))
            with col_r:
                st.subheader("📈 QQQ 趨勢生命線")
                plot_df = pd.DataFrame({"實際價格": raw_data.xs('Adj Close', axis=1, level=0)['QQQ'][-60:] if isinstance(raw_data.columns, pd.MultiIndex) else raw_data['QQQ'][-60:]})
                plot_df["預測趨勢"] = res['ts_p'][-60:]
                st.line_chart(plot_df)
            
            st.info(f"📍 審計結論