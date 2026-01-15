import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from sklearn.linear_model import LinearRegression
from datetime import datetime, date, timedelta

# --- 2026 核心參數 ---
FRED_API_KEY = "你的_FRED_API_KEY_在此" 

@st.cache_resource
def get_fred_client(api_key):
    try:
        if "你的" in api_key: return None
        return Fred(api_key=api_key)
    except: return None

fred_client = get_fred_client(FRED_API_KEY)

# 1. 數據洗滌模組
def module_integrity(df_raw):
    if isinstance(df_raw.columns, pd.MultiIndex):
        df = df_raw['Adj Close'].copy() if 'Adj Close' in df_raw.columns.levels[0] else df_raw.copy()
    else:
        df = df_raw.copy()
    df = df.ffill().dropna(how='all')
    if 'QQQ' not in df.columns:
        return None, "請選取 QQQ 作為基準"
    clean_df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return clean_df, None

# 2. 趨勢與預測模組
def module_projection(df):
    y = df['QQQ'].values.reshape(-1, 1)
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    k, eff = model.coef_[0][0], model.score(x, y)
    ts_p = model.predict(x).flatten()
    p1 = model.predict([[len(y) + 22]])[0][0]
    ema20 = df['QQQ'].ewm(span=20).mean().iloc[-1]
    status = "🔥 加速上升" if y[-1][0] > ema20 and k > 0 else "🛡️ 區間盤整"
    return {"k": k, "eff": eff, "p1": p1, "ts_p": ts_p, "status": status, "ema20_val": ema20}

# 3. 資金配比與 Kelly 審計
def module_portfolio_optimization(df, core, investment_dict):
    rets = df.pct_change().dropna()
    eps = 1e-12
    total_capital = sum(investment_dict.values()) if sum(investment_dict.values()) > 0 else 1
    
    # 計算當前持倉權重
    current_weights = {k: v / total_capital for k, v in investment_dict.items()}
    
    # Kelly 建議 (2026 修正)
    win_rate = 0.6 if core['k'] > 0 else 0.4
    kelly_suggested = np.clip((win_rate - (1 - win_rate)) / 1, 0, 0.75)
    
    # 對標 QLD/TQQQ 效率
    target = [c for c in df.columns if c not in ['QQQ', 'QLD', 'TQQQ']][0] if len(df.columns) > 3 else 'QQQ'
    qld_ret = rets['QLD'].sum() if 'QLD' in rets.columns else eps
    target_ret = rets[target].sum() if target in rets.columns else 0
    alpha_grade = "Alpha+" if target_ret > (rets['TQQQ'].sum() if 'TQQQ' in rets.columns else eps) else "Underperform"
    
    std = np.std(df['QQQ'].values - core['ts_p'].reshape(-1, 1))
    shells = {f'L{i}': core['p1'] - i*std for i in range(1, 4)}
    return {"weights": current_weights, "kelly": kelly_suggested, "grade": alpha_grade, "shells": shells, "total": total_capital}

# --- UI 介面 ---
st.set_page_config(page_title="Alpha 2.0 Strategic Audit", layout="wide")
st.sidebar.header("🎯 進攻調度中心 (2026)")

# 使用 Form 建立確認機制
with st.sidebar.form("audit_form"):
    monitored = st.multiselect("監控資產", ["QQQ","QLD","TQQQ","BTC-USD","AMD","TSM","NVDA"], default=["QQQ","QLD","TQQQ","AMD"])
    
    st.write("---")
    st.write("💰 輸入各標的持倉金額 (USD)")
    user_investments = {}
    for asset in monitored:
        user_investments[asset] = st.number_input(f"{asset} 金額", min_value=0, value=1000, step=100)
    
    earn_in = st.date_input("財報日", value=date(2026, 1, 28))
    exit_in = st.date_input("2026 清倉日", value=date(2026, 5, 31))
    
    submit_button = st.form_submit_button("🚀 執行進攻型審計")

@st.cache_data(ttl=3600)
def fetch_data(tickers):
    return yf.download(tickers, start="2024-01-01", end="2026-01-16")

st.title("🚀 Alpha 2.0 進攻型深度審計 (2026 版)")

if submit_button:
    raw_data = fetch_data(monitored)
    if not raw_data.empty:
        clean, err = module_integrity(raw_data)
        if not err:
            core = module_projection(clean)
            port = module_portfolio_optimization(clean, core, user_investments)
            
            # 展示結果
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("進攻斜率 (k)", f"{core['k']:.2f}", delta=core['status'])
            c2.metric("Kelly 建議上限", f"{port['kelly']:.1%}")
            c3.metric("選股等級", port['grade'])
            c4.metric("總資產 (USD)", f"${port['total']:,.0f}")
            
            st.divider()
            
            # 持倉比重分析圖 
            st.subheader("📊 當前持倉比重分析")
            weight_df = pd.DataFrame(port['weights'].items(), columns=['資產', '權重'])
            st.bar_chart(weight_df.set_index('資產'))
            
            st.subheader("📈 20EMA 趨勢生命線")
            st.line_chart(pd.DataFrame({"實際 QQQ": clean['QQQ'][-60:], "趨勢線": core['ts_p'][-60:]}))
            
            st.info(f"📍 審計結論：目前 {list(port['weights'].keys())[0]} 權重最高。距離 2026/05 撤退日剩餘 { (exit_in - date(2026,1,15)).days } 天。")
        else:
            st.error(err)
else:
    st.info("請在左側輸入資產金額並按下「執行進攻型審計」鈕。")