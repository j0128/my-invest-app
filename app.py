import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from sklearn.linear_model import LinearRegression
from datetime import datetime, date, timedelta

# --- 2026 核心配置 ---
FRED_API_KEY = "你的_FRED_API_KEY_在此" 

@st.cache_resource
def get_fred_client(api_key):
    try:
        if "你的" in api_key: return None
        return Fred(api_key=api_key)
    except: return None

fred_client = get_fred_client(FRED_API_KEY)

# 1. 財報日自動預測邏輯 (2026 Q1 版)
def get_2026_earnings(ticker):
    schedule = {
        'AMD': '2026-01-27', 'NVDA': '2026-02-25', 'TSM': '2026-01-16',
        'QQQ': '2026-01-29', 'AAPL': '2026-01-30', 'MSFT': '2026-01-27'
    }
    return schedule.get(ticker.upper(), "2026-02-15")

# 2. 數據清洗模組：強力處理 MultiIndex 與殘差審計
def module_integrity(df_raw):
    df = df_raw.copy()
    # 解決 image_4f476c 的 KeyError: 強制扁平化索引
    if isinstance(df.columns, pd.MultiIndex):
        if 'Adj Close' in df.columns.levels[0]:
            df = df['Adj Close']
        else:
            df.columns = df.columns.get_level_values(-1)
    
    df = df.ffill().dropna(how='all')
    if 'QQQ' not in df.columns:
        return None, "❌ 基準缺失：請務必在左側監控資產中勾選 QQQ。"
    
    clean_df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return clean_df, None

# 3. 趨勢與預測模組：解決 Series 比較與 TypeError
def module_projection(df):
    y_vals = df['QQQ'].values.reshape(-1, 1)
    x_vals = np.arange(len(y_vals)).reshape(-1, 1)
    model = LinearRegression().fit(x_vals, y_vals)
    
    k_val = float(model.coef_[0][0])
    eff_val = float(model.score(x_vals, y_vals))
    ts_p = model.predict(x_vals).flatten()
    p1_val = float(model.predict([[len(y_vals) + 22]])[0][0])
    
    ema20_series = df['QQQ'].ewm(span=20).mean()
    curr_price = float(y_vals[-1][0])
    last_ema_val = float(ema20_series.iloc[-1])
    
    # 強制使用標量比較，防止 ValueError
    if curr_price > last_ema_val and k_val > 0:
        status_tag = "🔥 加速上升"
    elif curr_price < last_ema_val:
        status_tag = "🛑 趨勢損毀"
    else:
        status_tag = "🛡️ 區間盤整"
    
    return {"k": k_val, "eff": eff_val, "p1": p1_val, "ts_p": ts_p, "status": status_tag, "ema20": ema20_series}

# 4. 組合審計模組：解決 division by zero 與 Series 標籤比較
def module_portfolio(df, core, investments):
    rets_df = df.pct_change().dropna()
    rets_sum_dict = rets_df.sum().to_dict() # 轉字典避開 Series 標籤
    eps_val = 1e-12
    
    target_ticker = [a for a in investments.keys() if a != 'QQQ'][0] if len(investments) > 1 else 'QQQ'
    
    v_target = float(rets_sum_dict.get(target_ticker, 0))
    v_qld = float(rets_sum_dict.get('QLD', eps_val))
    v_tqqq = float(rets_sum_dict.get('TQQQ', eps_val))
    
    # 解決 image_501165：純數值比較
    if v_target > v_tqqq:
        grade_tag = "Alpha+"
    elif v_target > v_qld:
        grade_tag = "Beta+"
    else:
        grade_tag = "Underperform"
    
    total_cap_val = sum(investments.values()) if sum(investments.values()) > 0 else 1
    std_val = np.std(df['QQQ'].values - core['ts_p'].reshape(-1, 1))
    shells_dict = {f'L{i}': core['p1'] - i*std_val for i in range(1, 4)}
    
    return {"grade": grade_tag, "total": total_cap_val, "shells": shells_dict, "target": target_ticker}

# --- UI 介面 ---
st.set_page_config(page_title="Alpha 2.0 Strategic Audit", layout="wide")
st.sidebar.header("🎯 進攻調度中心 (2026)")

with st.sidebar.form("audit_form"):
    monitored_list = st.multiselect("監控資產", ["QQQ","QLD","TQQQ","BTC-USD","AMD","NVDA","TSM"], default=["QQQ","QLD","TQQQ","AMD"])
    st.write("---")
    user_investments_dict = {}
    for asset_name in monitored_list:
        user_investments_dict[asset_name] = st.number_input(f"{asset_name} 持倉 (USD)", min_value=0, value=1000)
    
    exit_date_in = st.date_input("2026 清倉目標日", value=date(2026, 5, 31))
    submit_btn = st.form_submit_button("🚀 執行進攻型深度審計")

st.title("🚀 Alpha 2.0 進攻型深度審計 (2026 版)")

if submit_btn:
    raw_df = yf.download(monitored_list, start="2024-01-01", end="2026-01-16")
    
    if not raw_df.empty:
        clean_df, error_msg = module_integrity(raw_df)
        if not error_msg:
            core_res = module_projection(clean_df)
            port_res = module_portfolio(clean_df, core_res, user_investments_dict)
            
            # 數據展示
            cols = st.columns(4)
            cols[0].metric("進攻斜率 (k)", f"{core_res['k']:.2f}", delta=core_res['status'])
            cols[1].metric("自動財報日", get_2026_earnings(port_res['target']))
            cols[2].metric("1M 預測價", f"${core_res['p1']:.2f}")
            cols[3].metric("總曝險價值", f"${port_res['total']:,.0f}")
            
            st.divider()
            
            l_col, r_col = st.columns(2)
            with l_col:
                st.subheader(f"📊 選股等級：{port_res['grade']}")
                weights_df = pd.DataFrame({k: [v/port_res['total']] for k,v in user_investments_dict.items()}).T
                st.bar_chart(weights_df)
            with r_col:
                st.subheader("📈 QQQ 趨勢生命線 (20EMA)")
                plot_data = pd.DataFrame({"實際價格": clean_df['QQQ'][-60:], "預測趨勢": core_res['ts_p'][-60:]})
                st.line_chart(plot_data)
            
            st.info(f"📍 審計結論：目前主要標的為 {port_res['target']}。距離 2026/05 撤退日剩餘 {(exit_date_in - date(2026,1,15)).days} 天。")
        else:
            st.error(error_msg)
else:
    st.info("請在左側輸入資產持倉金額，並確保選中 QQQ 後點擊「執行進攻型深度審計」。")