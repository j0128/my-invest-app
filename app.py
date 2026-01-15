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
        if "你的" in api_key or not api_key: return None
        return Fred(api_key=api_key)
    except: return None

fred_client = get_fred_client(FRED_API_KEY)

# 1. 數據洗滌核心：徹底解決 KeyError 與 MultiIndex
def module_integrity_v6(df_raw):
    df = df_raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        # 優先尋找收盤價層級
        if 'Adj Close' in df.columns.levels[0]:
            df = df['Adj Close']
        else:
            df.columns = df.columns.get_level_values(-1)
    
    df = df.ffill().dropna(how='all')
    
    if 'QQQ' not in df.columns:
        return None, "❌ 基準缺失：請務必在側邊欄監控資產中勾選 QQQ。"
    
    return df.replace([np.inf, -np.inf], np.nan).dropna(), None

# 2. 趨勢審計引擎：解決 ValueError 與 TypeError
def module_core_v6(df):
    y = df['QQQ'].values.reshape(-1, 1)
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    
    k = float(model.coef_[0][0])
    eff = float(model.score(x, y))
    ts_p = model.predict(x).flatten()
    
    # 提取純數值 (Scalar)
    current_p = float(df['QQQ'].iloc[-1])
    ema20_val = float(df['QQQ'].ewm(span=20).mean().iloc[-1])
    
    status = "🔥 加速上升" if current_p > ema20_val and k > 0 else "🛡️ 區間盤整"
    if current_p < ema20_val: status = "🛑 趨勢損毀"
    
    return {"k": k, "eff": eff, "ts_p": ts_p, "status": status, "p1": float(model.predict([[len(y)+22]])[0][0])}

# 3. 組合優化與選股等級
def module_portfolio_v6(df, core, investments):
    rets = df.pct_change().dropna().sum().to_dict()
    eps = 1e-12
    
    # 動態鎖定非指數標的
    target_asset = [a for a in investments.keys() if a not in ['QQQ', 'QLD', 'TQQQ']][0] if len(investments) > 1 else 'QQQ'
    
    v_target = float(rets.get(target_asset, 0))
    v_qld = float(rets.get('QLD', eps))
    v_tqqq = float(rets.get('TQQQ', eps))
    
    grade = "Alpha+" if v_target > v_tqqq else ("Beta+" if v_target > v_qld else "Underperform")
    total_cap = sum(investments.values()) if sum(investments.values()) > 0 else 1
    
    return {"grade": grade, "target": target_asset, "total": total_cap, "weights": {k: v/total_cap for k, v in investments.items()}}

# --- UI 介面 ---
st.set_page_config(page_title="Alpha 2.0 Strategic Audit", layout="wide")
st.sidebar.header("🎯 進攻調度中心 (2026)")

with st.sidebar.form("master_form"):
    monitored = st.multiselect("核心資產", ["QQQ","QLD","TQQQ","BTC-USD","AMD","NVDA","TSM"], default=["QQQ","QLD","TQQQ","AMD"])
    st.write("---")
    invest_map = {}
    for asset in monitored:
        invest_map[asset] = st.number_input(f"{asset} 持倉 (USD)", min_value=0, value=1000)
    
    final_date = st.date_input("2026 清倉目標日", value=date(2026, 5, 31))
    submit = st.form_submit_button("🚀 執行量化深度審計")

st.title("🚀 Alpha 2.0 進攻型深度審計 (2026 版)")

if submit:
    # 數據抓取
    raw = yf.download(monitored, start="2024-01-01", end="2026-01-16")
    
    if not raw.empty:
        clean, err = module_integrity_v6(raw)
        if not err:
            core_res = module_core_v6(clean)
            port_res = module_portfolio_v6(clean, core_res, invest_map)
            
            # 儀表板渲染 [Image of a clean Streamlit dashboard with 4 metric columns, a bar chart, and a trend line chart]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("進攻斜率 (k)", f"{core_res['k']:.2f}", delta=core_res['status'])
            c2.metric("選股等級", port_res['grade'], help=f"對標基準為 {port_res['target']}")
            c3.metric("1M 預測價", f"${core_res['p1']:.2f}")
            c4.metric("總持倉價值", f"${port_res['total']:,.0f}")
            
            st.divider()
            
            l, r = st.columns(2)
            with l:
                st.subheader("📊 持倉比重分析")
                st.bar_chart(pd.DataFrame(port_res['weights'].items(), columns=['Asset', 'Weight']).set_index('Asset'))
            with r:
                st.subheader("📈 QQQ 趨勢生命線")
                plot_data = pd.DataFrame({
                    "實際價格": clean['QQQ'][-60:].values,
                    "預測路徑": core_res['ts_p'][-60:]
                })
                st.line_chart(plot_data)
            
            st.info(f"📍 審計結論：目前趨勢狀態為 {core_res['status']}。距離 2026 撤退日剩餘 {(final_date - date(2026,1,15)).days} 天。")
        else: