import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from sklearn.linear_model import LinearRegression
from datetime import datetime, date, timedelta

# --- 2026 核心參數配置 ---
FRED_API_KEY = "你的_FRED_API_KEY_在此" 

# 1. 財報日自動預測邏輯 (2026 Q1 模擬)
def get_2026_earnings(ticker):
    schedule = {
        'AMD': '2026-01-27', 'NVDA': '2026-02-25', 'TSM': '2026-01-16',
        'QQQ': '2026-01-29', 'AAPL': '2026-01-30', 'MSFT': '2026-01-27'
    }
    return schedule.get(ticker.upper(), "2026-02-15")

# 2. 數據清洗模組：強力處理 MultiIndex 與殘差審計
def module_integrity(df_raw):
    df = df_raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        if 'Adj Close' in df.columns.levels[0]:
            df = df['Adj Close']
        else:
            df.columns = df.columns.get_level_values(-1)
    
    df = df.ffill().dropna(how='all')
    if 'QQQ' not in df.columns:
        return None, "❌ 基準缺失：請務必在側邊欄監控資產中勾選 QQQ。"
    
    clean_df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return clean_df, None

# 3. 核心運算引擎 (k, eff, p1, p3)
def module_projection(df):
    y = df['QQQ'].values.reshape(-1, 1)
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    
    k = float(model.coef_[0][0])
    eff = float(model.score(x, y))
    ts_p = model.predict(x).flatten()
    
    p1 = float(model.predict([[len(y) + 22]])[0][0]) # 1M
    p3 = float(model.predict([[len(y) + 66]])[0][0]) # 1Q
    
    ema20_series = df['QQQ'].ewm(span=20).mean()
    curr_p = float(y[-1][0])
    last_ema = float(ema20_series.iloc[-1])
    
    # 標量化比較判定趨勢
    if curr_p > last_ema and k > 0:
        status = "🔥 加速上升"
    elif curr_p < last_ema:
        status = "🛑 趨勢損毀"
    else:
        status = "🛡️ 區間盤整"
    
    return {"k": k, "eff": eff, "p1": p1, "p3": p3, "ts_p": ts_p, "status": status, "ema20": ema20_series}

# 4. 資金配比與組合審計
def module_portfolio_audit(df, core, investments):
    rets_df = df.pct_change().dropna()
    rets_sum = rets_df.sum().to_dict()
    eps = 1e-12
    
    target = [a for a in investments.keys() if a != 'QQQ'][0] if len(investments) > 1 else 'QQQ'
    
    val_target = float(rets_sum.get(target, 0))
    val_qld = float(rets_sum.get('QLD', eps))
    val_tqqq = float(rets_sum.get('TQQQ', eps))
    
    # 純數值比較 (解決 ValueError)
    if val_target > val_tqqq:
        grade = "Alpha+"
    elif val_target > val_qld:
        grade = "Beta+"
    else:
        grade = "Underperform"
    
    total_cap = sum(investments.values()) if sum(investments.values()) > 0 else 1
    
    # 波動殼層計算
    std = np.std(df['QQQ'].values - core['ts_p'].reshape(-1, 1))
    shells = {f'L{i}': core['p1'] - i*std for i in range(1, 4)}
    
    return {"grade": grade, "total": total_cap, "shells": shells, "target": target}

# --- UI 介面實作 ---
st.set_page_config(page_title="Alpha 2.0 Quant", layout="wide")
st.sidebar.header("🎯 進攻調度中心 (2026)")

with st.sidebar.form("audit_form"):
    monitored = st.multiselect("監控資產", ["QQQ","QLD","TQQQ","BTC-USD","AMD","NVDA","TSM"], default=["QQQ","QLD","TQQQ","AMD"])
    st.write("---")
    investments = {}
    for asset in monitored:
        investments[asset] = st.number_input(f"{asset} 持倉 (USD)", min_value=0, value=1000)
    
    exit_date = st.date_input("2026 清倉目標日", value=date(2026, 5, 31))
    submit = st.form_submit_button("🚀 執行進攻型深度審計")

st.title("🚀 Alpha 2.0 進攻型深度審計 (2026 版)")

if submit:
    # yfinance 下載
    raw_data = yf.download(monitored, start="2024-01-01", end="2026-01-16")
    
    if not raw_data.empty:
        clean, err = module_integrity(raw_data)
        if not err:
            core = module_core_projection(clean)
            res = module_portfolio_audit(clean, core, investments)
            
            # 第一排：核心看板
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("進攻斜率 (k)", f"{res['k']:.2f}", delta=res['status'])
            c2.metric("自動偵測財報日", get_2026_earnings(res['target']))
            c3.metric("1M 預測價 (QQQ)", f"${res['p1']:.2f}")
            c4.metric("總曝險價值", f"${res['total']:,.0f}")
            
            st.divider()
            
            # 第二排：圖表分析
            col_l, col_r = st.columns(2)
            with col_l:
                st.subheader(f"📊 選股等級：{res['grade']}")
                w_df = pd.DataFrame({k: [v/res['total']] for k,v in investments.items()}).T
                st.bar_chart(w_df)
                st.write(f"當前主要審計標的：**{res['target']}**")
            with col_r:
                st.subheader("📈 QQQ 趨勢生命線 (20EMA)")
                plot_df = pd.DataFrame({"實際價格": clean['QQQ'][-60:], "預測趨勢": core['ts_p'][-60:]})
                st.line_chart(plot_df)
            
            st.info(f"📍 審計結論：距離 2026/05 撤退日剩餘 {(exit_date - date(2026,1,15)).days} 天。")
        else:
            st.error(err)
else:
    st.info("請在左側輸入資產金額，並按下「執行進攻型深度審計」鈕。")