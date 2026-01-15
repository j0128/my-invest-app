import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from sklearn.linear_model import LinearRegression
from datetime import datetime, date, timedelta

# --- 2026 核心配置 ---
# 請在此處輸入你的 API Key
FRED_API_KEY = "你的_FRED_API_KEY_在此" 
fred = Fred(api_key=FRED_API_KEY)

def module_data_integrity(data_dict):
    """功能：殘差審計 (residual_audit) 與 缺口風險因子 [19, 20]"""
    df = pd.DataFrame(data_dict).ffill()
    df['gap_risk'] = df['QQQ'].pct_change().abs() > 0.03
    clean_df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean_df) < 60:
        raise ValueError("數據樣本不足 60 日")
    return clean_df

def module_core_projection(df):
    """功能：k, eff, p1, p3, ts_p 與 未來預測 [1, 2, 3, 4, 5]"""
    y = df['QQQ'].values.reshape(-1, 1)
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    
    k = model.coef_[0][0]
    eff = model.score(x, y) # R2
    ts_p = model.predict(x).flatten()
    
    # 預測：1w, 1m (p1), 1q (p3)
    p_1w = model.predict([[len(y) + 5]])[0][0]
    p_1m = model.predict([[len(y) + 22]])[0][0]
    p_1q = model.predict([[len(y) + 66]])[0][0]
    
    # 20EMA 趨勢判定 [A]
    ema20 = df['QQQ'].ewm(span=20).mean().iloc[-1]
    curr_p = y[-1][0]
    trend_state = "🔥 加速上升" if curr_p > ema20 and k > 0 else ("🛑 趨勢損毀" if curr_p < ema20 else "🛡️ 區間盤整")
    
    return {"k": k, "eff": eff, "p1": p_1m, "p3": p_1q, "ts_p": ts_p, "p_1w": p_1w, "status": trend_state, "ema20": ema20}

def module_volatility_and_alpha(df, core):
    """功能：六維殼層與 QLD/TQQQ 對標 [6-11, 12, 21]"""
    std = np.std(df['QQQ'].values - core['ts_p'].reshape(-1, 1))
    shells = {f'l{i}': core['p1'] - i*std for i in range(1, 4)}
    shells.update({f'h{i}': core['p1'] + i*std for i in range(1, 4)})
    
    # 對標審計 [(3)]
    rets = df.pct_change().dropna()
    amd_ret = rets['AMD'].sum() if 'AMD' in rets else 0
    qld_ret = rets['QLD'].sum() if 'QLD' in rets else 1
    tqqq_ret = rets['TQQQ'].sum() if 'TQQQ' in rets else 1
    
    alpha_grade = "Alpha+" if (amd_ret/tqqq_ret) > 1 else ("Beta+" if (amd_ret/qld_ret) > 1 else "Underperform")
    
    return {"shells": shells, "alpha_grade": alpha_grade}

def module_fred_macro_audit():
    """功能：重要消息量化篩選 [3] 與 Pi Cycle Top [4]"""
    try:
        # 獲取聯準會基準利率與 CPI 趨勢
        fed_rate = fred.get_series('FEDFUNDS').iloc[-1]
        cpi_data = fred.get_series('CPIAUCSL').pct_change(12).iloc[-1]
        
        # 模擬 Importance Score 邏輯 (Actual - Consensus)
        # 在 2026 年，若利率高於預期，ImportanceScore 會飆升
        importance_score = abs(fed_rate - 4.5) * 1.5 # 假設 2026 基準為 4.5%
        
        # Pi Cycle Top (使用 BTC 數據，這裡示範邏輯)
        # TopSignal = (111DMA > 350DMA * 2)
        return {"macro_score": importance_score, "fed_rate": fed_rate, "cpi": cpi_data}
    except:
        return {"macro_score": 0, "fed_rate": 0, "cpi": 0}

def run_strategic_audit_v5(p_data, earnings_date_str, exit_date_obj):
    clean_df = module_data_integrity(p_data)
    core = module_core_projection(clean_df)
    vol_alpha = module_volatility_and_alpha(clean_df, core)
    macro = module_fred_macro_audit()
    
    # 2026 五月撤退倒數 [18]
    today = date(2026, 1, 15)
    days_left = (exit_date_obj - today).days
    exit_factor = np.clip(days_left / 136, 0, 1)
    
    # 財報風險判定 [B]
    earn_dt = datetime.strptime(earnings_date_str, "%Y-%m-%d").date()
    earn_days = (earn_dt - today).days
    earn_risk = "⚠️ 高風險" if earn_days <= 7 else ("🛡️ 觀察期" if earn_days <= 14 else "SAFE")
    
    return {**core, **vol_alpha, **macro, "exit_factor": exit_factor, "earn_risk": earn_risk, "earn_days": earn_days}

st.set_page_config(page_title="Alpha 2.0 2026 Quant", layout="wide")
st.sidebar.header("🎯 進攻調度中心 (FRED 已接入)")

# 側邊欄輸入
monitored = st.sidebar.multiselect("監控資產", ["QQQ","QLD","TQQQ","BTC-USD","AMD"], default=["QQQ","QLD","TQQQ","BTC-USD","AMD"])
earn_date = st.sidebar.date_input("下一季財報日", value=date(2026, 1, 28))
exit_date = st.sidebar.date_input("2026 獲利清倉日", value=date(2026, 5, 31))

# 真實數據抓取 (yfinance)
@st.cache_data(ttl=3600)
def fetch_data(assets):
    return yf.download(assets, start="2024-01-01", end="2026-01-15")['Adj Close'].ffill().dropna()

p = fetch_data(monitored)

st.title("🚀 Alpha 2.0 進攻型深度審計 (2026 版)")

if not p.empty:
    res = run_strategic_audit_v5(p, earn_date.strftime("%Y-%m-%d"), exit_date)
    
    # 儀表板呈現
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("進攻斜率 (k)", f"{res['k']:.2f}", delta=res['status'])
    col2.metric("FRED 宏觀分值", f"{res['macro_score']:.2f}")
    col3.metric("選股等級", res['alpha_grade'])
    col4.metric("撤退因子", f"{res['exit_factor']:.1%}")

    st.divider()
    st.subheader("📊 20EMA 生命線與預測路徑")
    st.line_chart(pd.DataFrame({"實際 QQQ": p['QQQ'][-60:], "20EMA": p['QQQ'].ewm(span=20).mean()[-60:], "預測趨勢": res['ts_p'][-60:]}))
    
    st.info(f"📍 審計結果：財報風險 [{res['earn_risk']}，剩餘 {res['earn_days']} 天] | FRED 聯準會利率: {res['fed_rate']}%")
else:
    st.error("無法從 yfinance 獲取數據，請檢查網路連線。")

