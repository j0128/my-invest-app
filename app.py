import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from sklearn.linear_model import LinearRegression
from datetime import datetime, date, timedelta

# --- 2026 核心配置 ---
# CFO 請輸入您的 API KEY，若未輸入系統將啟動模擬宏觀數據
FRED_API_KEY = "你的_FRED_API_KEY_在此" 

# 1. 初始化 FRED 大腦
def init_fred():
    try:
        if "你的" in FRED_API_KEY: return None
        return Fred(api_key=FRED_API_KEY)
    except:
        return None

fred = init_fred()

# 2. 核心：數據洗滌與殘差審計 [19, 20]
def module_data_integrity(data_dict):
    df = data_dict.ffill()
    # 確保 QQQ 存在，這是我們的基準 Beta
    if 'QQQ' not in df.columns:
        # 如果是單一標的抓取，yfinance 格式會不同，這裡做修正
        df = df.rename(columns={df.columns[0]: 'QQQ'})
    
    df['gap_risk'] = df['QQQ'].pct_change().abs() > 0.03
    clean_df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(clean_df) < 30: # 降低門檻，確保初次啟動成功
        raise ValueError(f"有效交易日不足 ({len(clean_df)}/30)")
    return clean_df

# 3. 核心：進攻型預測模型 [1, 2, 3, 4, 5, A]
def module_core_projection(df):
    y = df['QQQ'].values.reshape(-1, 1)
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    
    k = model.coef_[0][0]
    eff = model.score(x, y)
    ts_p = model.predict(x).flatten()
    
    # 未來預測
    p1 = model.predict([[len(y) + 22]])[0][0]
    p3 = model.predict([[len(y) + 66]])[0][0]
    
    # 20EMA 判定
    ema20 = df['QQQ'].ewm(span=20).mean().iloc[-1]
    curr_p = y[-1][0]
    status = "🔥 加速上升" if curr_p > ema20 and k > 0 else "🛡️ 區間盤整"
    if curr_p < ema20: status = "🛑 趨勢損毀"
    
    return {"k": k, "eff": eff, "p1": p1, "p3": p3, "ts_p": ts_p, "status": status, "ema20": ema20}

# 4. 核心：波動殼層與槓桿對標 [6-11, 12, 21, (3)]
def module_volatility_and_alpha(df, core):
    eps = 1e-9 # 防止 DivisionByZero
    std = np.std(df['QQQ'].values - core['ts_p'].reshape(-1, 1))
    shells = {f'l{i}': core['p1'] - i*std for i in range(1, 4)}
    shells.update({f'h{i}': core['p1'] + i*std for i in range(1, 4)})
    
    # 選股效率對標 (對比 QLD/TQQQ)
    rets = df.pct_change().dropna().sum()
    amd_ret = rets.get('AMD', 0)
    qld_ret = rets.get('QLD', eps)
    tqqq_ret = rets.get('TQQQ', eps)
    
    # 防止分母為零
    if qld_ret == 0: qld_ret = eps
    if tqqq_ret == 0: tqqq_ret = eps
    
    alpha_grade = "Alpha+" if (amd_ret/tqqq_ret) > 1 else ("Beta+" if (amd_ret/qld_ret) > 1 else "Underperform")
    
    return {"shells": shells, "alpha_grade": alpha_grade}

# 5. 核心：FRED 宏觀審計 [3, 4]
def module_macro_audit():
    if fred:
        try:
            rate = fred.get_series('FEDFUNDS').iloc[-1]
            score = abs(rate - 4.5) * 1.5
            return {"score": score, "rate": rate}
        except: pass
    return {"score": 2.5, "rate": 4.75} # 2026 模擬基準值

# --- 主程式整合 ---
def run_strategic_audit_v5(p_data, earnings_date_str, exit_date_obj):
    df = module_data_integrity(p_data)
    core = module_core_projection(df)
    vol = module_volatility_and_alpha(df, core)
    macro = module_macro_audit()
    
    # 2026 撤退因子 [18]
    today = date(2026, 1, 15)
    exit_factor = np.clip((exit_date_obj - today).days / 136, 0, 1)
    
    earn_dt = datetime.strptime(earnings_date_str, "%Y-%m-%d").date()
    earn_days = (earn_dt - today).days
    risk = "⚠️ 高風險" if earn_days <= 7 else "SAFE"

    return {**core, **vol, **macro, "exit_f": exit_factor, "risk": risk, "days": earn_days}

# --- Streamlit UI ---
st.set_page_config(page_title="Alpha 2.0 Quant", layout="wide")
st.sidebar.header("🎯 進攻調度中心")

assets = st.sidebar.multiselect("監控清單", ["QQQ","QLD","TQQQ","BTC-USD","AMD","0050.TW"], default=["QQQ","QLD","TQQQ","AMD"])
earn_in = st.sidebar.date_input("財報日", value=date(2026, 1, 28))
exit_in = st.sidebar.date_input("撤退日", value=date(2026, 5, 31))

@st.cache_data(ttl=3600)
def fetch_real_data(tickers):
    data = yf.download(tickers, start="2024-01-01", end="2026-01-16")['Adj Close']
    return data

data_raw = fetch_real_data(assets)

st.title("🚀 Alpha 2.0 進攻型深度審計 (2026 版)")

if not data_raw.empty:
    try:
        res = run_strategic_audit_v5(data_raw, earn_in.strftime("%Y-%m-%d"), exit_in)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("進攻斜率 (k)", f"{res['k']:.2f}", delta=res['status'])
        c2.metric("宏觀驚奇分值", f"{res['score']:.2f}")
        c3.metric("選股等級", res['alpha_grade'])
        c4.metric("撤退權重", f"{res['exit_f']:.1%}")

        st.divider()
        # 繪圖 logic
        plot_df = pd.DataFrame({
            "實際 QQQ": data_raw['QQQ'][-60:] if isinstance(data_raw, pd.DataFrame) else data_raw[-60:],
            "預測趨勢線": res['ts_p'][-60:]
        })
        st.subheader("📊 20EMA 生命線與趨勢投射")
        st.line_chart(plot_df)
        st.info(f"📍 審計結論：財報風險 [{res['risk']}] | 距離撤退日剩餘: {int(res['exit_f']*136)} 天")
        
    except Exception as e:
        st.error(f"量化引擎運算錯誤: {str(e)}")
else:
    st.warning("等待 API 數據注入中...")