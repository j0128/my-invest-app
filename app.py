import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import datetime, date

# 1. 數據健康與異常監控 (Data Integrity)
def module_data_integrity(data_dict):
    """功能：殘差審計 (residual_audit) 與 缺口風險因子 (gap_risk_factor)"""
    df = pd.DataFrame(data_dict).ffill()
    # 檢測跳空缺口 (Gap Risk)
    df['gap_risk'] = df['QQQ'].pct_change().abs() > 0.03
    # 執行殘差審計
    clean_df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean_df) < 60:
        raise ValueError("Alpha 2.0 警告：有效樣本不足 60 日，審計無法啟動。")
    return clean_df

# 2. 核心趨勢與多週期預測
def module_core_projection(df):
    y = df['QQQ'].values.reshape(-1, 1)
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    
    k = model.coef_[0][0]
    eff = model.score(x, y)
    ts_p = model.predict(x).flatten()
    
    # 預測：1w (5d), 1m (22d, p1), 1q (66d, p3)
    p_1w = model.predict([[len(y) + 5]])[0][0]
    p_1m = model.predict([[len(y) + 22]])[0][0]
    p_1q = model.predict([[len(y) + 66]])[0][0]
    
    return {"k": k, "eff": eff, "p1": p_1m, "p3": p_1q, "ts_p": ts_p, "p_1w": p_1w}

# 3. 六維波動防禦區算與趨勢判定
def module_volatility_trend(df, core_results):
    current_p = df['QQQ'].iloc[-1]
    ts_p = core_results['ts_p']
    std = np.std(df['QQQ'].values - ts_p.reshape(-1, 1))
    
    shells = {f'l{i}': core_results['p1'] - i*std for i in range(1, 4)}
    shells.update({f'h{i}': core_results['p1'] + i*std for i in range(1, 4)})
    
    ema20 = df['QQQ'].ewm(span=20).mean().iloc[-1]
    trend_status = "🔥 加速上升" if current_p > ema20 and core_results['k'] > 0 else "🛡️ 盤整/損毀"
    
    return {"shells": shells, "ema20": ema20, "status": trend_status}

# 4. 槓桿與資產配置 (Portfolio Logic)
def module_portfolio_logic(df, core_results):
    returns = df.pct_change().dropna()
    # 審計是否跑贏 QLD (2倍) / TQQQ (3倍)
    bench_qld = (returns['AMD'].sum() / returns['QLD'].sum()) if 'AMD' in df.columns else 1.0
    bench_tqqq = (returns['AMD'].sum() / returns['TQQQ'].sum()) if 'TQQQ' in df.columns else 0.5
    
    alpha_grade = "Alpha+" if bench_tqqq > 1 else ("Beta+" if bench_qld > 1 else "Underperform")
    pQ = core_results['eff'] * bench_qld
    kelly_f = np.clip((0.6 if core_results['k'] > 0 else 0.4) * 2 - 1, 0, 0.75)
    
    return {"pQ": pQ, "kelly": kelly_f, "alpha_grade": alpha_grade}

# 5. 跨資產相關性與 Pi Cycle 頂部
def module_external_audit(df, exit_date_obj):
    ma111 = df['BTC'].rolling(window=111).mean().iloc[-1] if 'BTC' in df.columns else 0
    ma350_2 = (df['BTC'].rolling(window=350).mean().iloc[-1] * 2) if 'BTC' in df.columns else 1
    pi_top_signal = ma111 > ma350_2
    
    today = datetime(2026, 1, 15).date()
    days_left = (exit_date_obj - today).days
    exit_factor = np.clip(days_left / 136, 0, 1)
    
    return {"pi_top": pi_top_signal, "exit_factor": exit_factor}

# 6. 進攻型審計整合主程式
def run_strategic_audit_v5(data_dict, earnings_date_str, exit_date_obj):
    clean_df = module_data_integrity(data_dict)
    core = module_core_projection(clean_df)
    vol = module_volatility_trend(clean_df, core)
    port = module_portfolio_logic(clean_df, core)
    ext = module_external_audit(clean_df, exit_date_obj)
    
    # 財報監控
    today = datetime(2026, 1, 15).date()
    earn_dt = datetime.strptime(earnings_date_str, "%Y-%m-%d").date()
    days_to_earn = (earn_dt - today).days
    earn_risk = "⚠️ 禁區" if days_to_earn <= 7 else ("🛡️ 觀察" if days_to_earn <= 14 else "SAFE")

    return {**core, **vol, **port, **ext, "earn_risk": earn_risk, "gap_active": clean_df['gap_risk'].iloc[-1]}

# --- UI 渲染區 ---
st.set_page_config(page_title="Alpha 2.0 Quant", layout="wide")

# 側邊欄輸入
st.sidebar.header("🎯 進攻調度中心")
monitored_assets = st.sidebar.multiselect("監控資產", ["QQQ","QLD","TQQQ","BTC","AMD","0050"], default=["QQQ","QLD","TQQQ","BTC","AMD"])
earn_date = st.sidebar.date_input("下一季財報日", value=date(2026, 1, 28))
final_exit_date = st.sidebar.date_input("2026 獲利清倉日", value=date(2026, 5, 31))

# 數據補全邏輯 (Mock Data for 2026/01/15)
if 'p' not in globals():
    st.sidebar.warning("⚡ 啟動模擬數據模式")
    dates = pd.date_range(end='2026-01-15', periods=400)
    p = pd.DataFrame(index=dates)
    p['QQQ'] = np.linspace(400, 485, 400) + np.random.normal(0, 3, 400)
    p['QLD'] = p['QQQ'] * 0.2 + np.random.normal(0, 1, 400)
    p['TQQQ'] = p['QQQ'] * 0.15 + np.random.normal(0, 5, 400)
    p['BTC'] = np.linspace(80000, 105000, 400) + np.random.normal(0, 1000, 400)
    p['AMD'] = np.linspace(140, 210, 400) + np.random.normal(0, 8, 400)

# 執行審計並展示
st.title("🚀 Alpha 2.0 進攻型深度審計 (2026 版)")
try:
    res = run_strategic_audit_v5(p, earn_date.strftime("%Y-%m-%d"), final_exit_date)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("進攻斜率 (k)", f"{res['k']:.2f}", delta=res['status'])
    c2.metric("1M 目標價", f"${res['p1']:.2f}")
    c3.metric("選股等級", res['alpha_grade'])
    c4.metric("撤退因子", f"{res['exit_factor']:.1%}")

    st.divider()
    st.subheader("📊 20EMA 趨勢生命線與預測路徑")
    st.line_chart(pd.DataFrame({"實際價格": p['QQQ'][-60:], "預測趨勢": res['ts_p'][-60:]}))
    
    st.info(f"📍 狀態審計：財報風險 [{res['earn_risk']}] | BTC 頂部訊號 [{'⚠️ 觸發' if res['pi_top'] else '✅ 安全'}] | 跳空風險 [{'存在' if res['gap_active'] else '無'}]")

except Exception as e:
    st.error(f"系統啟動失敗：{e}")

