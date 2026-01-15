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

# 1. 數據洗滌模組：解決 KeyError 與 MultiIndex 問題
def module_integrity(df_raw):
    # 處理 yfinance 多標的下載產生的 MultiIndex 結構
    if isinstance(df_raw.columns, pd.MultiIndex):
        if 'Adj Close' in df_raw.columns.levels[0]:
            df = df_raw['Adj Close'].copy()
        else:
            df = df_raw.copy()
            df.columns = df.columns.get_level_values(0)
    else:
        df = df_raw.copy()

    df = df.ffill().dropna(how='all')
    
    # 2026 審計依準：必須有 QQQ 作為 Beta 基準
    if 'QQQ' not in df.columns:
        return None, "請在側邊欄選取 QQQ 作為審計基準"
        
    df['gap_risk'] = df['QQQ'].pct_change().abs() > 0.03
    clean_df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return clean_df, None

# 2. 趨勢審計模組
def module_projection(df):
    y = df['QQQ'].values.reshape(-1, 1)
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    
    k, eff = model.coef_[0][0], model.score(x, y)
    ts_p = model.predict(x).flatten()
    p1 = model.predict([[len(y) + 22]])[0][0]
    p3 = model.predict([[len(y) + 66]])[0][0]
    
    ema20 = df['QQQ'].ewm(span=20).mean()
    curr_p = y[-1][0]
    status = "🔥 加速上升" if curr_p > ema20.iloc[-1] and k > 0 else "🛡️ 區間盤整"
    if curr_p < ema20.iloc[-1]: status = "🛑 趨勢損毀"
    
    return {"k": k, "eff": eff, "p1": p1, "p3": p3, "ts_p": ts_p, "status": status, "ema20": ema20}

# 3. 組合審計模組：解決 Division by zero
def module_portfolio(df, core):
    rets = df.pct_change().dropna()
    eps = 1e-12 
    
    # 動態獲取非 QQQ 的第一個標的進行 Alpha 審計
    other_assets = [c for c in df.columns if c not in ['QQQ', 'QLD', 'TQQQ', 'BTC-USD', 'gap_risk']]
    target = other_assets[0] if other_assets else 'QQQ'
    
    target_sum = rets[target].sum() if target in rets.columns else 0
    qld_sum = rets['QLD'].sum() if 'QLD' in rets.columns else eps
    tqqq_sum = rets['TQQQ'].sum() if 'TQQQ' in rets.columns else eps
    
    # 零值防禦
    div_qld = qld_sum if abs(qld_sum) > eps else eps
    div_tqqq = tqqq_sum if abs(tqqq_sum) > eps else eps
    
    grade = "Alpha+" if target_sum > div_tqqq else ("Beta+" if target_sum > div_qld else "Underperform")
    pQ = core['eff'] * (target_sum / div_qld)
    kelly = np.clip(((0.6 if core['k'] > 0 else 0.4) - 0.4) / 1, 0, 0.75)
    
    std = np.std(df['QQQ'].values - core['ts_p'].reshape(-1, 1))
    shells = {f'L{i}': core['p1'] - i*std for i in range(1, 4)}
    shells.update({f'H{i}': core['p1'] + i*std for i in range(1, 4)})
    
    return {"pQ": pQ, "kelly": kelly, "grade": grade, "shells": shells, "target_name": target}

# 4. 外部因素與撤退倒數
def module_external(df, fred, exit_date_obj):
    res = {"btc_corr": 0, "pi_top": False, "imp_score": 2.1, "fed_rate": 4.75}
    if 'BTC-USD' in df.columns:
        res['btc_corr'] = df['QQQ'].pct_change().corr(df['BTC-USD'].pct_change())
        ma111, ma350_2 = df['BTC-USD'].rolling(111).mean().iloc[-1], df['BTC-USD'].rolling(350).mean().iloc[-1] * 2
        res['pi_top'] = ma111 > ma350_2
    
    if fred:
        try:
            res['fed_rate'] = fred.get_series('FEDFUNDS').iloc[-1]
            res['imp_score'] = abs(res['fed_rate'] - 4.5) * 1.5
        except: pass
        
    days_left = (exit_date_obj - date(2026, 1, 15)).days
    res['exit_factor'] = np.clip(days_left / 136, 0, 1)
    return res

# --- 整合主引擎 ---
def run_strategic_audit_v5(data, earn_date_str, exit_date_obj):
    clean, err = module_integrity(data)
    if err: return {"error": err}
    if clean is None or len(clean) < 30: return {"error": "有效樣本不足"}
    
    core = module_projection(clean)
    port = module_portfolio(clean, core)
    ext = module_external(clean, fred_client, exit_date_obj)
    
    earn_dt = datetime.strptime(earn_date_str, "%Y-%m-%d").date()
    days = (earn_dt - date(2026, 1, 15)).days
    risk = "⚠️ 禁區" if days <= 7 else ("🛡️ 觀察" if days <= 14 else "SAFE")
    
    return {**core, **port, **ext, "risk": risk, "days": days}

# --- UI 介面 ---
st.set_page_config(page_title="Alpha 2.0 Strategic Audit", layout="wide")
st.sidebar.header("🎯 進攻調度中心 (2026)")

monitored = st.sidebar.multiselect("核心資產", ["QQQ","QLD","TQQQ","BTC-USD","AMD","TSM","NVDA"], default=["QQQ","QLD","TQQQ","BTC-USD","AMD"])
earn_in = st.sidebar.date_input("下一季財報日", value=date(2026, 1, 28))
exit_in = st.sidebar.date_input("2026 清倉日", value=date(2026, 5, 31))

@st.cache_data(ttl=3600)
def fetch_2026_data(tickers):
    try:
        # 修正：yf.download 直接獲取 Adj Close 以減少索引複雜度
        return yf.download(tickers, start="2024-01-01", end="2026-01-16")
    except: return pd.DataFrame()

raw_data = fetch_2026_data(monitored)

st.title("🚀 Alpha 2.0 進攻型深度審計 (2026 版)")

if not raw_data.empty:
    res = run_strategic_audit_v5(raw_data, earn_in.strftime("%Y-%m-%d"), exit_in)
    
    if "error" not in res:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("進攻斜率 (k)", f"{res['k']:.2f}", delta=res['status'])
        c2.metric("趨勢純度 (eff)", f"{res['eff']:.2%}")
        c3.metric("1M 預測價 (QQQ)", f"${res['p1']:.2f}")
        c4.metric("2026 撤退因子", f"{res['exit_factor']:.1%}")

        st.divider()
        r1, r2, r3, r4 = st.columns(4)
        r1.metric(f"等級 ({res['target_name']})", res['grade'])
        r2.metric("Kelly 倉位", f"{res['kelly']:.1%}")
        r3.metric("宏觀驚奇分值", f"{res['imp_score']:.2f}")
        r4.metric("3M 目標價", f"${res['p3']:.2f}")

        st.subheader("📊 20EMA 生命線與預測路徑")
        plot_df = pd.DataFrame({"實際 QQQ": raw_data.xs('Adj Close', axis=1, level=0)['QQQ'][-60:] if isinstance(raw_data.columns, pd.MultiIndex) else raw_data['QQQ'][-60:]})
        plot_df["20EMA"] = res['ema20'][-60:]
        plot_df["預測趨勢"] = res['ts_p'][-60:]
        st.line_chart(plot_df)
        
        st.info(f"📍 審計結論：財報風險 [{res['risk']}] | BTC Pi-Cycle [{'⚠️ 頂部' if res['pi_top'] else '✅ 安全'}] | 聯準會利率: {res['fed_rate']}%")
        with st.expander("🔍 完整六維波動殼層點位"):
            st.table(pd.DataFrame(res['shells'].items(), columns=['區間', '目標點位']))
    else:
        st.error(res["error"])
else:
    st.warning("數據讀取中，請檢查網路連線或資產輸入是否正確。")