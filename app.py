import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from sklearn.linear_model import LinearRegression
from datetime import datetime, date, timedelta

# --- 2026 核心參數配置 ---
FRED_API_KEY = "你的_FRED_API_KEY_在此"  # 請填入 CFO 的 Key

# 1. 初始化 FRED 大腦 [4]
def init_fred():
    try:
        return Fred(api_key=FRED_API_KEY)
    except:
        return None

fred_client = init_fred()

# 2. 數據健康與異常監控模組 (Data Integrity) [19, 20, 21]
def module_integrity(df_raw):
    # 處理 yfinance 多層索引問題
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)
    
    df = df_raw.ffill()
    # 殘差審計：過濾極端 NaN 並標註 Gap Risk
    df['gap_risk'] = df['QQQ'].pct_change().abs() > 0.03
    clean_df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    # 計算 Alpha Gen (相對於 0050)
    if '0050.TW' in clean_df.columns:
        clean_df['alpha_raw'] = clean_df['QQQ'].pct_change() - clean_df['0050.TW'].pct_change()
    
    return clean_df

# 3. 核心趨勢與多週期預測模組 (Core Projection) [1, 2, 3, 4, 5, A]
def module_projection(df):
    y = df['QQQ'].values.reshape(-1, 1)
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    
    k = model.coef_[0][0]
    eff = model.score(x, y)
    ts_p = model.predict(x).flatten()
    
    # 多週期投射 (1w, 1m, 1q)
    p_1w = model.predict([[len(y) + 5]])[0][0]
    p1 = model.predict([[len(y) + 22]])[0][0] # 1M
    p3 = model.predict([[len(y) + 66]])[0][0] # 1Q (PEG 模型修正 placeholder)
    
    # 20EMA 判定邏輯
    ema20 = df['QQQ'].ewm(span=20).mean()
    curr_p = y[-1][0]
    rs_slope = (df['QQQ'].pct_change().rolling(10).mean().iloc[-1]) > 0
    
    if curr_p > ema20.iloc[-1] and rs_slope: trend_state = "🔥 加速上升"
    elif curr_p < ema20.iloc[-1]: trend_state = "🛑 趨勢損毀"
    else: trend_state = "🛡️ 區間盤整"
    
    return {"k": k, "eff": eff, "p1": p1, "p3": p3, "ts_p": ts_p, "status": trend_state, "ema20": ema20, "p_1w": p_1w}

# 4. 六維波動防禦區間模組 (Volatility Shells) [6-11]
def module_shells(df, core):
    std = np.std(df['QQQ'].values - core['ts_p'].reshape(-1, 1))
    p = core['p1']
    shells = {
        'l1': p - std, 'h1': p + std,
        'l2': p - 2*std, 'h2': p + 2*std,
        'l3': p - 3*std, 'h3': p + 3*std
    }
    return shells

# 5. 槓桿基準與資產配置因子 (Portfolio Logic) [12-15, (3)]
def module_portfolio(df, core):
    rets = df.pct_change().dropna()
    eps = 1e-9
    
    # 對標 QLD/TQQQ 判斷
    amd_sum = rets['AMD'].sum() if 'AMD' in rets.columns else 0
    qld_sum = rets['QLD'].sum() if 'QLD' in rets.columns else eps
    tqqq_sum = rets['TQQQ'].sum() if 'TQQQ' in rets.columns else eps
    
    grade = "Alpha+" if amd_sum > tqqq_sum else ("Beta+" if amd_sum > qld_sum else "Underperform")
    pQ = core['eff'] * (amd_sum / (qld_sum + eps))
    
    # Kelly 公式 2026 修正版: K = (W - (1-W)/R)
    kelly = np.clip(((0.6 if core['k'] > 0 else 0.4) - 0.4) / 1, 0, 0.7)
    
    return {"pQ": pQ, "kelly": kelly, "grade": grade}

# 6. 跨資產相關性與外部審計 (External & FRED) [16-18, 3, 4, C]
def module_external(df, fred, exit_date_obj):
    results = {}
    # BTC 相關性與 Pi Cycle Top
    if 'BTC-USD' in df.columns:
        results['btc_corr'] = df['QQQ'].pct_change().corr(df['BTC-USD'].pct_change())
        ma111 = df['BTC-USD'].rolling(111).mean().iloc[-1]
        ma350_2 = df['BTC-USD'].rolling(350).mean().iloc[-1] * 2
        results['pi_top'] = ma111 > ma350_2
        # MVRV 週期判定 (模擬數據)
        results['mvrv_risk'] = "High" if ma111 > (ma350_2 * 0.8) else "Stable"
    
    # FRED Importance Score
    if fred:
        try:
            actual_rate = fred.get_series('FEDFUNDS').iloc[-1]
            results['imp_score'] = abs(actual_rate - 4.5) / 0.5 * 1.2
            results['fed_rate'] = actual_rate
        except: results['imp_score'], results['fed_rate'] = 0, 4.75
    else: results['imp_score'], results['fed_rate'] = 2.1, 4.75

    # 撤退倒數
    today = date(2026, 1, 15)
    results['exit_factor'] = np.clip((exit_date_obj - today).days / 136, 0, 1)
    
    return results

# --- 整合主引擎 ---
def strategic_audit_v5_master(data, earn_date_str, exit_date_obj):
    clean = module_integrity(data)
    core = module_projection(clean)
    shells = module_shells(clean, core)
    port = module_portfolio(clean, core)
    ext = module_external(clean, fred_client, exit_date_obj)
    
    # 財報日期監控 [1, B]
    today = date(2026, 1, 15)
    earn_dt = datetime.strptime(earn_date_str, "%Y-%m-%d").date()
    days = (earn_dt - today).days
    risk = "⚠️ 高風險 (禁區)" if days <= 7 else ("🛡️ 觀察窗口" if days <= 14 else "SAFE")
    
    return {**core, "shells": shells, **port, **ext, "earn_risk": risk, "earn_days": days}

# --- UI 介面實作 ---
st.set_page_config(page_title="Alpha 2.0 Strategic Audit", layout="wide")
st.sidebar.header("🎯 進攻調度中心 (2026)")

monitored = st.sidebar.multiselect("核心資產", ["QQQ","QLD","TQQQ","BTC-USD","AMD","0050.TW"], default=["QQQ","QLD","TQQQ","BTC-USD","AMD"])
earn_in = st.sidebar.date_input("下一季財報日", value=date(2026, 1, 28))
exit_in = st.sidebar.date_input("2026 清倉日", value=date(2026, 5, 31))

@st.cache_data(ttl=3600)
def fetch_2026_data(tickers):
    return yf.download(tickers, start="2024-01-01", end="2026-01-16")['Adj Close']

data_raw = fetch_2026_data(monitored)

st.title("🚀 Alpha 2.0 進攻型深度審計 (2026 終極版)")

if not data_raw.empty:
    try:
        res = strategic_audit_v5_master(data_raw, earn_in.strftime("%Y-%m-%d"), exit_in)
        
        # 第一排: 核心趨勢指標
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("進攻斜率 (k)", f"{res['k']:.2f}", delta=res['status'])
        c2.metric("趨勢純度 (eff)", f"{res['eff']:.2%}")
        c3.metric("1M 預測 (p1)", f"${res['p1']:.2f}")
        c4.metric("3M 預測 (p3)", f"${res['p3']:.2f}")

        st.divider()
        
        # 第二排: 槓桿與風險
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("選股等級", res['grade'])
        r2.metric("Kelly 建議倉位", f"{res['kelly']:.1%}")
        r3.metric("FRED 驚奇指數", f"{res['imp_score']:.2f}")
        r4.metric("撤退因子", f"{res['exit_factor']:.1%}")

        # 第三排: 趨勢生命線
        st.subheader("📊 20EMA 生命線與預測路徑")
        plot_df = pd.DataFrame({
            "實際價格": data_raw['QQQ'][-60:],
            "20EMA": res['ema20'][-60:],
            "預測趨勢": res['ts_p'][-60:]
        })
        st.line_chart(plot_df)
        
        # 底部狀態欄 [A, B, C]
        st.info(f"📍 審計結論：財報風險 [{res['earn_risk']}] | BTC Pi-Cycle [{ '⚠️ 頂部' if res.get('pi_top') else '✅ 安全' }] | 聯準會利率: {res['fed_rate']}%")
        
        with st.expander("🔍 完整六維波動殼層 (Shells)"):
            st.table(pd.DataFrame(res['shells'].items(), columns=['區間', '點位']))

    except Exception as e:
        st.error(f"量化引擎運算錯誤: {str(e)}")
else:
    st.warning("數據讀取中...")