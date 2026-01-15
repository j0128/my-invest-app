import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from sklearn.linear_model import LinearRegression
from datetime import datetime, date, timedelta

# --- 2026 核心配置 (CFO 請在此輸入 API Key) ---
FRED_API_KEY = "你的_FRED_API_KEY_在此" 

# 1. 宏觀數據核心：FRED 接入 (帶有回退機制)
@st.cache_resource
def get_fred_client(api_key):
    try:
        return Fred(api_key=api_key)
    except:
        return None

fred_client = get_fred_client(FRED_API_KEY)

# 2. 數據清洗：解決 ValueError 與 Gap Risk [19, 20]
def module_integrity(df_raw):
    # 強制降維處理 yfinance 多層索引
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)
    
    df = df_raw.ffill().dropna(how='all')
    if 'QQQ' not in df.columns:
        return None
        
    df['gap_risk'] = df['QQQ'].pct_change().abs() > 0.03
    clean_df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return clean_df

# 3. 趨勢審計：k、eff 與 20EMA 生命線 [1, 2, 3, 4, 5, A]
def module_projection(df):
    y = df['QQQ'].values.reshape(-1, 1)
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    
    k = model.coef_[0][0]
    eff = model.score(x, y)
    ts_p = model.predict(x).flatten()
    
    p1 = model.predict([[len(y) + 22]])[0][0] # 1M
    p3 = model.predict([[len(y) + 66]])[0][0] # 1Q
    
    ema20 = df['QQQ'].ewm(span=20).mean()
    curr_p = y[-1][0]
    # RS 斜率判定
    rs_slope = (df['QQQ'].pct_change().rolling(10).mean().iloc[-1] > 0)
    status = "🔥 加速上升" if curr_p > ema20.iloc[-1] and rs_slope else "🛡️ 區間盤整"
    if curr_p < ema20.iloc[-1]: status = "🛑 趨勢損毀"
    
    return {"k": k, "eff": eff, "p1": p1, "p3": p3, "ts_p": ts_p, "status": status, "ema20": ema20}

# 4. 波動殼層與槓桿對標：防禦 division by zero [6-11, 12, 21, (3)]
def module_portfolio(df, core):
    rets = df.pct_change().dropna()
    eps = 1e-12 # 極小值防禦
    
    # 選股效率審計
    amd_sum = rets['AMD'].sum() if 'AMD' in rets.columns else 0
    qld_sum = rets['QLD'].sum() if 'QLD' in rets.columns else eps
    tqqq_sum = rets['TQQQ'].sum() if 'TQQQ' in rets.columns else eps
    
    # 修正 division by zero 報錯位置
    div_qld = qld_sum if abs(qld_sum) > eps else eps
    div_tqqq = tqqq_sum if abs(tqqq_sum) > eps else eps
    
    grade = "Alpha+" if amd_sum > div_tqqq else ("Beta+" if amd_sum > div_qld else "Underperform")
    pQ = core['eff'] * (amd_sum / div_qld)
    kelly = np.clip(((0.6 if core['k'] > 0 else 0.4) - 0.4) / 1, 0, 0.75)
    
    std = np.std(df['QQQ'].values - core['ts_p'].reshape(-1, 1))
    shells = {f'l{i}': core['p1'] - i*std for i in range(1, 4)}
    shells.update({f'h{i}': core['p1'] + i*std for i in range(1, 4)})
    
    return {"pQ": pQ, "kelly": kelly, "grade": grade, "shells": shells}

# 5. 外部因子：Pi Cycle Top 與 2026 撤退權重 [16-18, 3, 4]
def module_external(df, fred, exit_date_obj):
    res = {"btc_corr": 0, "pi_top": False, "imp_score": 2.1, "fed_rate": 4.75}
    if 'BTC-USD' in df.columns:
        res['btc_corr'] = df['QQQ'].pct_change().corr(df['BTC-USD'].pct_change())
        ma111 = df['BTC-USD'].rolling(111).mean().iloc[-1]
        ma350_2 = df['BTC-USD'].rolling(350).mean().iloc[-1] * 2
        res['pi_top'] = ma111 > ma350_2
    
    if fred:
        try:
            res['fed_rate'] = fred.get_series('FEDFUNDS').iloc[-1]
            res['imp_score'] = abs(res['fed_rate'] - 4.5) * 1.5
        except: pass
        
    today = date(2026, 1, 15)
    res['exit_factor'] = np.clip((exit_date_obj - today).days / 136, 0, 1)
    return res

# --- 終極整合主程式 ---
def run_strategic_audit_v5(data, earn_date_str, exit_date_obj):
    clean = module_integrity(data)
    if clean is None or len(clean) < 30: return None
    
    core = module_projection(clean)
    port = module_portfolio(clean, core)
    ext = module_external(clean, fred_client, exit_date_obj)
    
    # 財報風險 [B]
    today = date(2026, 1, 15)
    earn_dt = datetime.strptime(earn_date_str, "%Y-%m-%d").date()
    days = (earn_dt - today).days
    risk = "⚠️ 禁區" if days <= 7 else ("🛡️ 觀察" if days <= 14 else "SAFE")
    
    return {**core, **port, **ext, "risk": risk, "days": days}

# --- UI 介面 ---
st.set_page_config(page_title="Alpha 2.0 Strategic Audit", layout="wide")
st.sidebar.header("🎯 進攻調度中心 (2026)")

# 側邊欄輸入
monitored = st.sidebar.multiselect("核心資產", ["QQQ","QLD","TQQQ","BTC-USD","AMD","0050.TW"], default=["QQQ","QLD","TQQQ","BTC-USD","AMD"])
earn_in = st.sidebar.date_input("下一季財報日", value=date(2026, 1, 28))
exit_in = st.sidebar.date_input("2026 清倉日", value=date(2026, 5, 31))

@st.cache_data(ttl=3600)
def fetch_2026_data(tickers):
    # 使用 yfinance 抓取 2026 年 1 月之前的真實數據
    return yf.download(tickers, start="2024-01-01", end="2026-01-16")['Adj Close']

data_raw = fetch_2026_data(monitored)

st.title("🚀 Alpha 2.0 進攻型深度審計 (2026 版)")

if not data_raw.empty:
    res = run_strategic_audit_v5(data_raw, earn_in.strftime("%Y-%m-%d"), exit_in)
    
    if res:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("進攻斜率 (k)", f"{res['k']:.2f}", delta=res['status'])
        c2.metric("趨勢純度 (eff)", f"{res['eff']:.2%}")
        c3.metric("1M 預測價", f"${res['p1']:.2f}")
        c4.metric("撤退因子", f"{res['exit_factor']:.1%}")

        st.divider()
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("選股等級", res['grade'])
        r2.metric("Kelly 建議倉位", f"{res['kelly']:.1%}")
        r3.metric("宏觀驚奇分值", f"{res['imp_score']:.2f}")
        r4.metric("3M 目標 (p3)", f"${res['p3']:.2f}")

        # 20EMA 生命線圖表化 [2]
        st.subheader("📊 20EMA 生命線與預測路徑")
        plot_df = pd.DataFrame({
            "實際價格": data_raw['QQQ'][-60:],
            "20EMA": res['ema20'][-60:],
            "預測趨勢": res['ts_p'][-60:]
        })
        st.line_chart(plot_df)
        
        st.info(f"📍 審計結論：財報風險 [{res['risk']}] | BTC Pi-Cycle [{'⚠️ 頂部' if res['pi_top'] else '✅ 安全'}] | 聯準會利率: {res['fed_rate']}%")
        
        with st.expander("🔍 完整六維波動殼層 (Volatility Shells)"):
            st.table(pd.DataFrame(res['shells'].items(), columns=['區間', '點位']))
    else:
        st.warning("數據對齊失敗，請嘗試在左側更換資產。")
else:
    st.error("API 數據注入失敗，請檢查網路連線或 API 金鑰。")