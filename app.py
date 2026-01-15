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

# 1. 數據清洗模組：強力處理 MultiIndex 與 NaN
def module_integrity(df_raw):
    df = df_raw.copy()
    # 解決 image_4f476c 的 KeyError: 強制扁平化 yfinance 下載的多層索引
    if isinstance(df.columns, pd.MultiIndex):
        if 'Adj Close' in df.columns.levels[0]:
            df = df['Adj Close']
        else:
            df.columns = df.columns.get_level_values(-1)
    
    df = df.ffill().dropna(how='all')
    # 解決 image_4fb011: 確保 QQQ 存在且名稱正確
    if 'QQQ' not in df.columns:
        return None, "❌ 基準缺失：請務必在左側監控資產中勾選 QQQ"
    
    clean_df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return clean_df, None

# 2. 趨勢與預測模組：解決 image_50190a 的 Series 比較報錯
def module_projection(df):
    y = df['QQQ'].values.reshape(-1, 1)
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    
    k, eff = model.coef_[0][0], model.score(x, y)
    ts_p = model.predict(x).flatten()
    p1 = float(model.predict([[len(y) + 22]])[0][0])
    
    ema20_series = df['QQQ'].ewm(span=20).mean()
    curr_p = float(y[-1][0])
    last_ema = float(ema20_series.iloc[-1])
    
    # 修正 Series 比較邏輯：使用標量比較，防止 ValueError
    if curr_p > last_ema and k > 0:
        status = "🔥 加速上升"
    elif curr_p < last_ema:
        status = "🛑 趨勢損毀"
    else:
        status = "🛡️ 區間盤整"
    
    return {"k": k, "eff": eff, "p1": p1, "ts_p": ts_p, "status": status, "ema20": ema20_series}

# 3. 組合審計模組：解決 image_501165 的分母與標籤比較報錯
def module_portfolio(df, core, investments):
    rets_df = df.pct_change().dropna()
    # 關鍵修正：將所有 Series 轉換為標量字典
    rets_sum = rets_df.sum().to_dict()
    eps = 1e-12 
    
    target = [a for a in investments.keys() if a != 'QQQ'][0] if len(investments) > 1 else 'QQQ'
    
    val_target = float(rets_sum.get(target, 0))
    val_qld = float(rets_sum.get('QLD', eps))
    val_tqqq = float(rets_sum.get('TQQQ', eps))
    
    # 純標量數值比較，徹底根除 Pandas 報錯
    if val_target > val_tqqq:
        grade = "Alpha+"
    elif val_target > val_qld:
        grade = "Beta+"
    else:
        grade = "Underperform"
    
    total_cap = sum(investments.values()) if sum(investments.values()) > 0 else 1
    kelly = np.clip(((0.6 if core['k'] > 0 else 0.4) - 0.4) / 1, 0, 0.75)
    
    std = np.std(df['QQQ'].values - core['ts_p'].reshape(-1, 1))
    shells = {f'L{i}': core['p1'] - i*std for i in range(1, 4)}
    
    return {"grade": grade, "kelly": kelly, "shells": shells, "target": target, "total": total_cap}

# 4. 外部因子模組
def module_external(df, fred, exit_date_obj):
    res = {"btc_corr": 0, "pi_top": False, "imp_score": 2.1, "fed_rate": 4.75}
    if 'BTC-USD' in df.columns:
        res['btc_corr'] = df['QQQ'].pct_change().corr(df['BTC-USD'].pct_change())
        ma111 = df['BTC-USD'].rolling(111).mean().iloc[-1]
        ma350_2 = df['BTC-USD'].rolling(350).mean().iloc[-1] * 2
        res['pi_top'] = bool(ma111 > ma350_2)
    
    if fred:
        try:
            res['fed_rate'] = fred.get_series('FEDFUNDS').iloc[-1]
            res['imp_score'] = abs(res['fed_rate'] - 4.5) * 1.5
        except: pass
    
    today = date(2026, 1, 15)
    res['exit_factor'] = np.clip((exit_date_obj - today).days / 136, 0, 1)
    return res

# --- UI 介面 ---
st.set_page_config(page_title="Alpha 2.0 Strategic Audit", layout="wide")
st.sidebar.header("🎯 進攻調度中心 (2026)")

with st.sidebar.form("audit_form"):
    monitored = st.multiselect("監控資產", ["QQQ","QLD","TQQQ","BTC-USD","AMD","NVDA","TSM"], default=["QQQ","QLD","TQQQ","AMD"])
    st.write("---")
    investments = {}
    for asset in monitored:
        investments[asset] = st.number_input(f"{asset} 持倉 (USD)", min_value=0, value=1000)
    exit_in = st.date_input("2026 清倉日", value=date(2026, 5, 31))
    submit = st.form_submit_button("🚀 執行進攻型深度審計")

st.title("🚀 Alpha 2.0 進攻型深度審計 (2026 版)")

if submit:
    raw_data = yf.download(monitored, start="2024-01-01", end="2026-01-16")
    
    if not raw_data.empty:
        clean, err = module_integrity(raw_data)
        if not err:
            core = module_projection(clean)
            port = module_portfolio(clean, core, investments)
            ext = module_external(clean, fred_client, exit_in)
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("進攻斜率 (k)", f"{core['k']:.2f}", delta=core['status'])
            c2.metric("趨勢純度 (eff)", f"{core['eff']:.2%}")
            c3.metric("1M 預測價", f"${core['p1']:.2f}")
            c4.metric("總持倉價值", f"${port['total']:,.0f}")
            
            st.divider()
            col_l, col_r = st.columns(2)
            with col_l:
                st.subheader(f"📊 選股等級：{port['grade']}")
                w_df = pd.DataFrame({k: [v/port['total']] for k,v in investments.items()}).T
                st.bar_chart(w_df)
                st.write(f"針對 **{port['target']}** 進行對標審計。")
            with col_r:
                st.subheader("📈 QQQ 趨勢生命線")
                plot_df = pd.DataFrame({"實際價格": clean['QQQ'][-60:].values, "預測趨勢": core['ts_p'][-60:]})
                st.line_chart(plot_df)
            
            st.info(f"📍 審計結論：撤退倒數中，剩餘因子: {ext['exit_factor']:.2%} | BTC Pi-Cycle: {'⚠️ 頂部' if ext['pi_top'] else '✅ 安全'}")
        else:
            st.error(err)
else:
    st.info("請在左側輸入資產持倉，按下確認鍵啟動量化引擎。")