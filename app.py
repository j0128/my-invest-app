import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from sklearn.linear_model import LinearRegression
from datetime import datetime, date

# --- 頁面配置 ---
st.set_page_config(page_title="Alpha 2.0 Pro", layout="wide")

# --- 1. 核心：FRED API 金鑰掛鉤 (Secrets 優先) ---
@st.cache_resource
def init_fred():
    """
    優先從 Streamlit Secrets 讀取金鑰。
    格式要求: secrets.toml 中需包含 [FRED_API_KEY] 或直接在 dashboard 設定
    """
    api_key = None
    # 1. 嘗試從 secrets 讀取
    if "FRED_API_KEY" in st.secrets:
        api_key = st.secrets["FRED_API_KEY"]
    
    # 2. 初始化客戶端
    if api_key:
        try:
            client = Fred(api_key=api_key)
            return client
        except:
            return None
    return None

fred_client = init_fred()

# --- 2. 數據清洗：絕對標量化 (Anti-Series Logic) ---
def module_integrity_pro(df_raw):
    df = df_raw.copy()
    
    # [關鍵修正] 強制提取 Adj Close 並拋棄多層索引
    # 使用 xs (cross-section) 是處理 MultiIndex 最穩定的方法
    if isinstance(df.columns, pd.MultiIndex):
        try:
            # 嘗試提取 'Adj Close' 層級
            df = df.xs('Adj Close', axis=1, level=0, drop_level=True)
        except:
            # 如果失敗，嘗試扁平化最後一層
            df.columns = df.columns.get_level_values(-1)
    
    df = df.ffill().dropna(how='all')
    
    # 基準檢查
    if 'QQQ' not in df.columns:
        return None, "❌ 數據錯誤：未包含 QQQ，無法計算 Beta。"
    
    # 清洗 Inf/NaN
    clean_df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    if clean_df.empty:
        return None, "❌ 有效數據不足 (Empty DataFrame)。"
        
    return clean_df, None

# --- 3. 趨勢核心：使用 .item() 根除 TypeError ---
def module_core_pro(df):
    # 準備數據
    y = df['QQQ'].values.reshape(-1, 1)
    x = np.arange(len(y)).reshape(-1, 1)
    
    model = LinearRegression().fit(x, y)
    
    # [關鍵修正] 使用 .item() 強制轉為 Python 原生 float
    # 這能解決 "cannot convert series to float" 的所有變體
    k = model.coef_[0].item()
    eff = model.score(x, y).item()
    
    # 預測值
    ts_p = model.predict(x).flatten()
    p1 = model.predict([[len(y) + 22]])[0].item()
    
    # 20EMA 計算
    ema20_series = df['QQQ'].ewm(span=20).mean()
    
    # [關鍵修正] 取最後一個值時，務必使用 .item()
    current_price = df['QQQ'].iloc[-1].item()
    last_ema = ema20_series.iloc[-1].item()
    
    if current_price > last_ema and k > 0:
        status = "🔥 加速上升"
    elif current_price < last_ema:
        status = "🛑 趨勢損毀"
    else:
        status = "🛡️ 區間盤整"
        
    return {
        "k": k, "eff": eff, "p1": p1, 
        "ts_p": ts_p, "status": status, "ema20": ema20_series
    }

# --- 4. 組合審計：字典化處理 ---
def module_portfolio_pro(df, core, investments):
    # 將回報率轉為純字典，避開 Pandas Series 索引對齊問題
    rets_dict = df.pct_change().dropna().sum().to_dict()
    eps = 1e-12
    
    # 自動尋找目標 (非指數)
    indices = ['QQQ', 'QLD', 'TQQQ', 'BTC-USD']
    target = next((a for a in investments.keys() if a not in indices), 'QQQ')
    
    # 提取數值
    v_target = float(rets_dict.get(target, 0))
    v_qld = float(rets_dict.get('QLD', eps))
    v_tqqq = float(rets_dict.get('TQQQ', eps))
    
    if v_target > v_tqqq:
        grade = "Alpha+ (強於 3x)"
    elif v_target > v_qld:
        grade = "Beta+ (強於 2x)"
    else:
        grade = "Underperform (弱於槓桿)"
        
    total_cap = sum(investments.values()) if sum(investments.values()) > 0 else 1.0
    
    # Kelly
    win_rate = 0.6 if core['k'] > 0 else 0.4
    kelly = np.clip((win_rate * 2 - 1), 0, 0.75)
    
    return {
        "grade": grade, "target": target, "total": total_cap, 
        "kelly": kelly, "weights": {k: v/total_cap for k, v in investments.items()}
    }

# --- 5. 宏觀與外部因子 (整合 Secrets FRED) ---
def module_external(df, fred, exit_date):
    res = {"imp_score": 0.0, "fed_rate": 0.0, "pi_top": False, "msg": ""}
    
    # FRED 數據 (如果 secrets 有設定，fred 就不會是 None)
    if fred:
        try:
            # 抓取聯邦基金利率
            fed_data = fred.get_series('FEDFUNDS', limit=1)
            if not fed_data.empty:
                rate = fed_data.iloc[-1].item()
                res['fed_rate'] = rate
                res['imp_score'] = abs(rate - 4.5) * 1.5
        except Exception as e:
            res['msg'] = f"FRED 連線異常"
    else:
        res['msg'] = "未偵測到 Secrets FRED Key"

    # Pi Cycle (BTC)
    if 'BTC-USD' in df.columns:
        ma111 = df['BTC-USD'].rolling(111).mean().iloc[-1].item()
        ma350 = df['BTC-USD'].rolling(350).mean().iloc[-1].item() * 2
        res['pi_top'] = ma111 > ma350
        
    # 倒數
    today = date(2026, 1, 15)
    days = (exit_date - today).days
    res['days_left'] = days
    
    return res

# --- 6. 自動財報日 ---
def get_auto_earnings(ticker):
    calendar = {
        'AMD': '2026-01-27', 'NVDA': '2026-02-25', 'TSM': '2026-01-16',
        'QQQ': '2026-01-29', 'AAPL': '2026-01-30', 'MSFT': '2026-01-27'
    }
    return calendar.get(ticker.upper(), "N/A")

# --- UI 層 ---
st.sidebar.header("🎯 Alpha 2.0 調度中心")

# 使用 Form 防止重複刷新
with st.sidebar.form("audit_form"):
    st.caption(f"FRED API 狀態: {'✅ 已從 Secrets 載入' if fred_client else '⚠️ 未設定'}")
    
    monitored = st.multiselect(
        "核心資產 (必選 QQQ)", 
        ["QQQ","QLD","TQQQ","BTC-USD","AMD","NVDA","TSM","AAPL"], 
        default=["QQQ","QLD","TQQQ","AMD"]
    )
    
    st.markdown("---")
    st.write("💰 **持倉金額 (USD)**")
    invest_map = {}
    for asset in monitored:
        invest_map[asset] = st.number_input(f"{asset}", min_value=0, value=1000, step=100)
        
    exit_date_in = st.date_input("2026 清倉日", value=date(2026, 5, 31))
    
    btn = st.form_submit_button("🚀 執行 Alpha 2.0 審計")

st.title("🚀 Alpha 2.0 Pro: 進攻型深度審計 (2026 旗艦版)")

if btn:
    with st.spinner('正在從 Yahoo Finance 下載高頻數據...'):
        try:
            # 下載數據
            raw = yf.download(monitored, start="2024-01-01", end="2026-01-16", progress=False)
            
            if raw.empty:
                st.error("無法獲取數據，請檢查資產代號。")
            else:
                # 執行清洗
                clean, err = module_integrity_pro(raw)
                
                if err:
                    st.error(err)
                else:
                    # 執行模組
                    core = module_core_pro(clean)
                    port = module_portfolio_pro(clean, core, invest_map)
                    ext = module_external(clean, fred_client, exit_date_in)
                    
                    # 財報風險
                    e_date = get_auto_earnings(port['target'])
                    risk = "SAFE"
                    if e_date != "N/A":
                        d_left = (datetime.strptime(e_date, "%Y-%m-%d").date() - date(2026, 1, 15)).days
                        if d_left <= 7: risk = "⚠️ 禁區"
                    
                    # --- 儀表板 ---
                    k_c, f_c, p_c, t_c = st.columns(4)
                    k_c.metric("進攻斜率 (k)", f"{core['k']:.2f}", delta=core['status'])
                    
                    fred_val = f"{ext['imp_score']:.2f}" if fred_client else "N/A"
                    f_c.metric("FRED 驚奇指數", fred_val, delta=f"利率: {ext['fed_rate']}%" if fred_client else "未連線")
                    
                    p_c.metric("1M 預測價", f"${core['p1']:.2f}")
                    t_c.metric("總持倉價值", f"${port['total']:,.0f}")
                    
                    st.divider()
                    
                    lc, rc = st.columns(2)
                    with lc:
                        st.subheader(f"📊 選股等級：{port['grade']}")
                        st.caption(f"審計對象: {port['target']} | 財報日: {e_date} ({risk})")
                        st.bar_chart(pd.DataFrame(port['weights'].items(), columns=['A','W']).set_index('A'))
                        
                        if ext['pi_top']: st.error("🚨 BTC Pi Cycle 觸發頂部訊號！")
                        if ext['msg']: st.caption(ext['msg'])
                        
                    with rc:
                        st.subheader("📈 QQQ 趨勢生命線")
                        chart_df = pd.DataFrame({
                            "實際價格": clean['QQQ'][-60:].values,
                            "20EMA": core['ema20'][-60:].values,
                            "趨勢預測": core['ts_p'][-60:]
                        })
                        st.line_chart(chart_df, color=["#FF4B4B", "#1F77B4", "#FFA500"])
                        
                    st.success(f"📍 審計結論：Kelly 建議倉位上限 {port['kelly']:.0%}。距離 2026 撤退日剩餘 {ext['days_left']} 天。")
                    
        except Exception as e:
            st.error(f"執行時發生錯誤: {str(e)}")
else:
    st.info("請在左側配置持倉並點擊 **「🚀 執行 Alpha 2.0 審計」**。FRED Key 將自動從 Secrets 讀取。")