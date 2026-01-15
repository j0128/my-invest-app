import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from sklearn.linear_model import LinearRegression
from datetime import datetime, date

# --- 頁面配置 ---
st.set_page_config(page_title="Alpha 2.0 Pro", layout="wide")

# --- 1. FRED API (Secrets 優先) ---
@st.cache_resource
def init_fred():
    api_key = st.secrets.get("FRED_API_KEY", None)
    if api_key:
        try:
            return Fred(api_key=api_key)
        except: return None
    return None

fred_client = init_fred()

# --- 2. 數據清洗：解鎖 QQQ 限制 + 隱形基準 ---
def module_integrity_unlocked(df_raw, user_selected_assets):
    df = df_raw.copy()
    
    # 強制扁平化 MultiIndex (防彈邏輯)
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs('Adj Close', axis=1, level=0, drop_level=True)
        except:
            df.columns = df.columns.get_level_values(-1)
            
    df = df.ffill().dropna(how='all')
    
    # 這裡不再報錯 "QQQ 缺失"，而是自動決定誰是主角
    # 如果數據完全空了才報錯
    clean_df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    if clean_df.empty:
        return None, "❌ 數據為空，請檢查代號。"
        
    return clean_df, None

# --- 3. 趨勢核心：動態鎖定主角 (Dynamic Core) ---
def module_core_dynamic(df, target_ticker):
    """
    如果 QQQ 存在，優先分析 QQQ (大盤)。
    如果 QQQ 不在，直接分析 target_ticker (個股)。
    """
    # 決定分析對象
    analyze_target = 'QQQ' if 'QQQ' in df.columns else target_ticker
    
    # 防禦：如果連 target 都不在數據裡 (極端情況)
    if analyze_target not in df.columns:
        analyze_target = df.columns[0]
        
    y = df[analyze_target].values.reshape(-1, 1)
    x = np.arange(len(y)).reshape(-1, 1)
    
    model = LinearRegression().fit(x, y)
    
    # 純標量提取 (.item)
    k = model.coef_[0].item()
    eff = model.score(x, y).item()
    ts_p = model.predict(x).flatten()
    p1 = model.predict([[len(y) + 22]])[0].item()
    
    # 20EMA
    ema20_series = df[analyze_target].ewm(span=20).mean()
    curr_price = df[analyze_target].iloc[-1].item()
    last_ema = ema20_series.iloc[-1].item()
    
    if curr_price > last_ema and k > 0:
        status = "🔥 加速上升"
    elif curr_price < last_ema:
        status = "🛑 趨勢損毀"
    else:
        status = "🛡️ 區間盤整"
        
    return {
        "k": k, "eff": eff, "p1": p1, "ts_p": ts_p, 
        "status": status, "ema20": ema20_series, 
        "analyzed_subject": analyze_target
    }

# --- 4. 組合審計：強制對標 QLD/TQQQ ---
def module_portfolio_compare(df, core, investments):
    rets_dict = df.pct_change().dropna().sum().to_dict()
    eps = 1e-12
    
    # 找出使用者最關注的個股 (排除基準 ETF)
    benchmarks = ['QQQ', 'QLD', 'TQQQ', 'BTC-USD']
    # 從使用者輸入的持倉中找，如果找不到就隨便拿一個
    user_picks = [a for a in investments.keys() if a not in benchmarks]
    target = user_picks[0] if user_picks else (list(investments.keys())[0] if investments else 'N/A')
    
    # 提取回報數值 (若 QLD/TQQQ 沒被選，這裡會是 0 或 eps，但不崩潰)
    v_target = float(rets_dict.get(target, 0))
    # 關鍵：這裡我們假設 QLD/TQQQ 已經在 df 裡 (因為我們會強制下載)
    v_qld = float(rets_dict.get('QLD', eps)) 
    v_tqqq = float(rets_dict.get('TQQQ', eps))
    
    # 評級邏輯
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

# --- 5. 輔助功能 ---
def module_external(df, fred, exit_date):
    res = {"imp_score": 0.0, "fed_rate": 0.0, "pi_top": False}
    if fred:
        try:
            fed_data = fred.get_series('FEDFUNDS', limit=1)
            if not fed_data.empty:
                rate = fed_data.iloc[-1].item()
                res['fed_rate'] = rate
                res['imp_score'] = abs(rate - 4.5) * 1.5
        except: pass

    if 'BTC-USD' in df.columns:
        ma111 = df['BTC-USD'].rolling(111).mean().iloc[-1].item()
        ma350 = df['BTC-USD'].rolling(350).mean().iloc[-1].item() *