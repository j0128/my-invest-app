import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go

# --- 0. 全局設定 ---
st.set_page_config(page_title="Alpha 2.0 Pro: 戰略資產中控台", layout="wide", page_icon="📈")

# 自定義 CSS 美化
st.markdown("""
<style>
    .metric-card {background-color: #0E1117; border: 1px solid #262730; border-radius: 5px; padding: 15px; color: white;}
    .bullish {color: #00FF7F; font-weight: bold;}
    .bearish {color: #FF4B4B; font-weight: bold;}
    .neutral {color: #FFD700; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# --- 1. 核心數據引擎 (OHLC 升級版) ---
@st.cache_data(ttl=3600)
def fetch_data(tickers):
    """
    一支一支下載 OHLC 數據，確保 K 線圖能畫出來，且不會因為 API 限制而崩潰。
    """
    benchmarks = ['QQQ', 'QLD', 'TQQQ', 'BTC-USD']
    all_tickers = list(set(tickers + benchmarks))
    
    # 準備容器
    dict_close = {}
    dict_open = {}
    dict_high = {}
    dict_low = {}
    
    # 顯示進度條
    progress_bar = st.progress(0, text="Alpha 正在建立加密連線...")
    
    for i, t in enumerate(all_tickers):
        try:
            progress_bar.progress((i + 1) / len(all_tickers), text=f"正在下載數據: {t} ...")
            
            # 使用 Ticker.history 抓取 1 年數據 (畫圖最佳長度)
            df = yf.Ticker(t).history(period="1y", auto_adjust=True)
            
            if df.empty: continue
                
            dict_close[t] = df['Close']
            dict_open[t] = df['Open']
            dict_high[t] = df['High']
            dict_low[t] = df['Low']
            
        except Exception:
            continue
            
    progress_bar.empty()

    # 轉為 DataFrame 並補值
    return (pd.DataFrame(dict_close).ffill(), 
            pd.DataFrame(dict_open).ffill(), 
            pd.DataFrame(dict_high).ffill(), 
            pd.DataFrame(dict_low).ffill())

# --- 2. 核心趨勢模組 ---
def analyze_trend(series):
    if series is None: return None
    series = series.dropna()
    if series.empty or len(series) < 20: return None

    try:
        y = series.values.reshape(-1, 1)
        x = np.arange(len(y)).reshape(-1, 1)
        
        model = LinearRegression().fit(x, y)
        k = model.coef_[0].item()
        r2 = model.score(x, y).item()
        
        p_now = series.iloc[-1].item()
        p_1m = model.predict([[len(y) + 22]])[0].item()
        ema20 = series.ewm(span=20).mean().iloc[-1].item()
        
        if p_now > ema20 and k > 0:
            status = "🔥 加速進攻"
            color = "bullish"
        elif p_now < ema20:
            status = "🛑 趨勢損毀"
            color = "bearish"
        else:
            status = "🛡️ 區間盤整"
            color = "neutral"
            
        return {"k": k, "r2": r2, "p_now": p_now, "p_1m": p_1m, "ema20": ema20, "status": status, "color": color}
    except:
        return None

# --- 3. 六維波動防禦 ---
def calc_volatility_shells(series):
    if series is None: return {}, "無數據"
    series = series.dropna()
    if series.empty: return {}, "無數據"
    try:
        window = 20
        rolling_mean = series.rolling(window).mean().iloc[-1].item()
        rolling_std = series.rolling(window).std().iloc[-1].item()
        curr_price = series.iloc[-1].item()
        
        levels = {}
        for i in range(1, 4):
            levels[f'H{i}'] = rolling_mean + (i * rolling_std)
            levels[f'L{i}'] = rolling_mean - (i * rolling_std)
            
        pos_desc = "正常波動"
        if curr_price > levels.get('H2', 9999