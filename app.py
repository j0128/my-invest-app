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

# --- 1. 核心數據引擎 (Data Engine - 修復版) ---
@st.cache_data(ttl=3600)
def fetch_data(tickers):
    """
    下載數據並強制標準化格式
    """
    benchmarks = ['QQQ', 'QLD', 'TQQQ', 'BTC-USD']
    all_tickers = list(set(tickers + benchmarks))
    
    try:
        # 下載數據
        data = yf.download(all_tickers, period="1y", auto_adjust=True, progress=False)
        
        if data.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        # [BUG 修復] 安全地判斷 Close 欄位名稱
        # 先預設為 'Close'
        close_col = 'Close'
        
        # 檢查是否為 MultiIndex 並且包含 Adj Close
        if isinstance(data.columns, pd.MultiIndex):
            # 檢查第一層級是否有 Adj Close
            if 'Adj Close' in data.columns.get_level_values(0):
                close_col = 'Adj Close'
        # 如果是 Single Index
        elif 'Adj Close' in data.columns:
            close_col = 'Adj Close'

        # 提取數據的輔助函數
        def extract_price_type(data, price_col_name):
            if isinstance(data.columns, pd.MultiIndex):
                try:
                    return data.xs(price_col_name, axis=1, level=0)
                except KeyError:
                    # 容錯：有時候 yfinance 結構會變
                    try:
                         return data.xs(price_col_name, axis=1, level=1)
                    except:
                        return pd.DataFrame()
            else:
                # Single Index: 通常發生在只剩一個有效 Ticker 時
                # 我們嘗試直接回傳，但不強制改名以免長度不符報錯
                if price_col_name in data.columns:
                    return data[[price_col_name]]
                return pd.DataFrame()

        df_close = extract_price_type(data, close_col)
        df_open  = extract_price_type(data, 'Open')
        df_high  = extract_price_type(data, 'High')
        df_low   = extract_price_type(data, 'Low')

        return df_close.ffill(), df_open.ffill(), df_high.ffill(), df_low.ffill()

    except Exception as e:
        # 捕捉所有錯誤，避免頁面崩潰，改為顯示警告
        st.warning(f"部分數據下載異常，系統將嘗試繼續運行。錯誤詳情: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# --- 2. 核心趨勢模組 (Trend Projection - 純標量版) ---
def analyze_trend(series):
    # 確保輸入是乾淨的 Series
    if series is None: return None
    series = series.dropna()
    if series.empty: return None
    if len(series) < 20: return None # 數據