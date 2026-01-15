import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go

# --- 0. 全局設定 ---
st.set_page_config(page_title="Alpha 2.0: 戰略資產中控台", layout="wide", page_icon="📈")

# 自定義 CSS
st.markdown("""
<style>
    .metric-card {background-color: #0E1117; border: 1px solid #262730; border-radius: 5px; padding: 15px; color: white;}
    .bullish {color: #00FF7F; font-weight: bold;}
    .bearish {color: #FF4B4B; font-weight: bold;}
    .neutral {color: #FFD700; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# --- 1. 數據引擎 (增加容錯) ---
@st.cache_data(ttl=3600)
def fetch_data(tickers):
    benchmarks = ['QQQ', 'QLD', 'TQQQ', 'BTC-USD']
    all_tickers = list(set(tickers + benchmarks))
    try:
        data = yf.download(all_tickers, period="1y", auto_adjust=True)
        # 處理 yfinance 多層索引問題
        if isinstance(data.columns, pd.MultiIndex):
            try:
                df_close = data['Close']
                df_open = data['Open']
                df_high = data['High']
                df_low = data['Low']
                return df_close, df_open, df_high, df_low
            except:
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        return data['Close'], data['Open'], data['High'], data['Low']
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# --- 2. 趨勢模組 (簡化回穩) ---
def analyze_trend(series):
    # 安全檢查：如果數據