import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go
import requests

# --- 0. 全局設定 ---
st.set_page_config(page_title="Alpha 2.0 Pro: 戰略資產中控台", layout="wide", page_icon="📈")

# 自定義 CSS
st.markdown("""
<style>
    .metric-card {background-color: #0E1117; border: 1px solid #262730; border-radius: 5px; padding: 15px; color: white;}
    .bullish {color: #00FF7F; font-weight: bold;}
    .bearish {color: #FF4B4B; font-weight: bold;}
    .neutral {color: #FFD700; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# --- 1. 強力數據引擎 (Double-Try Engine) ---
@st.cache_data(ttl=600) # 縮短緩存時間方便測試
def fetch_data_robust(tickers):
    """
    雙重機制下載 + Session 偽裝，專治 Yahoo 擋 IP
    """
    benchmarks = ['QQQ', 'QLD', 'TQQQ', 'BTC-USD']
    all_tickers = list(set(tickers + benchmarks))
    
    dict_close = {}
    dict_open = {}
    dict_high = {}
    dict_low = {}
    
    # 建立一個日誌區塊
    log_text = []
    
    # 嘗試建立偽裝 Session
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    })

    progress_bar = st.progress(0, text="初始化下載引擎...")
    
    for i, t in enumerate(all_tickers):
        progress_bar.progress((i + 1) / len(all_tickers), text=f"正在處理: {t}")
        success = False
        
        # --- 方法 A: 使用 yf.download (偽裝 Session) ---
        try:
            # 這裡我們不使用 session=session，因為 yfinance 新版有時候會衝突
            # 我們直接用最純粹的 download，但加上 ignore_tz
            df = yf.download(t, period="1y", auto_adjust=True, progress=False)
            
            # 檢查是否為空
            if not df.empty:
                # 處理可能的 MultiIndex (當只下載一支時，有時不會有 MultiIndex，有時會有)
                if isinstance(df.columns, pd.MultiIndex):
                    # 嘗試抓取 Close
                    try:
                        dict_close[t] = df.xs('Close', axis=1, level=0).iloc[:, 0]
                        dict_open[t]  = df.xs('Open', axis=1, level=0).iloc[:, 0]
                        dict_high[t]  = df.xs('High', axis=1, level=0).iloc[:, 0]
                        dict_low[t]   = df.xs('Low', axis=1, level=0).iloc[:, 0]
                    except:
                        # 如果結構不一樣，嘗試直接讀取
                        dict_close[t] = df['Close']
                        dict_open[t] = df['Open']
                        dict_high[t] = df['High']
                        dict_low[t] = df['Low']
                else:
                    # 單層索引
                    dict_close[t] = df['Close']
                    dict_open[t] = df['Open']
                    dict_high[t] = df['High']
                    dict_low[t] = df['Low']
                
                success = True
                log_text.append(f"✅ {t}: 下載成功 (Method A)")
        except Exception as e:
            log_text.append(f"⚠️ {t}: Method A 失敗 ({e})")

        # --- 方法 B: Ticker.history (備案) ---
        if not success:
            try:
                ticker_obj = yf.Ticker(t)
                # 這裡不傳入 session，使用預設
                df = ticker_obj.history(period="1y", auto_adjust=True)
                
                if not df.empty:
                    dict_close[t] = df['Close']
                    dict_open[t] = df['Open']
                    dict_high[t] = df['High']
                    dict_low[t] = df['Low']
                    success = True
                    log_text.append(f"✅ {t}: 下載成功 (Method B)")
                else:
                    log_text.append(f"❌ {t}: 數據為空 (可能代號錯誤或下市)")
            except Exception as e:
                log_text.append(f"❌ {t}: Method B 失敗 ({e})")

    progress_bar.empty()
    
    # 將日誌回傳，以便在前端顯示
    return pd.DataFrame(dict_close).ffill(), pd.DataFrame(dict_open).ffill(), \
           pd.DataFrame(dict_high).ffill(), pd.DataFrame(dict_low).ffill(), log_text

# --- 2. 趨勢模組 ---
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

# --- 3. 六維波動 ---
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
        if curr_price > levels.get('H2', 999999): pos_desc = "⚠️ 情緒過熱 (H2)"
        if curr_price < levels.get('L2', -999999): pos_desc = "💎 超賣機會 (L2)"
        return levels, pos_desc
    except:
        return {}, "計算錯誤"

# --- 4. 凱利公式 ---
def calc_kelly_position(trend_data):
    if not trend_data: return 0, 0
    win_rate = 0.55
    if trend_data['k'] > 0: win_rate += 0.05
    if trend_data['r2'] > 0.6: win_rate += 0.05
    if "損毀" in trend_data['status']: win_rate -= 0.15
    odds = 2.0 
    f_star = (odds * win_rate - (1 - win_rate)) / odds
    safe_kelly = max(0, f_star * 0.5) 
    return safe_kelly * 100, win_rate

# --- 5. 繪圖模組 ---
def plot_kline_chart(ticker, df_close, df_open, df_high, df_low):
    if ticker not in df_close.columns: return None
    try:
        lookback = 120
        dates = df_close.index[-lookback:]
        # 安全取值
        def get_series(df, t):
            if t in df.columns: return df[t].iloc[-len(dates):]
            return pd.Series()
            
        opens = get_series(df_open, ticker)
        highs = get_series(df_high, ticker)
        lows = get_series(df_low, ticker)
        closes = get_series(df_close, ticker)
        
        if len(closes) == 0: return None

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=dates, open=opens, high=highs, low=lows, close=closes,
            name='Price', increasing_line_color='#00FF7F', decreasing_line_color='#FF4B4B'
        ))
        ema20 = df_close[ticker].ewm(span=20).mean().iloc[-len(dates):]
        fig.add_trace(go.Scatter(
            x=dates, y=ema20, mode='lines', name='20 EMA',
            line=dict(color='#FFD700', width=1.5)
        ))
        fig.update_layout(
            title=f"{ticker} - Daily Chart", height=350, margin=dict(l=0, r=0, t=30, b=0),
            xaxis_rangeslider_visible=False, paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white')
        )
        return fig
    except:
        return None

# --- 6. 績效對比 ---
def plot_comparison(tickers, df_close):
    lookback = 120 
    valid = [t for t in tickers if t in df_close.columns]
    if not valid: return None
    try:
        df_slice = df_close[valid].iloc[-lookback:].copy()
        if df_slice.iloc[0].min() <= 0: return None
        df_norm = (df_slice / df_slice.iloc[0]) - 1
        fig = px.line(df_norm, x=df_norm.index, y=df_norm.columns, 
                      title="🔥 強弱對決：累積報酬率 (近120天)",
                      labels={'value': 'ROI', 'variable': 'Ticker'})
        fig.update_layout(
            height=400, hovermode="x unified", margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
            font=dict(color='white'), legend=dict(orientation="h", y=1.1)
        )
        return fig
    except:
        return None

def parse_input(input_text):
    portfolio = {}
    lines = input_text.strip().split('\n')
    for line in lines:
        if ',' in line:
            parts = line.split(',')
            ticker = parts[0].strip().upper()
            try: value = float(parts[1].strip())
            except: value = 0.0
            if ticker: portfolio[ticker] = value
        else:
            ticker = line.strip().upper()
            if ticker: portfolio[ticker] = 0.0
    return portfolio

# --- MAIN ---
def main():
    st.title("Alpha 2.0 Pro: 戰略資產中控台")
    st.caption("v10.0 診斷破防版 | 雙重下載機制")
    st.markdown("---")

    with st.sidebar:
        st.header("⚙️ 資產配置輸入")
        st.caption("格式：代號, 持倉金額")
        # 預設把 0050.TW 拿掉先測試美股，避免台股干擾
        default_input = """BTC-USD, 50000
QQQ, 30000
BNSOL-USD, 15000
NVDA, 10000"""
        user_input = st.text_area("持倉清單", default_input, height=200)
        portfolio_dict = parse_input(user_input)
        tickers_list = list(portfolio_dict.keys())
        total_value = sum(portfolio_dict.values())
        st.metric("總資產估值 (Est.)", f"${total_value:,.0f}")
        
        if st.button("🚀 啟動量化審計", type="primary"):
            st.session_state['run_analysis'] = True
        
    if not st.session_state.get('run_analysis', False):
        st.info("👈 請點擊『啟動量化審計』開始診斷。")
        return

    with st.spinner("Alpha 正在嘗試突破防火牆下載數據..."):
        # 呼叫新的強力下載函數
        df_close, df_open, df_high, df_low, log_text = fetch_data_robust(tickers_list)

    # --- 顯示診斷日誌 (Expander) ---
    with st.expander("📝 數據下載詳細日誌 (Debug Log)", expanded=True):
        for line in log_text:
            if "❌" in line:
                st.error(line)
            elif "⚠️" in line:
                st.warning(line)
            else:
                st.success(line)

    if df_close.empty:
        st.error("🚨 嚴重錯誤：所有下載嘗試均失敗。請檢查上方日誌。")
        return

    # --- A. 績效對比 ---
    st.subheader("1. 績效對比實驗室 (Benchmark Lab)")
    compare_list = ['QQQ', 'QLD', 'TQQQ'] + tickers_list[:3]
    compare_list = list(set(compare_list))
    fig_comp = plot_comparison(compare_list, df_close)
    if fig_comp: st.plotly_chart(fig_comp, use_container_width=True)
    
    st.markdown("#### 🇺🇸 美國大盤基準")
    b_col1, b