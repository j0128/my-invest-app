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

# --- 1. 核心數據引擎 (OHLC 序列下載版) ---
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
            
            # 使用 Ticker.history 抓取 1 年數據
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
    if series.empty: return None

    y = series.values.reshape(-1, 1)
    x = np.arange(len(y)).reshape(-1, 1)
    
    # 線性回歸
    model = LinearRegression().fit(x, y)
    k = model.coef_[0].item()
    r2 = model.score(x, y)
    
    # 價格預測
    p_now = series.iloc[-1]
    p_1m = model.predict([[len(y) + 22]])[0].item()
    
    # 20EMA 狀態判定
    ema20 = series.ewm(span=20).mean().iloc[-1]
    
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

# --- 3. 六維波動防禦 ---
def calc_volatility_shells(series):
    window = 20
    rolling_mean = series.rolling(window).mean().iloc[-1]
    rolling_std = series.rolling(window).std().iloc[-1]
    curr_price = series.iloc[-1]
    
    levels = {}
    for i in range(1, 4):
        levels[f'H{i}'] = rolling_mean + (i * rolling_std)
        levels[f'L{i}'] = rolling_mean - (i * rolling_std)
        
    # 判斷當前位置
    pos_desc = "正常波動"
    if curr_price > levels['H2']: pos_desc = "⚠️ 情緒過熱 (H2)"
    if curr_price < levels['L2']: pos_desc = "💎 超賣機會 (L2)"
    
    return levels, pos_desc

# --- 4. 凱利公式 ---
def calc_kelly_position(trend_data):
    if not trend_data: return 0, 0

    # 簡單勝率估計
    win_rate = 0.55
    if trend_data['k'] > 0: win_rate += 0.05
    if trend_data['r2'] > 0.6: win_rate += 0.05
    if trend_data['status'] == "🛑 趨勢損毀": win_rate -= 0.15
    
    odds = 2.0 
    f_star = (odds * win_rate - (1 - win_rate)) / odds
    safe_kelly = max(0, f_star * 0.5) 
    
    return safe_kelly * 100, win_rate

# --- 5. 比特幣逃頂 ---
def check_pi_cycle(btc_series):
    if btc_series.empty: return False, 0, 0, 0
    
    ma111 = btc_series.rolling(111).mean().iloc[-1]
    ma350_x2 = btc_series.rolling(350).mean().iloc[-1] * 2
    
    signal = ma111 > ma350_x2
    dist = (ma350_x2 - ma111) / ma111 
    
    return signal, ma111, ma350_x2, dist

# --- 6. 繪圖模組 ---
def plot_kline_chart(ticker, df_close, df_open, df_high, df_low):
    if ticker not in df_close.columns: return None
    try:
        lookback = 120 # 顯示過去半年
        dates = df_close.index[-lookback:]
        
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

# --- 7. 輸入解析 ---
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
    st.caption("v13.0 防撞版 | 修復 Duplicate Element ID 錯誤")
    st.markdown("---")

    # --- 側邊欄 ---
    with st.sidebar:
        st.header("⚙️ 資產配置輸入")
        st.caption("格式：代號, 持倉金額")
        default_input = """BTC-USD, 50000
QQQ, 30000
BNSOL-USD, 15000
0050.TW, 20000
NVDA, 10000"""
        user_input = st.text_area("持倉清單", default_input, height=200)
        portfolio_dict = parse_input(user_input)
        tickers_list = list(portfolio_dict.keys())
        total_value = sum(portfolio_dict.values())
        st.metric("總資產估值 (Est.)", f"${total_value:,.0f}")
        
        if st.button("🚀 啟動量化審計", type="primary"):
            st.session_state['run_analysis'] = True
        
    if not st.session_state.get('run_analysis', False):
        st.info("👈 請在左側輸入您的持倉，並點擊『啟動量化審計』。")
        return

    # 下載數據
    with st.spinner("Alpha 正在下載 K 線數據..."):
        df_close, df_open, df_high, df_low = fetch_data(tickers_list)
            
    if df_close.empty:
        st.error("數據獲取失敗，請檢查代號。")
        return

    # --- A. 宏觀戰情室 ---
    st.subheader("1. 宏觀戰情室 (Macro Audit)")
    col1, col2, col3 = st.columns(3)
    
    # BTC Pi Cycle
    if 'BTC-USD' in df_close.columns:
        pi_sig, ma111, ma350x2, dist = check_pi_cycle(df_close['BTC-USD'])
        btc_price = df_close['BTC-USD'].iloc[-1]
        with col1:
            st.markdown("#### ₿ 比特幣逃頂指標")
            st.metric("BTC 現價", f"${btc_price:,.0f}")
            if pi_sig: st.error("🚨 逃頂信號已觸發!")
            else: st.success(f"✅ 安全 (距離交叉: {dist:.1%})")
            st.caption(f"111DMA: {ma111:,.0f} | 350DMAx2: {ma350x2:,.0f}")

    # QQQ 趨勢
    if 'QQQ' in df_close.columns:
        q_trend = analyze_trend(df_close['QQQ'])
        with col2:
            st.markdown("#### 🇺🇸 美股大盤 (QQQ)")
            st.metric("趨勢狀態", q_trend['status'], delta=f"斜率: {q_trend['k']:.2f}")
            st.caption(f"R2 (趨勢純度): {q_trend['r2']:.2f}")

    # 槓桿對標
    if 'TQQQ' in df_close.columns and 'QQQ' in df_close.columns:
        ret_q = df_close['QQQ'].pct_change().sum()
        ret_tq = df_close['TQQQ'].pct_change().sum()
        with col3:
            st.markdown("#### ⚡ 槓桿效率")
            st.metric("TQQQ/QQQ 彈性", f"{ret_tq/ret_q:.2f}x")
            if ret_tq/ret_q < 2.5: st.warning("⚠️ 槓桿損耗過大")
            else: st.success("⚡ 槓桿效率優良")

    st.markdown("---")
    st.markdown("#### 🇺🇸 美國大盤基準 K 線")
    
    # [關鍵修復]：加入 unique key 防止 ID 衝突
    b_col1, b_col2, b_col3 = st.columns(3)
    benchmarks = ['QQQ', 'QLD', 'TQQQ']
    for i, b_ticker in enumerate(benchmarks):
        with [b_col1, b_col2, b_col3][i]:
            if b_ticker in df_close.columns:
                fig = plot_kline_chart(b_ticker, df_close, df_open, df_high, df_low)
                if fig: 
                    # 這裡加上了 key=f"bench_{b_ticker}"，這是修復的關鍵！
                    st.plotly_chart(fig, use_container_width=True, key=f"bench_{b_ticker}")

    st.markdown("---")

    # --- B. 資產整合總表 ---
    st.subheader("2. 資產整合總表 (Portfolio Overview)")
    table_data = []
    
    for ticker in tickers_list:
        if ticker not in df_close.columns: continue
        
        trend = analyze_trend(df_close[ticker])
        if not trend: continue
        levels, vol_status = calc_volatility_shells(df_close[ticker])
        kelly_pct, win_prob = calc_kelly_position(trend)
        
        current_val = portfolio_dict.get(ticker, 0)
        weight = (current_val / total_value) if total_value > 0 else 0
        
        action = "持有"
        if trend['status'] == "🛑 趨勢損毀": action = "減倉/止損"
        elif vol_status == "💎 超賣機會 (L2)": action = "加倉/抄底"
        elif vol_status == "⚠️ 情緒過熱 (H2)": action = "止盈觀察"

        table_data.append({
            "代號": ticker,
            "持倉價值": f"${current_val:,.0f}",
            "權重": f"{weight:.1%}",
            "現價": f"${trend['p_now']:.2f}",
            "趨勢": trend['status'],
            "1個月預測": f"${trend['p_1m']:.2f}",
            "凱利建議倉位": f"{kelly_pct:.1f}%",
            "六維狀態": vol_status,
            "建議": action
        })
    
    p_col1, p_col2 = st.columns([2, 1])
    with p_col1:
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)
    with p_col2:
        if total_value > 0:
            pie_df = pd.DataFrame(list(portfolio_dict.items()), columns=['Ticker', 'Value'])
            fig = px.pie(pie_df, values='Value', names='Ticker', title='資產配置', hole=0.4)
            fig.update_layout(margin=dict(t=30, b=0, l=0, r=0), height=300)
            # 這裡也加個 key 保險
            st.plotly_chart(fig, use_container_width=True, key="portfolio_pie")

    st.markdown("---")

    # --- C. 持倉 K 線深度審計 ---
    st.subheader("3. 持倉 K 線深度審計 (Deep Dive)")
    
    for ticker in tickers_list:
        if ticker not in df_close.columns: continue
        trend = analyze_trend(df_close[ticker])
        if not trend: continue
        
        with st.expander(f"📊 {ticker} - {trend['status']} (點擊展開 K 線圖)", expanded=True):
            k_col1, k_col2 = st.columns([3, 1])
            
            with k_col1:
                fig = plot_kline_chart(ticker, df_close, df_open, df_high, df_low)
                if fig: 
                    # [關鍵修復]：這裡加上 key=f"deep_{ticker}"，防止跟上面的圖撞車！
                    st.plotly_chart(fig, use_container_width=True, key=f"deep_{ticker}")
                
            with k_col2:
                st.markdown("#### 六維數據")
                levels, vol_status = calc_volatility_shells(df_close[ticker])
                st.caption(f"H2 (壓力): {levels.get('H2', 0):.2f}")
                st.info(f"現價: {trend['p_now']:.2f}")
                st.caption(f"L2 (支撐): {levels.get('L2', 0):.2f}")
                
                st.divider()
                st.markdown("#### Alpha 預測")
                st.metric("1個月目標", f"${trend['p_1m']:.2f}", delta=f"{(trend['p_1m']-trend['p_now'])/trend['p_now']:.1%}")

if __name__ == "__main__":
    main()