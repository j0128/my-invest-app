import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go

# --- 0. 全局設定 ---
st.set_page_config(page_title="Alpha 2.0 Pro: 決策金字塔", layout="wide", page_icon="🏛️")

# 自定義 CSS
st.markdown("""
<style>
    .metric-card {background-color: #0E1117; border: 1px solid #262730; border-radius: 5px; padding: 15px; color: white;}
    .bullish {color: #00FF7F; font-weight: bold;}
    .bearish {color: #FF4B4B; font-weight: bold;}
    .neutral {color: #FFD700; font-weight: bold;}
    .risk-box {background-color: #2D0000; padding: 10px; border-radius: 5px; border-left: 5px solid #FF4B4B;}
    .safe-box {background-color: #002D00; padding: 10px; border-radius: 5px; border-left: 5px solid #00FF7F;}
</style>
""", unsafe_allow_html=True)

# --- 1. 核心數據引擎 (OHLC + 宏觀數據) ---
@st.cache_data(ttl=3600)
def fetch_data(tickers):
    """
    下載個股、基準、以及宏觀指標 (VIX, TNX)
    """
    benchmarks = ['QQQ', 'QLD', 'TQQQ', 'BTC-USD', '^VIX', '^TNX'] # 加入 VIX 和 債券殖利率
    all_tickers = list(set(tickers + benchmarks))
    
    dict_close = {}
    dict_open = {}
    dict_high = {}
    dict_low = {}
    
    progress_bar = st.progress(0, text="Alpha 正在建立宏觀數據連線...")
    
    for i, t in enumerate(all_tickers):
        try:
            progress_bar.progress((i + 1) / len(all_tickers), text=f"正在下載: {t} ...")
            # 抓取 2 年數據以計算 200SMA
            df = yf.Ticker(t).history(period="2y", auto_adjust=True)
            
            if df.empty: continue
                
            dict_close[t] = df['Close']
            dict_open[t] = df['Open']
            dict_high[t] = df['High']
            dict_low[t] = df['Low']
            
        except Exception:
            continue
            
    progress_bar.empty()
    return (pd.DataFrame(dict_close).ffill(), 
            pd.DataFrame(dict_open).ffill(), 
            pd.DataFrame(dict_high).ffill(), 
            pd.DataFrame(dict_low).ffill())

# --- 2. 獲取基本面估值 (Layer 0) ---
@st.cache_data(ttl=3600*12) # 估值不用常變，12小時更新一次
def get_valuation_metrics(ticker):
    try:
        info = yf.Ticker(ticker).info
        fwd_pe = info.get('forwardPE', None)
        return fwd_pe
    except:
        return None

# --- 3. 核心趨勢模組 (含 200SMA) ---
def analyze_trend(series):
    if series is None: return None
    series = series.dropna()
    if series.empty or len(series) < 200: return None # 需要足夠數據算年線

    y = series.values.reshape(-1, 1)
    x = np.arange(len(y)).reshape(-1, 1)
    
    # 線性回歸
    model = LinearRegression().fit(x, y)
    k = model.coef_[0].item()
    r2 = model.score(x, y)
    
    p_now = series.iloc[-1].item()
    p_1m = model.predict([[len(y) + 22]])[0].item()
    
    # 指標計算
    ema20 = series.ewm(span=20).mean().iloc[-1].item()
    sma200 = series.rolling(200).mean().iloc[-1].item() # 長期趨勢線 (牛熊分界)
    
    # 狀態判定邏輯
    status = "🛡️ 區間盤整"
    color = "neutral"
    
    if p_now < sma200:
        status = "🛑 熊市防禦 (破年線)" # Layer 2: Trend Filter
        color = "bearish"
    elif p_now > ema20 and k > 0:
        status = "🔥 加速進攻"
        color = "bullish"
    elif p_now < ema20:
        status = "⚠️ 動能減弱"
        color = "neutral"
        
    return {
        "k": k, "r2": r2, "p_now": p_now, "p_1m": p_1m, 
        "ema20": ema20, "sma200": sma200, 
        "status": status, "color": color
    }

# --- 4. 六維波動防禦 ---
def calc_volatility_shells(series):
    if series is None or series.empty: return {}, "無數據"
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

# --- 5. 戰略檔位決策引擎 (The Gearbox) ---
def determine_strategy_gear(qqq_trend, vix_now, qqq_pe):
    """
    六層決策金字塔的核心邏輯
    """
    if not qqq_trend: return "N/A", "數據不足"
    
    price = qqq_trend['p_now']
    sma200 = qqq_trend['sma200']
    ema20 = qqq_trend['ema20']
    
    # 預設值處理
    vix = vix_now if vix_now else 20
    pe = qqq_pe if qqq_pe else 25 # 如果抓不到 PE，預設為 25 (中性)
    
    # --- Layer 2: 長期趨勢濾網 ---
    if price < sma200:
        return "檔位 0 (現金/避險)", "🛑 熊市訊號：價格跌破 200日均線。多頭禁入，強制防禦。"

    # --- Layer 0: 估值天花板 ---
    if pe > 32: # 歷史極端高位
        return "檔位 1 (QQQ)", "⚠️ 估值天花板：本益比過高 (>32)。禁止槓桿，僅持有現貨。"
    
    # --- Layer 3: 宏觀風險儀表 (VIX) ---
    if vix > 22:
        return "檔位 1 (QQQ)", "🌩️ 風暴警報：VIX > 22。市場恐慌，禁止槓桿。"
    
    # --- Layer 0 (Part 2): 合理估值 ---
    if pe > 28: # 稍微偏貴
        # 允許 QLD (2x) 但禁止 TQQQ
        if price > ema20:
            return "檔位 2 (QLD)", "⚖️ 估值偏高：本益比 > 28。限制最大 2倍槓桿。"
        else:
            return "檔位 1 (QQQ)", "📉 動能不足：雖在牛市但短期轉弱。"
            
    # --- Layer 4: 動能確認 (All Clear) ---
    if price > ema20:
        return "檔位 3 (TQQQ)", "🚀 完美風口：估值合理 + 趨勢向上 + 情緒穩定。允許 3倍槓桿。"
    else:
        return "檔位 2 (QLD)", "🛡️ 趨勢回調：牛市中的回檔。保持 2倍槓桿或觀望。"

# --- 6. 凱利公式 ---
def calc_kelly_position(trend_data):
    if not trend_data: return 0, 0
    win_rate = 0.55
    if trend_data['k'] > 0: win_rate += 0.05
    if trend_data['r2'] > 0.6: win_rate += 0.05
    if "熊市" in trend_data['status']: win_rate -= 0.2 # 熊市勝率大減
    
    odds = 2.0 
    f_star = (odds * win_rate - (1 - win_rate)) / odds
    safe_kelly = max(0, f_star * 0.5) 
    return safe_kelly * 100, win_rate

# --- 7. 比特幣逃頂 ---
def check_pi_cycle(btc_series):
    if btc_series.empty: return False, 0, 0, 0
    ma111 = btc_series.rolling(111).mean().iloc[-1]
    ma350_x2 = btc_series.rolling(350).mean().iloc[-1] * 2
    signal = ma111 > ma350_x2
    dist = (ma350_x2 - ma111) / ma111 
    return signal, ma111, ma350_x2, dist

# --- 8. 繪圖模組 ---
def plot_kline_chart(ticker, df_close, df_open, df_high, df_low, trend_data=None):
    if ticker not in df_close.columns: return None
    try:
        lookback = 250 # 看一年，才能看到 200SMA
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
        # K 線
        fig.add_trace(go.Candlestick(
            x=dates, open=opens, high=highs, low=lows, close=closes,
            name='Price', increasing_line_color='#00FF7F', decreasing_line_color='#FF4B4B'
        ))
        
        # 20EMA
        ema20 = df_close[ticker].ewm(span=20).mean().iloc[-len(dates):]
        fig.add_trace(go.Scatter(
            x=dates, y=ema20, mode='lines', name='20 EMA (短期)',
            line=dict(color='#FFD700', width=1.5)
        ))
        
        # 200SMA (年線) - 新增
        sma200 = df_close[ticker].rolling(200).mean().iloc[-len(dates):]
        fig.add_trace(go.Scatter(
            x=dates, y=sma200, mode='lines', name='200 SMA (牛熊線)',
            line=dict(color='#00BFFF', width=2.0, dash='dash')
        ))

        fig.update_layout(
            title=f"{ticker} - Daily Chart (含年線)", height=350, margin=dict(l=0, r=0, t=30, b=0),
            xaxis_rangeslider_visible=False, paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white')
        )
        return fig
    except:
        return None

# --- 9. 輸入解析 ---
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
    st.caption("v15.0 六層決策金字塔 | 防禦型 Alpha 核心")
    st.markdown("---")

    # --- 側邊欄 ---
    with st.sidebar:
        st.header("⚙️ 資產配置輸入")
        default_input = """BTC-USD, 70000
BNSOL-USD, 130000
ETH-USD, 10000
0050.TW, 95000
AMD, 65000
CLS, 15000
URA, 35000"""
        user_input = st.text_area("持倉清單", default_input, height=200)
        portfolio_dict = parse_input(user_input)
        tickers_list = list(portfolio_dict.keys())
        total_value = sum(portfolio_dict.values())
        st.metric("總資產估值 (Est.)", f"${total_value:,.0f}")
        
        if st.button("🚀 啟動量化審計", type="primary"):
            st.session_state['run_analysis'] = True
        
    if not st.session_state.get('run_analysis', False):
        st.info("👈 請點擊『啟動量化審計』開始分析。")
        return

    # 下載數據
    with st.spinner("Alpha 正在同步宏觀數據與股價..."):
        df_close, df_open, df_high, df_low = fetch_data(tickers_list)
        # 嘗試獲取 QQQ 估值
        qqq_pe = get_valuation_metrics('QQQ')
            
    if df_close.empty:
        st.error("數據獲取失敗。")
        return

    # --- A. 宏觀戰情室 (The War Room) ---
    st.subheader("1. 宏觀戰情室 (The War Room)")
    
    # 準備數據
    qqq_trend = analyze_trend(df_close.get('QQQ'))
    vix_series = df_close.get('^VIX')
    vix_now = vix_series.iloc[-1] if vix_series is not None and not vix_series.empty else None
    tnx_series = df_close.get('^TNX')
    tnx_now = tnx_series.iloc[-1] if tnx_series is not None and not tnx_series.empty else None
    
    # 決策引擎運算
    gear, reason = determine_strategy_gear(qqq_trend, vix_now, qqq_pe)
    
    # 顯示儀表板
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    
    with m_col1:
        st.metric("VIX 恐慌指數", f"{vix_now:.2f}" if vix_now else "N/A", 
                 delta="高風險 > 20" if vix_now and vix_now > 20 else "安全", 
                 delta_color="inverse")
    with m_col2:
        st.metric("10年期公債殖利率", f"{tnx_now:.2f}%" if tnx_now else "N/A")
    with m_col3:
        pe_display = f"{qqq_pe:.1f}" if qqq_pe else "N/A (預設25)"
        st.metric("QQQ 遠期本益比", pe_display, 
                 delta="昂貴 > 28" if qqq_pe and qqq_pe > 28 else "合理", 
                 delta_color="inverse")
    with m_col4:
        # 顯示 QQQ 200MA 狀態
        if qqq_trend:
            dist_sma = (qqq_trend['p_now'] - qqq_trend['sma200']) / qqq_trend['sma200']
            st.metric("QQQ vs 年線", f"{dist_sma:.1%}", "牛市區" if dist_sma>0 else "熊市區")

    # 顯示最終決策
    st.info(f"### 🤖 Alpha 戰略指令：{gear}")
    st.markdown(f"> **決策邏輯：** {reason}")

    st.markdown("---")
    st.markdown("#### 🇺🇸 美國大盤基準 K 線 (含年線)")
    
    b_col1, b_col2, b_col3 = st.columns(3)
    benchmarks = ['QQQ', 'QLD', 'TQQQ']
    for i, b_ticker in enumerate(benchmarks):
        with [b_col1, b_col2, b_col3][i]:
            if b_ticker in df_close.columns:
                fig = plot_kline_chart(b_ticker, df_close, df_open, df_high, df_low)
                if fig: st.plotly_chart(fig, use_container_width=True, key=f"bench_{b_ticker}")

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
        
        # Action Logic (加入年線判斷)
        action = "持有"
        if trend['p_now'] < trend['sma200']: action = "熊市避險/清倉"
        elif trend['status'] == "🛑 熊市防禦 (破年線)": action = "減倉/止損"
        elif vol_status == "💎 超賣機會 (L2)": action = "加倉/抄底"
        elif vol_status == "⚠️ 情緒過熱 (H2)": action = "止盈觀察"

        table_data.append({
            "代號": ticker,
            "權重": f"{weight:.1%}",
            "現價": f"${trend['p_now']:.2f}",
            "趨勢狀態": trend['status'],
            "1個月預測": f"${trend['p_1m']:.2f}",
            "年線乖離": f"{(trend['p_now']-trend['sma200'])/trend['sma200']:.1%}",
            "凱利建議": f"{kelly_pct:.1f}%",
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
                if fig: st.plotly_chart(fig, use_container_width=True, key=f"deep_{ticker}")
                
            with k_col2:
                st.markdown("#### 六維數據")
                levels, vol_status = calc_volatility_shells(df_close[ticker])
                st.caption(f"H2 (壓力): {levels.get('H2', 0):.2f}")
                st.info(f"現價: {trend['p_now']:.2f}")
                st.caption(f"L2 (支撐): {levels.get('L2', 0):.2f}")
                
                st.divider()
                st.markdown("#### 趨勢濾網")
                if trend['p_now'] > trend['sma200']:
                    st.success("✅ 位於年線 (200SMA) 之上，長多格局。")
                else:
                    st.error("🛑 跌破年線 (200SMA)，進入熊市防禦區。")

    st.markdown("---")

    # --- D. 六層決策金字塔說明書 ---
    st.header("4. 終極投資框架：六層決策金字塔 (The Decision Pyramid)")
    st.markdown("""
    本系統融合了「防禦型 Alpha」與「動態槓桿」哲學，旨在確保投資人在牛市賺取超額收益，並在熊市存活。
    """)

    with st.container():
        st.markdown("#### 🏰 第零層：估值天花板 (Valuation Ceiling)")
        st.info("規則：當市場過於昂貴 (Forward P/E > 28) 時，禁止使用槓桿 (TQQQ)。這是避免「均值回歸」殺傷力的核心防線。")
        
        st.markdown("#### 🌊 第二層：長期趨勢濾網 (The Trend Filter)")
        st.info("規則：200日均線 (SMA200) 是牛熊分界線。價格在年線之下 = 熊市，系統會強制建議「防禦/現金」，優先級高於所有短期指標。")
        
        st.markdown("#### 🌩️ 第三層：宏觀儀表 (Risk Dashboard)")
        st.info("規則：監控 VIX 恐慌指數。當 VIX > 22 時，代表市場進入「風暴模式」，此時應降檔減速，而非冒險。")

if __name__ == "__main__":
    main()