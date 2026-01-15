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

# --- 1. 核心數據引擎 (Data Engine - 防彈版) ---
@st.cache_data(ttl=3600)
def fetch_data(tickers):
    """
    下載數據並強制標準化格式：
    回傳的 df_close 等一定是 DataFrame，且 Columns 為 Ticker 名稱。
    """
    benchmarks = ['QQQ', 'QLD', 'TQQQ', 'BTC-USD']
    all_tickers = list(set(tickers + benchmarks))
    
    try:
        # 下載過去 1 年數據，強制 progress=False 避免輸出干擾
        data = yf.download(all_tickers, period="1y", auto_adjust=True, progress=False)
        
        if data.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        # [關鍵修復] 處理 yfinance 不同版本的 MultiIndex 結構
        # 目標：提取出 Price 層級，並確保 Columns 是 Ticker
        def extract_price_type(data, price_col_name):
            # 情況 A: MultiIndex (Price, Ticker) -> 標準情況
            if isinstance(data.columns, pd.MultiIndex):
                try:
                    # 嘗試提取指定價格層級 (Close/Open...)
                    return data.xs(price_col_name, axis=1, level=0)
                except KeyError:
                    # 某些版本 yfinance 如果只有一個 ticker 但仍是 MultiIndex
                    return data[price_col_name]
            # 情況 B: Single Index (Date, Open, Close...) -> 單一 Ticker
            else:
                # 為了統一格式，我們必須把它轉成以 Ticker 為 Column 的 DataFrame
                # 如果只有一個 Ticker，all_tickers[0] 就是它
                single_df = data[[price_col_name]].copy()
                single_df.columns = all_tickers # 強制命名為 Ticker
                return single_df

        # 根據 auto_adjust=True，yfinance 通常回傳 'Close' (其實是 Adj Close)
        # 若有 'Adj Close' 則優先使用，否則用 'Close'
        close_col = 'Adj Close' if 'Adj Close' in data.columns.levels[0] else 'Close' if isinstance(data.columns, pd.MultiIndex) else 'Close'
        
        df_close = extract_price_type(data, close_col)
        df_open  = extract_price_type(data, 'Open')
        df_high  = extract_price_type(data, 'High')
        df_low   = extract_price_type(data, 'Low')

        # 資料補全，避免 NaN 造成計算崩潰
        return df_close.ffill(), df_open.ffill(), df_high.ffill(), df_low.ffill()

    except Exception as e:
        st.error(f"數據下載發生嚴重錯誤: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# --- 2. 核心趨勢模組 (Trend Projection - 純標量版) ---
def analyze_trend(series):
    # 確保輸入是乾淨的 Series
    series = series.dropna()
    if series.empty: return None

    y = series.values.reshape(-1, 1)
    x = np.arange(len(y)).reshape(-1, 1)
    
    model = LinearRegression().fit(x, y)
    
    # [關鍵修復] 使用 .item() 強制轉為 Python 原生 float
    k = model.coef_[0].item()
    r2 = model.score(x, y).item()
    
    # 提取數值 (Scalar)
    p_now = series.iloc[-1].item()
    p_1m = model.predict([[len(y) + 22]])[0].item()
    
    ema20 = series.ewm(span=20).mean().iloc[-1].item()
    
    # 純標量比較，絕不會報錯
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

# --- 3. 六維波動防禦 (Volatility Shells - 純標量版) ---
def calc_volatility_shells(series):
    series = series.dropna()
    if series.empty: return {}, "無數據"
    
    window = 20
    # [關鍵修復] .item()
    rolling_mean = series.rolling(window).mean().iloc[-1].item()
    rolling_std = series.rolling(window).std().iloc[-1].item()
    curr_price = series.iloc[-1].item()
    
    levels = {}
    for i in range(1, 4):
        levels[f'H{i}'] = rolling_mean + (i * rolling_std)
        levels[f'L{i}'] = rolling_mean - (i * rolling_std)
        
    pos_desc = "正常波動"
    if curr_price > levels['H2']: pos_desc = "⚠️ 情緒過熱 (H2)"
    if curr_price < levels['L2']: pos_desc = "💎 超賣機會 (L2)"
    
    return levels, pos_desc

# --- 4. 凱利公式 (Kelly) ---
def calc_kelly_position(trend_data):
    if not trend_data: return 0, 0
    win_rate = 0.55
    if trend_data['k'] > 0: win_rate += 0.05
    if trend_data['r2'] > 0.6: win_rate += 0.05
    if trend_data['status'] == "🛑 趨勢損毀": win_rate -= 0.15
    odds = 2.0 
    f_star = (odds * win_rate - (1 - win_rate)) / odds
    safe_kelly = max(0, f_star * 0.5) 
    return safe_kelly * 100, win_rate

# --- 5. 繪圖模組 (Plotly K-Line) ---
def plot_kline_chart(ticker, df_close, df_open, df_high, df_low):
    if ticker not in df_close.columns: return None
    
    lookback = 120
    # 確保只取該 Ticker 的數據
    try:
        dates = df_close.index[-lookback:]
        opens = df_open[ticker].iloc[-lookback:]
        highs = df_high[ticker].iloc[-lookback:]
        lows = df_low[ticker].iloc[-lookback:]
        closes = df_close[ticker].iloc[-lookback:]
        ema20 = df_close[ticker].ewm(span=20).mean().iloc[-lookback:]
    except KeyError:
        return None

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=dates, open=opens, high=highs, low=lows, close=closes,
        name='Price',
        increasing_line_color='#00FF7F', decreasing_line_color='#FF4B4B'
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=ema20, mode='lines', name='20 EMA',
        line=dict(color='#FFD700', width=1.5)
    ))

    fig.update_layout(
        title=f"{ticker} - Daily Chart",
        height=350,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_rangeslider_visible=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    return fig

# --- 6. 績效對比圖 (ROI) ---
def plot_comparison(tickers, df_close):
    lookback = 120 
    # 只取存在的 Columns
    valid_tickers = [t for t in tickers if t in df_close.columns]
    if not valid_tickers: return None
    
    df_slice = df_close[valid_tickers].iloc[-lookback:].copy()
    # 正規化
    df_norm = (df_slice / df_slice.iloc[0]) - 1
    
    fig = px.line(df_norm, x=df_norm.index, y=df_norm.columns, 
                  title="🔥 強弱對決：累積報酬率 (近120天)",
                  labels={'value': 'ROI', 'variable': 'Ticker'})
    
    fig.update_layout(
        height=400,
        hovermode="x unified",
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        legend=dict(orientation="h", y=1.1)
    )
    return fig

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
    st.caption("v8.0 終極修復版 | 全標量運算核心")
    st.markdown("---")

    # 側邊欄
    with st.sidebar:
        st.header("⚙️ 資產配置輸入")
        st.caption("格式：代號, 持倉金額")
        default_input = """BTC-USD, 50000
QQQ, 30000
AMD, 15000
NVDA, 10000"""
        user_input = st.text_area("持倉清單", default_input, height=200)
        portfolio_dict = parse_input(user_input)
        tickers_list = list(portfolio_dict.keys())
        total_value = sum(portfolio_dict.values())
        st.metric("總資產估值 (Est.)", f"${total_value:,.0f}")
        
        if st.button("🚀 啟動量化審計", type="primary"):
            st.session_state['run_analysis'] = True
        
    if not st.session_state.get('run_analysis', False):
        st.info("👈 請輸入持倉並點擊『啟動量化審計』。")
        return

    with st.spinner("Alpha 正在下載 K 線數據並計算模型..."):
        df_close, df_open, df_high, df_low = fetch_data(tickers_list)
            
    if df_close.empty:
        st.error("無法獲取數據，請檢查輸入的代號是否正確，或網路連線是否正常。")
        return

    # --- A. 績效對比實驗室 ---
    st.subheader("1. 績效對比實驗室 (Benchmark Lab)")
    compare_list = ['QQQ', 'QLD', 'TQQQ'] + tickers_list[:3]
    compare_list = list(set(compare_list))
    
    comp_fig = plot_comparison(compare_list, df_close)
    if comp_fig:
        st.plotly_chart(comp_fig, use_container_width=True)
    
    # 基準 K 線圖
    st.markdown("#### 🇺🇸 美國大盤基準 (Market Context)")
    b_col1, b_col2, b_col3 = st.columns(3)
    benchmarks = ['QQQ', 'QLD', 'TQQQ']
    
    for i, b_ticker in enumerate(benchmarks):
        col = [b_col1, b_col2, b_col3][i]
        with col:
            if b_ticker in df_close.columns:
                trend = analyze_trend(df_close[b_ticker])
                if trend:
                    st.markdown(f"**{b_ticker}** <span style='font-size:0.8em' class='{trend['color']}'>({trend['status']})</span>", unsafe_allow_html=True)
                    fig = plot_kline_chart(b_ticker, df_close, df_open, df_high, df_low)
                    if fig: st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- B. 資產整合總表 ---
    st.subheader("2. 資產整合總表")
    table_data = []
    for ticker in tickers_list:
        if ticker not in df_close.columns: continue
        trend = analyze_trend(df_close[ticker])
        if not trend: continue # 跳過無數據的
        
        levels, vol_status = calc_volatility_shells(df_close[ticker])
        kelly_pct, win_prob = calc_kelly_position(trend)
        current_val = portfolio_dict.get(ticker, 0)
        weight = (current_val / total_value) if total_value > 0 else 0
        
        action = "持有"
        if trend['status'] == "🛑 趨勢損毀": action = "減倉"
        elif vol_status == "💎 超賣機會 (L2)": action = "加倉"
        elif vol_status == "⚠️ 情緒過熱 (H2)": action = "止盈"

        table_data.append({
            "代號": ticker,
            "權重": f"{weight:.1%}",
            "現價": f"${trend['p_now']:.2f}",
            "趨勢": trend['status'],
            "1M 預測": f"${trend['p_1m']:.2f}",
            "凱利倉位": f"{kelly_pct:.1f}%",
            "六維狀態": vol_status,
            "建議": action
        })
    
    t_col1, t_col2 = st.columns([2, 1])
    with t_col1:
        if table_data:
            st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)
        else:
            st.warning("無有效資產數據可顯示")
            
    with t_col2:
        if total_value > 0:
            pie_df = pd.DataFrame(list(portfolio_dict.items()), columns=['Ticker', 'Value'])
            fig = px.pie(pie_df, values='Value', names='Ticker', hole=0.4)
            fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=300)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- C. 持倉 K 線深度審計 ---
    st.subheader("3. 持倉 K 線深度審計 (Deep Dive)")
    for ticker in tickers_list:
        if ticker not in df_close.columns: continue
        trend = analyze_trend(df_close[ticker])
        if not trend: continue
        
        with st.expander(f"📊 {ticker} - {trend['status']}", expanded=True):
            k_col1, k_col2 = st.columns([3, 1])
            with k_col1:
                fig = plot_kline_chart(ticker, df_close, df_open, df_high, df_low)
                if fig: st.plotly_chart(fig, use_container_width=True)
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