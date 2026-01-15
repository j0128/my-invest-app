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
    .warning {color: #FFA500; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# --- 1. 數據引擎 ---
@st.cache_data(ttl=3600)
def fetch_data(tickers):
    benchmarks = ['QQQ', 'QLD', 'TQQQ', 'BTC-USD']
    all_tickers = list(set(tickers + benchmarks))
    try:
        data = yf.download(all_tickers, period="1y", auto_adjust=True)
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
        st.error(f"數據下載失敗: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# --- 2. 趨勢與質性判斷模組 (Qualitative Logic) ---
def analyze_trend(series):
    if series.isnull().all(): return None

    y = series.values.reshape(-1, 1)
    x = np.arange(len(y)).reshape(-1, 1)
    
    # 線性回歸
    model = LinearRegression().fit(x, y)
    k = model.coef_[0].item()
    r2 = model.score(x, y)
    
    p_now = series.iloc[-1]
    p_1m = model.predict([[len(y) + 22]])[0].item() # 1個月後預測
    
    ema20 = series.ewm(span=20).mean().iloc[-1]
    
    # --- 質性判斷邏輯 (The Logic Fix) ---
    status = "盤整"
    color = "neutral"
    verdict = "觀望" # 新增：質性評語
    
    if p_now > ema20: # 價格在均線之上
        if k > 0: # 斜率向上
            status = "🔥 加速進攻"
            # 判斷是否過熱 (預測值竟然比現價低，代表現價飛太遠)
            if p_1m < p_now:
                verdict = "⚠️ 短線過熱 (乖離大)"
                color = "warning"
            else:
                verdict = "🚀 強勢上攻 (健康)"
                color = "bullish"
        else: # 斜率向下但價格在均線上
            status = "🛡️ 反彈測試"
            verdict = "⚡ 逆勢反彈"
            color = "neutral"
    else: # 價格在均線之下
        if k < 0:
            status = "❄️ 弱勢下跌"
            verdict = "📉 趨勢向下"
            color = "bearish"
        else:
            status = "🛑 趨勢回調"
            verdict = "💎 拉回測底" # 趨勢向上但價格跌破均線
            color = "warning"
        
    return {
        "k": k, "r2": r2, "p_now": p_now, "p_1m": p_1m, 
        "ema20": ema20, "status": status, "color": color, 
        "verdict": verdict # 回傳評語
    }

# --- 3. 六維波動 ---
def calc_volatility_shells(series):
    window = 20
    rolling_mean = series.rolling(window).mean().iloc[-1]
    rolling_std = series.rolling(window).std().iloc[-1]
    curr_price = series.iloc[-1]
    
    levels = {}
    for i in range(1, 4):
        levels[f'H{i}'] = rolling_mean + (i * rolling_std)
        levels[f'L{i}'] = rolling_mean - (i * rolling_std)
        
    pos_desc = "正常波動"
    if curr_price > levels['H2']: pos_desc = "⚠️ H2 (高風險區)"
    if curr_price < levels['L2']: pos_desc = "💎 L2 (超賣區)"
    
    return levels, pos_desc

# --- 4. 凱利公式 ---
def calc_kelly_position(trend_data):
    if not trend_data: return 0, 0
    win_rate = 0.55
    if trend_data['k'] > 0: win_rate += 0.05
    if trend_data['r2'] > 0.6: win_rate += 0.05
    if "下跌" in trend_data['status'] or "損毀" in trend_data['status']: win_rate -= 0.15
    
    odds = 2.0 
    f_star = (odds * win_rate - (1 - win_rate)) / odds
    safe_kelly = max(0, f_star * 0.5) 
    return safe_kelly * 100, win_rate

# --- 5. 繪圖模組 ---
def plot_kline_chart(ticker, df_close, df_open, df_high, df_low):
    if ticker not in df_close.columns: return None
    lookback = 120
    dates = df_close.index[-lookback:]
    
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=dates, open=df_open[ticker].iloc[-lookback:], 
        high=df_high[ticker].iloc[-lookback:], 
        low=df_low[ticker].iloc[-lookback:], 
        close=df_close[ticker].iloc[-lookback:],
        name='Price'
    ))
    ema20 = df_close[ticker].ewm(span=20).mean().iloc[-lookback:]
    fig.add_trace(go.Scatter(x=dates, y=ema20, mode='lines', name='20 EMA', line=dict(color='#FFD700', width=1.5)))
    
    fig.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0), xaxis_rangeslider_visible=False,
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
    return fig

# --- 6. 績效對比 ---
def plot_comparison(tickers, df_close):
    lookback = 120
    df_slice = df_close[tickers].iloc[-lookback:].copy()
    df_norm = (df_slice / df_slice.iloc[0]) - 1
    fig = px.line(df_norm, x=df_norm.index, y=df_norm.columns, title="🔥 累積報酬率對決 (近120天)")
    fig.update_layout(height=400, hovermode="x unified", margin=dict(l=0, r=0, t=30, b=0),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
                      legend=dict(orientation="h", y=1.1))
    return fig

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
    st.markdown("---")

    with st.sidebar:
        st.header("⚙️ 資產配置輸入")
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
        st.info("👈 請輸入持倉並點擊『啟動量化審計』。")
        return

    with st.spinner("Alpha 正在計算模型與質性判斷..."):
        df_close, df_open, df_high, df_low = fetch_data(tickers_list)
            
    if df_close.empty:
        st.error("無法獲取數據。")
        return

    # --- 1. 績效實驗室 ---
    st.subheader("1. 績效對比實驗室 (Benchmark Lab)")
    compare_list = list(set(['QQQ', 'QLD', 'TQQQ'] + tickers_list[:3]))
    valid_compare = [t for t in compare_list if t in df_close.columns]
    st.plotly_chart(plot_comparison(valid_compare, df_close), use_container_width=True)
    
    st.markdown("#### 🇺🇸 美國大盤基準 (Market Context)")
    b_col1, b_col2, b_col3 = st.columns(3)
    benchmarks = ['QQQ', 'QLD', 'TQQQ']
    for i, b_ticker in enumerate(benchmarks):
        with [b_col1, b_col2, b_col3][i]:
            if b_ticker in df_close.columns:
                trend = analyze_trend(df_close[b_ticker])
                # 這裡顯示質性判斷 verdict
                st.markdown(f"**{b_ticker}** <span style='font-size:0.9em' class='{trend['color']}'>[{trend['verdict']}]</span>", unsafe_allow_html=True)
                fig = plot_kline_chart(b_ticker, df_close, df_open, df_high, df_low)
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- 2. 資產整合總表 (新增質性判斷欄位) ---
    st.subheader("2. 資產整合總表")
    table_data = []
    for ticker in tickers_list:
        if ticker not in df_close.columns: continue
        trend = analyze_trend(df_close[ticker])
        levels, vol_status = calc_volatility_shells(df_close[ticker])
        kelly_pct, win_prob = calc_kelly_position(trend)
        current_val = portfolio_dict.get(ticker, 0)
        weight = (current_val / total_value) if total_value > 0 else 0
        
        # Action Logic
        action = "持有"
        if "過熱" in trend['verdict'] or "H2" in vol_status: action = "止盈/觀望"
        elif "強勢" in trend['verdict'] and "正常" in vol_status: action = "續抱"
        elif "超賣" in vol_status or "反彈" in trend['verdict']: action = "關注/加倉"
        elif "向下" in trend['verdict']: action = "減倉/避險"

        table_data.append({
            "代號": ticker,
            "權重": f"{weight:.1%}",
            "現價": f"${trend['p_now']:.2f}",
            "AI 質性判斷": trend['verdict'],  # 新增這一欄
            "1M 預測": f"${trend['p_1m']:.2f}",
            "波動狀態": vol_status,
            "建議": action
        })
    
    t_col1, t_col2 = st.columns([2, 1])
    with t_col1:
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)
    with t_col2:
        if total_value > 0:
            pie_df = pd.DataFrame(list(portfolio_dict.items()), columns=['Ticker', 'Value'])
            fig = px.pie(pie_df, values='Value', names='Ticker', hole=0.4)
            fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=300)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- 3. 深度審計 ---
    st.subheader("3. 持倉 K 線深度審計")
    for ticker in tickers_list:
        if ticker not in df_close.columns: continue
        trend = analyze_trend(df_close[ticker])
        
        # 標題加上質性判斷，例如：[⚠️ 短線過熱]
        header_text = f"📊 {ticker} - {trend['status']} [{trend['verdict']}]"
        
        with st.expander(header_text, expanded=True):
            k_col1, k_col2 = st.columns([3, 1])
            with k_col1:
                fig = plot_kline_chart(ticker, df_close, df_open, df_high, df_low)
                st.plotly_chart(fig, use_container_width=True)
            with k_col2:
                st.markdown("#### 🔍 AI 數據解讀")
                st.info(f"現價: ${trend['p_now']:.2f}")
                st.metric("1個月目標", f"${trend['p_1m']:.2f}", delta=f"{(trend['p_1m']-trend['p_now'])/trend['p_now']:.1%}")
                
                # 特別解釋預測值
                if trend['p_1m'] < trend['p_now'] and trend['k'] > 0:
                    st.warning("⚠️ **注意：** 股價漲速快於趨勢線，預測值較低暗示有「均值回歸」的短期回調壓力。")
                
                st.divider()
                st.caption(f"支撐 (L2): {calc_volatility_shells(df_close[ticker])[0]['L2']:.2f}")
                st.caption(f"壓力 (H2): {calc_volatility_shells(df_close[ticker])[0]['H2']:.2f}")

if __name__ == "__main__":
    main()