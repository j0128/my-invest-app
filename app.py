import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go

# --- 0. 全局設定 ---
st.set_page_config(page_title="Alpha 2.0 Pro: 戰略資產中控台", layout="wide", page_icon="🏛️")

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

# --- 1. 核心數據引擎 ---
@st.cache_data(ttl=3600)
def fetch_data(tickers):
    benchmarks = ['QQQ', 'QLD', 'TQQQ', 'BTC-USD', '^VIX', '^TNX', 'HYG'] 
    all_tickers = list(set(tickers + benchmarks))
    
    dict_close = {}
    dict_open = {}
    dict_high = {}
    dict_low = {}
    
    progress_bar = st.progress(0, text="Alpha 正在建立全市場連線...")
    
    for i, t in enumerate(all_tickers):
        try:
            progress_bar.progress((i + 1) / len(all_tickers), text=f"正在下載: {t} ...")
            df = yf.Ticker(t).history(period="2y", auto_adjust=True)
            if df.empty: continue
            dict_close[t] = df['Close']
            dict_open[t] = df['Open']
            dict_high[t] = df['High']
            dict_low[t] = df['Low']
        except: continue
            
    progress_bar.empty()
    return (pd.DataFrame(dict_close).ffill(), 
            pd.DataFrame(dict_open).ffill(), 
            pd.DataFrame(dict_high).ffill(), 
            pd.DataFrame(dict_low).ffill())

# --- 2. 估值 ---
@st.cache_data(ttl=3600*12)
def get_valuation_metrics(ticker):
    try:
        info = yf.Ticker(ticker).info
        return info.get('forwardPE', None)
    except: return None

# --- 3. 趨勢模組 (含乖離判斷) ---
def analyze_trend(series):
    if series is None: return None
    series = series.dropna()
    if series.empty or len(series) < 200: return None

    y = series.values.reshape(-1, 1)
    x = np.arange(len(y)).reshape(-1, 1)
    
    model = LinearRegression().fit(x, y)
    k = model.coef_[0].item()
    r2 = model.score(x, y)
    
    p_now = series.iloc[-1].item()
    p_1m = model.predict([[len(y) + 22]])[0].item()
    
    ema20 = series.ewm(span=20).mean().iloc[-1].item()
    sma200 = series.rolling(200).mean().iloc[-1].item()
    
    status = "🛡️ 區間盤整"
    color = "neutral"
    
    if p_now < sma200:
        status = "🛑 熊市防禦 (破年線)"
        color = "bearish"
    elif p_now > ema20 and k > 0:
        status = "🔥 加速進攻"
        color = "bullish"
    elif p_now < ema20:
        status = "⚠️ 動能減弱"
        color = "neutral"
        
    # [新增] 乖離率過大判斷
    is_overheated = False
    if k > 0 and p_1m < p_now:
        is_overheated = True
        
    return {
        "k": k, "r2": r2, "p_now": p_now, "p_1m": p_1m, 
        "ema20": ema20, "sma200": sma200, 
        "status": status, "color": color,
        "is_overheated": is_overheated # 回傳乖離狀態
    }

# --- 4. 六維波動 ---
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
    except: return {}, "計算錯誤"

# --- 5. 決策引擎 ---
def determine_strategy_gear(qqq_trend, vix_now, qqq_pe, hyg_trend):
    if not qqq_trend: return "N/A", "數據不足"
    price = qqq_trend['p_now']
    sma200 = qqq_trend['sma200']
    ema20 = qqq_trend['ema20']
    vix = vix_now if vix_now else 20
    pe = qqq_pe if qqq_pe else 25 
    
    if hyg_trend and hyg_trend['p_now'] < hyg_trend['sma200']:
        return "檔位 0 (現金/避險)", "💧 流動性枯竭：HYG 跌破年線，強制防禦。"
    if price < sma200:
        return "檔位 0 (現金/避險)", "🛑 熊市訊號：QQQ 跌破年線，多頭禁入。"
    if pe > 32:
        return "檔位 1 (QQQ)", "⚠️ 估值天花板：PE > 32，禁止槓桿。"
    if vix > 22:
        return "檔位 1 (QQQ)", "🌩️ 風暴警報：VIX > 22，禁止槓桿。"
    if pe > 28:
        if price > ema20: return "檔位 2 (QLD)", "⚖️ 估值偏高：限制 2倍槓桿。"
        else: return "檔位 1 (QQQ)", "📉 動能不足：短期轉弱。"
    if price > ema20:
        return "檔位 3 (TQQQ)", "🚀 完美風口：流動性足 + 估值合理 + 趨勢向上。"
    else:
        return "檔位 2 (QLD)", "🛡️ 趨勢回調：牛市回檔，保持 2倍。"

# --- 6. 凱利公式 ---
def calc_kelly_position(trend_data):
    if not trend_data: return 0, 0
    win_rate = 0.55
    if trend_data['k'] > 0: win_rate += 0.05
    if trend_data['r2'] > 0.6: win_rate += 0.05
    if "熊市" in trend_data['status']: win_rate -= 0.2
    f_star = (2.0 * win_rate - (1 - win_rate)) / 2.0
    return max(0, f_star * 0.5) * 100, win_rate

# --- 7. 比特幣逃頂 ---
def check_pi_cycle(btc_series):
    if btc_series.empty: return False, 0, 0, 0
    ma111 = btc_series.rolling(111).mean().iloc[-1]
    ma350_x2 = btc_series.rolling(350).mean().iloc[-1] * 2
    return ma111 > ma350_x2, ma111, ma350_x2, (ma350_x2 - ma111) / ma111

# --- 8. 繪圖 ---
def plot_kline_chart(ticker, df_close, df_open, df_high, df_low):
    if ticker not in df_close.columns: return None
    try:
        lookback = 250
        dates = df_close.index[-lookback:]
        def get_s(df, t): return df[t].iloc[-len(dates):] if t in df.columns else pd.Series()
        
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=dates, open=get_s(df_open, ticker), high=get_s(df_high, ticker), 
                                     low=get_s(df_low, ticker), close=get_s(df_close, ticker), name='Price',
                                     increasing_line_color='#00FF7F', decreasing_line_color='#FF4B4B'))
        fig.add_trace(go.Scatter(x=dates, y=df_close[ticker].ewm(span=20).mean().iloc[-len(dates):], 
                                 mode='lines', name='20 EMA', line=dict(color='#FFD700', width=1.5)))
        fig.add_trace(go.Scatter(x=dates, y=df_close[ticker].rolling(200).mean().iloc[-len(dates):], 
                                 mode='lines', name='200 SMA', line=dict(color='#00BFFF', width=2.0, dash='dash')))
        fig.update_layout(title=f"{ticker} - Daily Chart", height=350, margin=dict(l=0, r=0, t=30, b=0),
                          xaxis_rangeslider_visible=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        return fig
    except: return None

# --- 9. 輸入解析 ---
def parse_input(text):
    port = {}
    for line in text.strip().split('\n'):
        if ',' in line:
            parts = line.split(',')
            try: port[parts[0].strip().upper()] = float(parts[1].strip())
            except: port[parts[0].strip().upper()] = 0.0
    return port

# --- MAIN ---
def main():
    st.title("Alpha 2.0 Pro: 戰略資產中控台")
    st.caption("v17.0 乖離警示版 | 新增乖離率偵測與紅字警告")
    st.markdown("---")

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
        if st.button("🚀 啟動量化審計", type="primary"): st.session_state['run_analysis'] = True
        
    if not st.session_state.get('run_analysis', False):
        st.info("👈 請點擊『啟動量化審計』。")
        return

    with st.spinner("Alpha 正在同步全市場數據..."):
        df_close, df_open, df_high, df_low = fetch_data(tickers_list)
        qqq_pe = get_valuation_metrics('QQQ')
            
    if df_close.empty:
        st.error("數據獲取失敗。"); return

    # --- A. 宏觀 ---
    st.subheader("1. 宏觀戰情室")
    qqq_trend = analyze_trend(df_close.get('QQQ'))
    hyg_trend = analyze_trend(df_close.get('HYG'))
    vix = df_close.get('^VIX').iloc[-1] if '^VIX' in df_close else None
    gear, reason = determine_strategy_gear(qqq_trend, vix, qqq_pe, hyg_trend)
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("VIX", f"{vix:.2f}" if vix else "N/A", delta="高風險" if vix and vix>22 else "安全", delta_color="inverse")
    with c2: 
        hyg_s = "充裕" if hyg_trend and hyg_trend['p_now'] > hyg_trend['sma200'] else "枯竭"
        st.metric("流動性 (HYG)", hyg_s, delta="風險高" if hyg_s=="枯竭" else "風險低", delta_color="inverse")
    with c3: st.metric("QQQ P/E", f"{qqq_pe:.1f}" if qqq_pe else "N/A", delta="昂貴" if qqq_pe and qqq_pe>28 else "合理", delta_color="inverse")
    with c4: st.metric("Alpha 指令", gear)
    
    if "熊市" in gear or "枯竭" in gear: st.error(f"決策：{reason}")
    else: st.success(f"決策：{reason}")
    
    st.markdown("---")
    
    # --- B. 總表 (含乖離警示) ---
    st.subheader("2. 資產整合總表")
    table_data = []
    for ticker in tickers_list:
        if ticker not in df_close.columns: continue
        trend = analyze_trend(df_close[ticker])
        if not trend: continue
        
        levels, vol_status = calc_volatility_shells(df_close[ticker])
        kelly_pct, _ = calc_kelly_position(trend)
        current_val = portfolio_dict.get(ticker, 0)
        weight = (current_val / total_value) if total_value > 0 else 0
        
        # [新增] 乖離過大判斷邏輯
        action = "持有"
        # 1. 優先處理年線 (最大風險)
        if trend['p_now'] < trend['sma200']: action = "熊市避險"
        # 2. 處理乖離 (漲太快)
        elif trend['is_overheated']: action = "⚠️ 乖離過大 (止盈)"
        # 3. 處理波動
        elif vol_status == "💎 超賣機會 (L2)": action = "加倉/抄底"
        elif vol_status == "⚠️ 情緒過熱 (H2)": action = "止盈觀察"

        table_data.append({
            "代號": ticker,
            "權重": f"{weight:.1%}",
            "現價": f"${trend['p_now']:.2f}",
            "趨勢": trend['status'],
            "1個月預測": f"${trend['p_1m']:.2f}",
            "乖離警示": "🔥 過熱" if trend['is_overheated'] else "正常", # 新增欄位
            "六維狀態": vol_status,
            "建議": action
        })
    
    c1, c2 = st.columns([2, 1])
    with c1: st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)
    with c2: 
        fig = px.pie(pd.DataFrame(list(portfolio_dict.items()), columns=['Ticker', 'Value']), values='Value', names='Ticker', title='配置', hole=0.4)
        fig.update_layout(margin=dict(t=30, b=0, l=0, r=0), height=300)
        st.plotly_chart(fig, use_container_width=True, key="pie")

    st.markdown("---")
    
    # --- C. 深度審計 ---
    st.subheader("3. 深度審計 (含乖離分析)")
    for ticker in tickers_list:
        if ticker not in df_close.columns: continue
        trend = analyze_trend(df_close[ticker])
        if not trend: continue
        
        with st.expander(f"📊 {ticker} - {trend['status']}", expanded=True):
            c1, c2 = st.columns([3, 1])
            with c1: 
                fig = plot_kline_chart(ticker, df_close, df_open, df_high, df_low)
                if fig: st.plotly_chart(fig, use_container_width=True, key=f"d_{ticker}")
            with c2:
                st.markdown("#### 關鍵數據")
                # 乖離警示
                if trend['is_overheated']:
                    st.warning(f"⚠️ **乖離過大**\n\n現價 ({trend['p_now']:.2f}) 已遠高於趨勢預測線 ({trend['p_1m']:.2f})。短線有回調壓力，不宜追高。")
                else:
                    st.info("✅ 價格與趨勢同步，健康上漲。")
                
                delta_val = (trend['p_1m']-trend['p_now'])/trend['p_now']
                st.metric("1個月目標", f"${trend['p_1m']:.2f}", delta=f"{delta_val:.1%}", 
                          delta_color="normal" if delta_val > 0 else "inverse") # 負數會變紅字

    st.markdown("---")
    st.header("4. 量化模型白皮書")
    st.info("**乖離率 (Deviation) 說明：** 當「趨勢向上」但「預測價格 < 現價」時，代表股價短期漲幅過大，脫離了統計學上的回歸中樞。這通常是「短線過熱」的訊號，建議止盈或等待回調，而非追價。")

if __name__ == "__main__":
    main()