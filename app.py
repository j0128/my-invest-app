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
    .risk-box {border-left: 5px solid #FF4B4B; background-color: #2D0000; padding: 10px;}
    .safe-box {border-left: 5px solid #00FF7F; background-color: #002D00; padding: 10px;}
</style>
""", unsafe_allow_html=True)

# --- 1. 核心數據引擎 (含流動性指標) ---
@st.cache_data(ttl=3600)
def fetch_data(tickers):
    """
    下載個股、基準、宏觀 (VIX, TNX) 以及 流動性指標 (HYG)
    """
    # 新增 HYG (高收益債) 作為流動性代理
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
@st.cache_data(ttl=3600*12)
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
def determine_strategy_gear(qqq_trend, vix_now, qqq_pe, hyg_trend):
    """
    六層決策金字塔 (含流動性檢查)
    """
    if not qqq_trend: return "N/A", "數據不足"
    
    price = qqq_trend['p_now']
    sma200 = qqq_trend['sma200']
    ema20 = qqq_trend['ema20']
    
    # 預設值
    vix = vix_now if vix_now else 20
    pe = qqq_pe if qqq_pe else 25 
    
    # 1. 流動性濾網 (Liquidity Filter) - 新增
    # 如果高收益債 (HYG) 跌破年線，代表市場資金正在枯竭
    if hyg_trend and hyg_trend['p_now'] < hyg_trend['sma200']:
        return "檔位 0 (現金/避險)", "💧 流動性枯竭：高收益債 (HYG) 跌破年線。信用市場發出警訊，強制防禦。"

    # 2. 長期趨勢濾網 (Trend Filter)
    if price < sma200:
        return "檔位 0 (現金/避險)", "🛑 熊市訊號：QQQ 跌破 200日均線。多頭禁入。"

    # 3. 估值天花板 (Valuation Ceiling)
    if pe > 32:
        return "檔位 1 (QQQ)", "⚠️ 估值天花板：本益比 > 32。禁止槓桿。"
    
    # 4. 宏觀風險儀表 (VIX)
    if vix > 22:
        return "檔位 1 (QQQ)", "🌩️ 風暴警報：VIX > 22。市場恐慌，禁止槓桿。"
    
    # 5. 合理估值檢查
    if pe > 28:
        if price > ema20:
            return "檔位 2 (QLD)", "⚖️ 估值偏高：限制最大 2倍槓桿。"
        else:
            return "檔位 1 (QQQ)", "📉 動能不足：雖在牛市但短期轉弱。"
            
    # 6. 動能確認 (All Clear)
    if price > ema20:
        return "檔位 3 (TQQQ)", "🚀 完美風口：流動性充足 + 估值合理 + 趨勢向上。允許 3倍槓桿。"
    else:
        return "檔位 2 (QLD)", "🛡️ 趨勢回調：牛市中的回檔。保持 2倍槓桿或觀望。"

# --- 6. 凱利公式 ---
def calc_kelly_position(trend_data):
    if not trend_data: return 0, 0
    win_rate = 0.55
    if trend_data['k'] > 0: win_rate += 0.05
    if trend_data['r2'] > 0.6: win_rate += 0.05
    if "熊市" in trend_data['status']: win_rate -= 0.2
    
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
        lookback = 250
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
        
        sma200 = df_close[ticker].rolling(200).mean().iloc[-len(dates):]
        fig.add_trace(go.Scatter(
            x=dates, y=sma200, mode='lines', name='200 SMA (年線)',
            line=dict(color='#00BFFF', width=2.0, dash='dash')
        ))

        fig.update_layout(
            title=f"{ticker} - Daily Chart", height=350, margin=dict(l=0, r=0, t=30, b=0),
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
    st.caption("v16.0 決策金字塔 | 增強流動性監測 & 模型白皮書")
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
    with st.spinner("Alpha 正在同步宏觀數據與流動性指標..."):
        df_close, df_open, df_high, df_low = fetch_data(tickers_list)
        qqq_pe = get_valuation_metrics('QQQ')
            
    if df_close.empty:
        st.error("數據獲取失敗。")
        return

    # --- A. 宏觀戰情室 ---
    st.subheader("1. 宏觀戰情室 (The War Room)")
    
    # 數據準備
    qqq_trend = analyze_trend(df_close.get('QQQ'))
    hyg_trend = analyze_trend(df_close.get('HYG')) # 流動性指標
    
    vix_series = df_close.get('^VIX')
    vix_now = vix_series.iloc[-1] if vix_series is not None and not vix_series.empty else None
    
    tnx_series = df_close.get('^TNX')
    tnx_now = tnx_series.iloc[-1] if tnx_series is not None and not tnx_series.empty else None
    
    # 決策引擎
    gear, reason = determine_strategy_gear(qqq_trend, vix_now, qqq_pe, hyg_trend)
    
    # 顯示儀表
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("VIX 恐慌指數", f"{vix_now:.2f}" if vix_now else "N/A", 
                 delta="高風險 > 20" if vix_now and vix_now > 20 else "安全", delta_color="inverse")
    with m_col2:
        # 流動性儀表：看 HYG 是否在年線之上
        hyg_status = "充裕" if hyg_trend and hyg_trend['p_now'] > hyg_trend['sma200'] else "枯竭"
        st.metric("市場流動性 (HYG)", hyg_status, 
                 delta="信用風險低" if hyg_status=="充裕" else "信用風險高", delta_color="normal" if hyg_status=="充裕" else "inverse")
    with m_col3:
        pe_display = f"{qqq_pe:.1f}" if qqq_pe else "N/A (預設25)"
        st.metric("QQQ 遠期本益比", pe_display, 
                 delta="昂貴 > 28" if qqq_pe and qqq_pe > 28 else "合理", delta_color="inverse")
    with m_col4:
        if qqq_trend:
            dist_sma = (qqq_trend['p_now'] - qqq_trend['sma200']) / qqq_trend['sma200']
            st.metric("QQQ vs 年線", f"{dist_sma:.1%}", "牛市區" if dist_sma>0 else "熊市區")

    # 顯示最終決策
    if "熊市" in gear or "流動性" in gear:
        st.error(f"### 🛑 Alpha 防禦指令：{gear}")
    else:
        st.success(f"### 🚀 Alpha 進攻指令：{gear}")
    st.markdown(f"> **決策邏輯：** {reason}")

    st.markdown("---")
    st.markdown("#### 🇺🇸 關鍵基準 K 線 (大盤 vs 流動性)")
    
    b_col1, b_col2, b_col3 = st.columns(3)
    # 加入 HYG 讓用戶直接看到流動性走勢
    benchmarks = ['QQQ', 'TQQQ', 'HYG'] 
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
        
        # Action Logic
        action = "持有"
        if trend['p_now'] < trend['sma200']: action = "熊市避險"
        elif vol_status == "💎 超賣機會 (L2)": action = "加倉/抄底"
        elif vol_status == "⚠️ 情緒過熱 (H2)": action = "止盈觀察"

        table_data.append({
            "代號": ticker,
            "權重": f"{weight:.1%}",
            "現價": f"${trend['p_now']:.2f}",
            "趨勢狀態": trend['status'],
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
        
        with st.expander(f"📊 {ticker} - {trend['status']}", expanded=True):
            k_col1, k_col2 = st.columns([3, 1])
            with k_col1:
                fig = plot_kline_chart(ticker, df_close, df_open, df_high, df_low)
                if fig: st.plotly_chart(fig, use_container_width=True, key=f"deep_{ticker}")
            with k_col2:
                st.markdown("#### 關鍵數據")
                levels, vol_status = calc_volatility_shells(df_close[ticker])
                st.info(f"現價: {trend['p_now']:.2f}")
                
                # 年線狀態
                if trend['p_now'] > trend['sma200']:
                    st.success("✅ 年線之上 (長多)")
                else:
                    st.error("🛑 年線之下 (長空)")
                
                st.caption(f"支撐 (L2): {levels.get('L2', 0):.2f}")
                st.divider()
                st.metric("1個月目標", f"${trend['p_1m']:.2f}", delta=f"{(trend['p_1m']-trend['p_now'])/trend['p_now']:.1%}")

    st.markdown("---")

    # --- D. 量化模型白皮書 (Whitepaper) ---
    st.header("4. 量化模型白皮書 (Quantitative Whitepaper)")
    st.markdown("本系統融合「防禦型 Alpha」哲學，以下為各模組之質性與數學原理解析：")

    with st.container():
        st.markdown("#### 💧 1. 流動性監測模組 (Liquidity Monitor)")
        st.info("""
        **質性解釋：** 「信用利差」是市場的礦坑金絲雀。我們使用 **高收益債 (HYG)** 作為流動性代理。當資金寬鬆時，投資人願意買入垃圾債；當流動性枯竭時，垃圾債最先崩盤。
        """)
        st.latex(r'''
        \text{Liquidity Crisis} = \text{Price}_{HYG} < \text{SMA}_{200}(HYG)
        ''')
        st.markdown("若 HYG 跌破年線，代表系統性風險極高，無論股市走勢如何，皆應強制降檔。")

        st.divider()

        st.markdown("#### 📐 2. 趨勢判定模型 (Trend Model)")
        st.info("""
        **質性解釋：** 採用雙重濾網：
        1. **長期 (SMA200)：** 決定牛熊分界。年線之下不作多。
        2. **短期 (EMA20 + Slope)：** 決定進攻時機。價格站上生命線且斜率向上，代表動能強勁。
        """)
        st.latex(r'''
        \text{Status} = \begin{cases} 
        \text{🛑 Bearish}, & \text{if } P < SMA_{200} \\
        \text{🔥 Bullish}, & \text{if } P > EMA_{20} \text{ and } Slope > 0 \\
        \text{🛡️ Neutral}, & \text{otherwise}
        \end{cases}
        ''')

        st.divider()

        st.markdown("#### 🏰 3. 估值天花板 (Valuation Ceiling)")
        st.info("""
        **質性解釋：** 樹不會長到天上去。當納斯達克 (QQQ) 的遠期本益比超過歷史極端值 (28x-32x) 時，即使趨勢向上，期望回報率也極低，且面臨巨大的「均值回歸」風險。此時禁止開槓桿。
        """)
        
        st.divider()

        st.markdown("#### 🎲 4. 凱利公式倉位建議 (Kelly Criterion)")
        st.info("""
        **質性解釋：** 賭場與對沖基金的資金管理聖杯。根據勝率與盈虧比，計算數學上最優的下注比例。本系統在熊市狀態下會自動懲罰勝率 ($p - 20\%$)，以保護本金。
        """)
        st.latex(r'''
        f^* = \frac{p(b+1)-1}{b} \times 0.5 \quad (\text{Half-Kelly})
        ''')
        
        st.divider()
        
        st.markdown("#### 🛡️ 5. 六維波動防禦 (Volatility Shells)")
        st.info("""
        **質性解釋：** 利用統計學標準差 ($\sigma$) 描繪價格運行的「道路邊界」。L2 (2倍標準差下緣) 通常是主力洗盤的極限，也是絕佳的左側抄底點。
        """)
        st.latex(r'''
        \text{Band} = \mu_{20} \pm (n \times \sigma_{20}), \quad n \in \{1, 2, 3\}
        ''')

if __name__ == "__main__":
    main()