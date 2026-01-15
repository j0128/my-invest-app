import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 0. 全局設定 ---
st.set_page_config(page_title="Alpha 3.2 Pro: 資金雷達戰情室", layout="wide", page_icon="📡")

# 自定義 CSS
st.markdown("""
<style>
    .metric-card {background-color: #0E1117; border: 1px solid #262730; border-radius: 5px; padding: 15px; color: white;}
    .bullish {color: #00FF7F; font-weight: bold;}
    .bearish {color: #FF4B4B; font-weight: bold;}
    .neutral {color: #FFD700; font-weight: bold;}
    .backtest-box {background-color: #1E1E1E; padding: 10px; border-radius: 5px; border-left: 5px solid #FF4B4B; margin-top: 10px;}
</style>
""", unsafe_allow_html=True)

# --- 1. 核心數據引擎 ---
@st.cache_data(ttl=3600)
def fetch_market_data(tickers):
    benchmarks = ['QQQ', 'QLD', 'TQQQ', 'BTC-USD', '^VIX', '^TNX', 'HYG']
    all_tickers = list(set(tickers + benchmarks))
    
    data = {col: {} for col in ['Close', 'Open', 'High', 'Low', 'Volume']}
    
    progress_bar = st.progress(0, text="Alpha 正在執行時光回溯測試...")
    
    for i, t in enumerate(all_tickers):
        try:
            progress_bar.progress((i + 1) / len(all_tickers), text=f"正在下載: {t} ...")
            df = yf.Ticker(t).history(period="2y", auto_adjust=True)
            if df.empty: continue
            
            data['Close'][t] = df['Close']
            data['Open'][t] = df['Open']
            data['High'][t] = df['High']
            data['Low'][t] = df['Low']
            data['Volume'][t] = df['Volume']
        except: continue
            
    progress_bar.empty()
    return (pd.DataFrame(data['Close']).ffill(), 
            pd.DataFrame(data['Open']).ffill(), 
            pd.DataFrame(data['High']).ffill(), 
            pd.DataFrame(data['Low']).ffill(),
            pd.DataFrame(data['Volume']).ffill())

@st.cache_data(ttl=3600*12)
def fetch_fred_liquidity(api_key):
    if not api_key: return None
    try:
        fred = Fred(api_key=api_key)
        walcl = fred.get_series('WALCL', observation_start='2024-01-01')
        tga = fred.get_series('WTREGEN', observation_start='2024-01-01')
        rrp = fred.get_series('RRPONTSYD', observation_start='2024-01-01')
        df = pd.DataFrame({'WALCL': walcl, 'TGA': tga, 'RRP': rrp}).ffill().dropna()
        df['Net_Liquidity'] = (df['WALCL'] - df['TGA'] - df['RRP']) / 1000 
        return df
    except: return None

def format_number(num):
    if num is None: return "N/A"
    abs_num = abs(num)
    if abs_num >= 1_000_000: return f"{num/1_000_000:.2f}M"
    elif abs_num >= 1_000: return f"{num/1_000:.2f}K"
    else: return f"{num:.2f}"

# --- 2. 三角定位算法 (含回測邏輯) ---

# A. ATR Target
def calc_atr_target(close, high, low, slice_idx=-1):
    try:
        # 切片數據到指定時間點
        c = close.iloc[:slice_idx+1] if slice_idx != -1 else close
        h = high.iloc[:slice_idx+1] if slice_idx != -1 else high
        l = low.iloc[:slice_idx+1] if slice_idx != -1 else low
        
        prev_close = c.shift(1)
        tr = pd.concat([h-l, (h-prev_close).abs(), (l-prev_close).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        
        # 預測 22 天後的極限
        monthly_range = atr * np.sqrt(22) * 1.2 
        return c.iloc[-1] + monthly_range
    except: return None

# B. Monte Carlo P50
def calc_monte_carlo_target(series, slice_idx=-1, days=22, simulations=500):
    try:
        s = series.iloc[:slice_idx+1] if slice_idx != -1 else series
        returns = s.pct_change().dropna()
        last_price = s.iloc[-1]
        mu = returns.mean()
        sigma = returns.std()
        
        simulation_df = pd.DataFrame()
        for i in range(simulations):
            daily_vol = np.random.normal(mu, sigma, days)
            price_series = [last_price]
            for x in daily_vol:
                price_series.append(price_series[-1] * (1 + x))
            simulation_df[i] = price_series
            
        final_prices = simulation_df.iloc[-1]
        return np.percentile(final_prices, 50)
    except: return None

# C. Fibonacci 1.618
def calc_fib_target(series, slice_idx=-1):
    try:
        s = series.iloc[:slice_idx+1] if slice_idx != -1 else series
        recent_window = s.iloc[-60:]
        high, low = recent_window.max(), recent_window.min()
        return high + (high - low) * 0.618
    except: return None

# --- 3. 回測引擎 (New Module) ---
def run_backtest_lab(ticker, df_close, df_high, df_low):
    """
    時光機：回到 22 天前 (約一個月)，計算當時的預測，並與今天比較
    """
    if ticker not in df_close.columns or len(df_close) < 250: return None
    
    # 1. 設定回測點：22 個交易日前
    lookback_days = 22 
    past_idx = len(df_close) - lookback_days - 1
    
    past_price = df_close[ticker].iloc[past_idx]
    current_price = df_close[ticker].iloc[-1]
    
    # 2. 用當時數據跑模型
    pred_atr = calc_atr_target(df_close[ticker], df_high[ticker], df_low[ticker], slice_idx=past_idx)
    pred_mc = calc_monte_carlo_target(df_close[ticker], slice_idx=past_idx)
    pred_fib = calc_fib_target(df_close[ticker], slice_idx=past_idx)
    
    # 3. 計算誤差
    def get_error(pred, actual):
        if not pred: return None, None
        err_pct = (pred - actual) / actual
        return pred, err_pct # 負值代表預測低了，正值代表預測高了

    res_atr, err_atr = get_error(pred_atr, current_price)
    res_mc, err_mc = get_error(pred_mc, current_price)
    res_fib, err_fib = get_error(pred_fib, current_price)
    
    return {
        "date_past": df_close.index[past_idx].strftime('%Y-%m-%d'),
        "price_past": past_price,
        "price_now": current_price,
        "ATR": (res_atr, err_atr),
        "MC": (res_mc, err_mc),
        "Fib": (res_fib, err_fib)
    }

# --- 4. 既有模組 ---
def calc_fund_flow(close, high, low, volume):
    if volume is None or volume.empty: return None
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    y = obv.values[-20:].reshape(-1, 1)
    x = np.arange(len(y)).reshape(-1, 1)
    obv_slope = LinearRegression().fit(x, y).coef_[0].item()
    
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    pos = np.where(typical_price > typical_price.shift(1), money_flow, 0)
    neg = np.where(typical_price < typical_price.shift(1), money_flow, 0)
    pos_sum = pd.Series(pos).rolling(14).sum().iloc[-1]
    neg_sum = pd.Series(neg).rolling(14).sum().iloc[-1]
    mfi = 100 - (100 / (1 + pos_sum / neg_sum)) if neg_sum != 0 else 100
    return {"obv_slope": obv_slope, "mfi": mfi, "obv_series": obv}

def analyze_trend(series):
    if series is None or len(series) < 200: return None
    series = series.dropna()
    y = series.values.reshape(-1, 1)
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    
    p_now = series.iloc[-1].item()
    p_2w = model.predict([[len(y) + 10]])[0].item()
    p_1m = model.predict([[len(y) + 22]])[0].item()
    p_3m = model.predict([[len(y) + 66]])[0].item()
    
    k = model.coef_[0].item()
    r2 = model.score(x, y)
    ema20 = series.ewm(span=20).mean().iloc[-1].item()
    sma200 = series.rolling(200).mean().iloc[-1].item()
    
    status = "🛡️ 區間盤整"
    if p_now < sma200: status = "🛑 熊市防禦"
    elif p_now > ema20 and k > 0: status = "🔥 加速進攻"
    elif p_now < ema20: status = "⚠️ 動能減弱"
    
    is_overheated = (k > 0 and p_1m < p_now)
    return {"k": k, "r2": r2, "p_now": p_now, "p_2w": p_2w, "p_1m": p_1m, "p_3m": p_3m, 
            "ema20": ema20, "sma200": sma200, "status": status, "is_overheated": is_overheated}

@st.cache_data(ttl=3600*12)
def get_valuation_metrics(ticker):
    try: return yf.Ticker(ticker).info.get('forwardPE', None)
    except: return None

def calc_volatility_shells(series):
    try:
        window = 20
        mean = series.rolling(window).mean().iloc[-1]
        std = series.rolling(window).std().iloc[-1]
        p = series.iloc[-1]
        levels = {f'H{i}': mean + i*std for i in range(1,4)}
        levels.update({f'L{i}': mean - i*std for i in range(1,4)})
        status = "正常波動"
        if p > levels['H2']: status = "⚠️ 情緒過熱 (H2)"
        if p < levels['L2']: status = "💎 超賣機會 (L2)"
        return levels, status
    except: return {}, "計算錯誤"

def calc_kelly_position(trend_data):
    if not trend_data: return 0, 0
    win_rate = 0.55
    if trend_data['k'] > 0: win_rate += 0.05
    if trend_data['r2'] > 0.6: win_rate += 0.05
    if "熊市" in trend_data['status']: win_rate -= 0.2
    f_star = (2.0 * win_rate - (1 - win_rate)) / 2.0
    return max(0, f_star * 0.5) * 100, win_rate

def determine_strategy_gear(qqq_trend, vix_now, qqq_pe, hyg_trend, net_liquidity_trend):
    if not qqq_trend: return "N/A", "數據不足"
    price = qqq_trend['p_now']
    sma200 = qqq_trend['sma200']
    ema20 = qqq_trend['ema20']
    vix = vix_now if vix_now else 20
    pe = qqq_pe if qqq_pe else 25 
    
    if net_liquidity_trend == "收縮": return "檔位 1 (QQQ)", "💧 聯準會縮表：淨流動性下降，市場缺乏燃料。"
    if hyg_trend and hyg_trend['p_now'] < hyg_trend['sma200']: return "檔位 0 (現金)", "💔 信用破裂：HYG 跌破年線，風險極高。"
    if price < sma200: return "檔位 0 (現金)", "🛑 熊市：跌破年線。"
    if pe > 32: return "檔位 1 (QQQ)", "⚠️ 估值天花板：PE > 32。"
    if vix > 22: return "檔位 1 (QQQ)", "🌩️ VIX 恐慌模式。"
    if price > ema20: return "檔位 3 (TQQQ)", "🚀 完美風口：流動性充裕 + 趨勢向上。"
    return "檔位 2 (QLD)", "🛡️ 牛市回調：保持中度槓桿。"

def plot_combo_chart(ticker, df_close, df_vol, trend_data, fund_flow):
    if ticker not in df_close.columns: return None
    dates = df_close.index[-150:]
    closes = df_close[ticker].iloc[-150:]
    obv = fund_flow['obv_series'].iloc[-150:]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=closes, name='Price', line=dict(color='#00FF7F', width=2)))
    fig.add_trace(go.Scatter(x=dates, y=df_close[ticker].ewm(span=20).mean().iloc[-150:], name='20 EMA', line=dict(color='#FFD700', width=1)))
    fig.add_trace(go.Scatter(x=dates, y=obv, name='OBV (資金)', line=dict(color='#00BFFF', width=2), yaxis='y2'))
    fig.update_layout(title=f"{ticker} - 量價關係圖", height=400,
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
                      xaxis=dict(showgrid=False), yaxis=dict(title="Price", showgrid=True, gridcolor='#333'),
                      yaxis2=dict(title="OBV", overlaying='y', side='right', showgrid=False))
    return fig

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
    st.title("Alpha 2.0 Pro: 雙引擎資金雷達版")
    st.caption("v26.0 回測驗證版 | 實時檢驗模型誤差率")
    st.markdown("---")

    with st.sidebar:
        st.header("⚙️ 參數設定")
        fred_key = st.secrets.get("FRED_API_KEY", None)
        if fred_key: st.success("🔑 FRED Key 已載入")
        else: fred_key = st.text_input("FRED API Key (選填)", type="password")
        
        st.header("💼 資產配置")
        default_input = """BTC-USD, 10000
0050.TW, 1000
AMD, 1000"""
        user_input = st.text_area("持倉清單", default_input, height=200)
        portfolio_dict = parse_input(user_input)
        tickers_list = list(portfolio_dict.keys())
        total_value = sum(portfolio_dict.values())
        st.metric("總資產估值 (Est.)", f"${total_value:,.0f}")
        if st.button("🚀 啟動全域掃描", type="primary"): st.session_state['run'] = True

    if not st.session_state.get('run', False):
        st.info("👈 請點擊『啟動全域掃描』。")
        return

    with st.spinner("正在執行三角定位與時光回溯測試..."):
        df_close, df_open, df_high, df_low, df_vol = fetch_market_data(tickers_list)
        df_liquidity = fetch_fred_liquidity(fred_key)
        qqq_pe = get_valuation_metrics('QQQ')

    if df_close.empty: st.error("市場數據獲取失敗"); return

    # --- A. 宏觀 ---
    st.subheader("1. 宏觀與流動性引擎")
    vix = df_close.get('^VIX').iloc[-1] if '^VIX' in df_close else None
    hyg_trend = analyze_trend(df_close.get('HYG'))
    
    liq_status, liq_trend_val = "未知 (無 Key)", "N/A"
    if df_liquidity is not None:
        curr, prev = df_liquidity['Net_Liquidity'].iloc[-1], df_liquidity['Net_Liquidity'].iloc[-5]
        liq_status = "擴張 (印鈔中)" if curr > prev else "收縮 (抽水中)"
        liq_trend_val = "擴張" if curr > prev else "收縮"
    
    qqq_trend = analyze_trend(df_close.get('QQQ'))
    gear, reason = determine_strategy_gear(qqq_trend, vix, qqq_pe, hyg_trend, liq_trend_val)
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: 
        if df_liquidity is not None: st.metric("美元淨流動性", liq_status, f"${df_liquidity['Net_Liquidity'].iloc[-1]:.2f}T")
        else: st.metric("美元淨流動性", "N/A", "No API Key")
    with c2: 
        h_stat = "充裕" if hyg_trend and hyg_trend['p_now'] > hyg_trend['sma200'] else "枯竭"
        st.metric("信用市場 (HYG)", h_stat, delta="違約風險" if h_stat=="枯竭" else "健康", delta_color="inverse")
    with c3: st.metric("VIX", f"{vix:.2f}" if vix else "N/A", delta="風暴" if vix and vix>22 else "平靜", delta_color="inverse")
    with c4: st.metric("Alpha 指令", gear)

    if "收縮" in liq_status or "枯竭" in h_stat: st.warning(f"⚠️ {reason}")
    else: st.success(f"✅ {reason}")

    if df_liquidity is not None:
        st.plotly_chart(px.line(df_liquidity, y='Net_Liquidity', title='聯準會淨流動性趨勢'), use_container_width=True)
    st.markdown("---")

    # --- B. 資金流向 ---
    st.subheader("2. 資金流向與三角定位")
    for ticker in tickers_list:
        if ticker not in df_close.columns: continue
        trend = analyze_trend(df_close[ticker])
        ff = calc_fund_flow(df_close[ticker], df_high[ticker], df_low[ticker], df_vol[ticker])
        if not trend or not ff: continue
        
        target_atr = calc_atr_target(df_close[ticker], df_high[ticker], df_low[ticker])
        target_mc = calc_monte_carlo_target(df_close[ticker])
        target_fib = calc_fib_target(df_close[ticker])
        obv_display = format_number(ff['obv_slope'])
        
        # 執行回測
        bt_res = run_backtest_lab(ticker, df_close, df_high, df_low)
        
        with st.expander(f"📡 {ticker} - 資金: {'流入' if ff['obv_slope']>0 else '流出'} | 中樞(MC): ${target_mc:.2f}", expanded=True):
            k1, k2 = st.columns([3, 1])
            with k1:
                st.plotly_chart(plot_combo_chart(ticker, df_close, df_vol, trend, ff), use_container_width=True, key=f"ff_{ticker}")
            with k2:
                st.markdown("#### 🎯 三角定位預測")
                if target_atr: st.write(f"**ATR Target:** ${target_atr:.2f}")
                if target_mc: st.write(f"**Monte Carlo P50:** ${target_mc:.2f}")
                if target_fib: st.write(f"**Fibonacci 1.618:** ${target_fib:.2f}")
                
                st.divider()
                st.write("**三階段推演:**")
                st.caption(f"2週: ${trend['p_2w']:.2f}")
                st.caption(f"1月: ${trend['p_1m']:.2f}")
                st.caption(f"3月: ${trend['p_3m']:.2f}")
                
                # 回測區塊
                if bt_res:
                    st.markdown("#### 🧪 真實回測")
                    st.caption(f"回測基準日: {bt_res['date_past']}")
                    st.caption(f"當時股價: ${bt_res['price_past']:.2f}")
                    st.caption(f"今日現價: ${bt_res['price_now']:.2f}")
                    
                    def show_err(name, data):
                        pred, err = data
                        if pred:
                            color = "green" if abs(err) < 0.05 else "red"
                            st.markdown(f"{name} 誤差: <span style='color:{color}'>{err:.1%}</span>", unsafe_allow_html=True)
                    
                    show_err("ATR", bt_res['ATR'])
                    show_err("MC", bt_res['MC'])
                    show_err("Fib", bt_res['Fib'])

    st.markdown("---")
    
    # --- C. 總表 ---
    st.subheader("3. 資產配置總表")
    table_data = []
    for ticker in tickers_list:
        if ticker not in df_close.columns: continue
        trend = analyze_trend(df_close[ticker])
        vol_levels, vol_status = calc_volatility_shells(df_close[ticker])
        ff = calc_fund_flow(df_close[ticker], df_high[ticker], df_low[ticker], df_vol[ticker])
        kelly_pct, _ = calc_kelly_position(trend)
        target_mc = calc_monte_carlo_target(df_close[ticker])
        
        current_val = portfolio_dict.get(ticker, 0)
        weight = (current_val / total_value) if total_value > 0 else 0
        action = "持有"
        if ff and ff['mfi']>85: action = "止盈 (過熱)"
        elif trend['status'] == "🛑 熊市防禦": action = "清倉/避險"
        elif ff and ff['obv_slope'] > 0 and vol_status == "💎 超賣機會 (L2)": action = "強力買進 (吸籌)"
        
        table_data.append({
            "代號": ticker, "權重": f"{weight:.1%}", "現價": f"${trend['p_now']:.2f}",
            "趨勢": trend['status'], 
            "2週預測": f"${trend['p_2w']:.2f}", "1月預測": f"${trend['p_1m']:.2f}", "3月預測": f"${trend['p_3m']:.2f}",
            "資金流": "流入" if ff and ff['obv_slope']>0 else "流出",
            "凱利建議": f"{kelly_pct:.1f}%", "建議": action
        })
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    st.markdown("---")

    # --- D. 白皮書 ---
    st.header("4. 量化模型白皮書 (Quantitative Logic & Formulas)")
    with st.container():
        st.subheader("🧪 回測實驗室 (Backtest Lab)")
        st.info("""
        **運作原理：** 為了驗證模型準確度，系統會自動將時間回撥至 22 個交易日前 (約 1 個月)，使用當時的數據計算三角目標價，並與今天的實際價格進行誤差對比。
        * **綠色誤差 (<5%)：** 代表該模型對此股票極具參考價值。
        * **紅色誤差 (>5%)：** 代表該模型近期失準，建議參考其他指標。
        """)
        st.divider()
        st.subheader("🎯 價格目標三角定位 (Triangulation Pricing)")
        c1, c2, c3 = st.columns(3)
        with c1: st.info("### 1. ATR Target\n**邏輯：物理波動極限**\n$$P_{target} = P_{now} + (ATR_{14} \\times \\sqrt{22} \\times 1.2)$$")
        with c2: st.info("### 2. Monte Carlo P50\n**邏輯：統計機率中樞**\n模擬 1000 次隨機漫步，取中位數。")
        with c3: st.info("### 3. Fibonacci 1.618\n**邏輯：群眾心理共識**\n$$P_{target} = H + (H - L) \\times 0.618$$")
    
    st.divider()
    st.markdown("#### 🔮 線性推演 (Linear Projection)")
    st.info("基於迴歸斜率，推演未來不同時間點的理論價格：2週 ($t+10$)、1個月 ($t+22$)、3個月 ($t+66$)。")

if __name__ == "__main__":
    main()