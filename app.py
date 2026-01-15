import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# --- 0. 全局設定 ---
st.set_page_config(page_title="Alpha 2.0: 戰略資產中控台", layout="wide", page_icon="📈")

# 自定義 CSS 美化
st.markdown("""
<style>
    .metric-card {background-color: #0E1117; border: 1px solid #262730; border-radius: 5px; padding: 15px; color: white;}
    .bullish {color: #00FF7F; font-weight: bold;}
    .bearish {color: #FF4B4B; font-weight: bold;}
    .neutral {color: #FFD700; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# --- 1. 核心數據引擎 (Data Engine) ---
@st.cache_data(ttl=3600)
def fetch_data(tickers):
    """
    獲取數據並自動處理 QQQ/QLD/TQQQ/BTC 用於基準對比
    """
    benchmarks = ['QQQ', 'QLD', 'TQQQ', 'BTC-USD']
    all_tickers = list(set(tickers + benchmarks))
    
    try:
        # 下載過去 2 年數據 (足夠計算 350DMA)
        data = yf.download(all_tickers, period="2y", auto_adjust=True)
        
        # 處理 MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            try:
                data = data['Close'] # yfinance 新版結構
            except:
                data = data.xs('Close', axis=1, level=0)
        
        return data.ffill().dropna()
    except Exception as e:
        st.error(f"數據下載失敗: {e}")
        return pd.DataFrame()

# --- 2. 核心趨勢模組 (Trend Projection) ---
def analyze_trend(series):
    """
    計算斜率 (k)、效率 (R2)、20EMA 狀態
    """
    y = series.values.reshape(-1, 1)
    x = np.arange(len(y)).reshape(-1, 1)
    
    # 線性回歸
    model = LinearRegression().fit(x, y)
    k = model.coef_[0].item()
    r2 = model.score(x, y)
    
    # 價格預測
    p_now = series.iloc[-1]
    p_1m = model.predict([[len(y) + 22]])[0].item() # 1個月後
    
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

# --- 3. 六維波動防禦 (Volatility Shells) ---
def calc_volatility_shells(series):
    """
    計算 1/2/3 倍標準差的支撐與壓力位
    """
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

# --- 4. 凱利公式與持倉建議 (Portfolio Logic) ---
def calc_kelly_position(trend_data, benchmark_ret, target_ret):
    """
    基於勝率與賠率計算最佳倉位
    """
    # 簡單勝率估計：如果趨勢向上 (k>0) 且 R2 高，勝率較高
    win_rate = 0.55
    if trend_data['k'] > 0: win_rate += 0.05
    if trend_data['r2'] > 0.6: win_rate += 0.05
    if trend_data['status'] == "🛑 趨勢損毀": win_rate -= 0.15
    
    # 賠率 (盈虧比)
    odds = 2.0 # 默認 2:1
    
    # 凱利公式: f* = (bp - q) / b
    # b = odds, p = win_rate, q = 1-p
    f_star = (odds * win_rate - (1 - win_rate)) / odds
    
    # 凱利減半 (Half-Kelly) 以策安全
    safe_kelly = max(0, f_star * 0.5) 
    
    return safe_kelly * 100, win_rate

# --- 5. 外部審計：比特幣逃頂 (Pi Cycle) ---
def check_pi_cycle(btc_series):
    if btc_series.empty: return False, 0, 0
    
    ma111 = btc_series.rolling(111).mean().iloc[-1]
    ma350_x2 = btc_series.rolling(350).mean().iloc[-1] * 2
    
    signal = ma111 > ma350_x2
    dist = (ma350_x2 - ma111) / ma111 # 距離交叉還有多遠
    
    return signal, ma111, ma350_x2, dist

# --- MAIN: 儀表板介面 ---
def main():
    st.title("Alpha 2.0 Pro: 戰略資產中控台")
    st.markdown("---")

    # 側邊欄輸入
    with st.sidebar:
        st.header("⚙️ 參數設定")
        user_tickers = st.text_input("輸入持倉代號 (逗號分隔)", "BTC-USD, QQQ, 0050.TW, NVDA").upper()
        tickers_list = [t.strip() for t in user_tickers.split(",")]
        
        st.info("💡 系統已自動鎖定 QQQ 與 BTC 作為宏觀錨點。")

    # 獲取數據
    if st.button("🚀 啟動量化審計", type="primary"):
        with st.spinner("Alpha 正在連接交易所數據庫..."):
            df = fetch_data(tickers_list)
            
        if df.empty:
            st.error("無法獲取數據，請檢查代號。")
            return

        # --- A. 宏觀戰情室 (Macro View) ---
        st.subheader("1. 宏觀戰情室 (Macro Audit)")
        col1, col2, col3 = st.columns(3)
        
        # BTC Pi Cycle
        if 'BTC-USD' in df.columns:
            pi_sig, ma111, ma350x2, dist = check_pi_cycle(df['BTC-USD'])
            btc_price = df['BTC-USD'].iloc[-1]
            
            with col1:
                st.markdown("#### ₿ 比特幣逃頂指標")
                st.metric("BTC 現價", f"${btc_price:,.0f}")
                if pi_sig:
                    st.error("🚨 逃頂信號已觸發 (Pi Cycle Crossed)!")
                else:
                    st.success(f"✅ 安全 (距離頂部交叉: {dist:.1%})")
                st.caption(f"111DMA: {ma111:,.0f} | 350DMAx2: {ma350x2:,.0f}")

        # QQQ 趨勢
        if 'QQQ' in df.columns:
            q_trend = analyze_trend(df['QQQ'])
            with col2:
                st.markdown("#### 🇺🇸 美股大盤 (QQQ)")
                st.metric("趨勢狀態", q_trend['status'], delta=f"斜率: {q_trend['k']:.2f}")
                st.caption(f"R2 (趨勢純度): {q_trend['r2']:.2f}")

        # 槓桿對標
        if 'TQQQ' in df.columns and 'QQQ' in df.columns:
            ret_q = df['QQQ'].pct_change().sum()
            ret_tq = df['TQQQ'].pct_change().sum()
            with col3:
                st.markdown("#### ⚡ 槓桿效率")
                st.metric("TQQQ/QQQ 彈性", f"{ret_tq/ret_q:.2f}x")
                if ret_tq/ret_q < 2.5:
                    st.warning("⚠️ 槓桿損耗過大 (震盪市)")
                else:
                    st.success("⚡ 槓桿效率優良")

        st.markdown("---")

        # --- B. 個股戰術分析 (Tactical Analysis) ---
        st.subheader("2. 持倉深度審計 (Portfolio X-Ray)")
        
        # 遍歷用戶輸入的代號
        for ticker in tickers_list:
            if ticker not in df.columns: continue
            if ticker in ['QQQ', 'QLD', 'TQQQ']: continue # 跳過基準
            
            st.markdown(f"### 🎯 {ticker}")
            t_col1, t_col2, t_col3 = st.columns([1, 1, 1])
            
            # 1. 趨勢與預測
            trend = analyze_trend(df[ticker])
            with t_col1:
                st.markdown(f"<span class='{trend['color']}'>{trend['status']}</span>", unsafe_allow_html=True)
                st.metric("當前價格", f"{trend['p_now']:.2f}")
                st.metric("1個月目標 (AI預測)", f"{trend['p_1m']:.2f}", delta=f"{(trend['p_1m']-trend['p_now'])/trend['p_now']:.1%}")
            
            # 2. 六維波動 (Volatility)
            levels, vol_status = calc_volatility_shells(df[ticker])
            with t_col2:
                st.markdown("**🛡️ 六維防禦區間**")
                st.text(f"H3 (極限): {levels['H3']:.2f}")
                st.text(f"H2 (止盈): {levels['H2']:.2f}")
                st.text(f"H1 (壓力): {levels['H1']:.2f}")
                st.info(f"📍 現價: {trend['p_now']:.2f} ({vol_status})")
                st.text(f"L1 (支撐): {levels['L1']:.2f}")
                st.text(f"L2 (止損): {levels['L2']:.2f}")
                st.text(f"L3 (崩盤): {levels['L3']:.2f}")

            # 3. 最佳持倉 (Kelly)
            kelly_pct, win_prob = calc_kelly_position(trend, 0, 0)
            with t_col3:
                st.markdown("**💰 資金控管建議**")
                st.progress(min(int(kelly_pct), 100), text=f"建議倉位: {kelly_pct:.1f}%")
                st.caption(f"預估勝率: {win_prob:.0%}")
                
                if trend['status'] == "🛑 趨勢損毀":
                    st.error("建議動作：減倉/止損")
                elif vol_status == "💎 超賣機會 (L2)":
                    st.success("建議動作：抄底/加倉")
                else:
                    st.info("建議動作：持有 (Hold)")
            
            st.divider()

if __name__ == "__main__":
    main()