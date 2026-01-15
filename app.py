import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
import plotly.express as px

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
        
        # 處理 MultiIndex (兼容 yfinance 新舊版)
        if isinstance(data.columns, pd.MultiIndex):
            try:
                data = data['Close'] 
            except:
                data = data.xs('Close', axis=1, level=0)
        
        # 簡單清理：移除完全沒有數據的列
        data = data.dropna(axis=1, how='all')
        return data.ffill().dropna()
    except Exception as e:
        st.error(f"數據下載失敗: {e}")
        return pd.DataFrame()

# --- 2. 核心趨勢模組 (Trend Projection) ---
def analyze_trend(series):
    """
    計算斜率 (k)、效率 (R2)、20EMA 狀態
    """
    if series.isnull().all(): return None

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
def calc_kelly_position(trend_data):
    """
    基於勝率與賠率計算最佳倉位
    """
    if not trend_data: return 0, 0

    # 簡單勝率估計：如果趨勢向上 (k>0) 且 R2 高，勝率較高
    win_rate = 0.55
    if trend_data['k'] > 0: win_rate += 0.05
    if trend_data['r2'] > 0.6: win_rate += 0.05
    if trend_data['status'] == "🛑 趨勢損毀": win_rate -= 0.15
    
    # 賠率 (盈虧比)
    odds = 2.0 # 默認 2:1
    
    # 凱利公式: f* = (bp - q) / b
    f_star = (odds * win_rate - (1 - win_rate)) / odds
    
    # 凱利減半 (Half-Kelly) 以策安全
    safe_kelly = max(0, f_star * 0.5) 
    
    return safe_kelly * 100, win_rate

# --- 5. 外部審計：比特幣逃頂 (Pi Cycle) ---
def check_pi_cycle(btc_series):
    if btc_series.empty: return False, 0, 0, 0
    
    ma111 = btc_series.rolling(111).mean().iloc[-1]
    ma350_x2 = btc_series.rolling(350).mean().iloc[-1] * 2
    
    signal = ma111 > ma350_x2
    dist = (ma350_x2 - ma111) / ma111 # 距離交叉還有多遠
    
    return signal, ma111, ma350_x2, dist

# --- 6. 輸入解析模組 ---
def parse_input(input_text):
    """
    解析側邊欄的 '代號, 金額' 格式
    """
    portfolio = {}
    lines = input_text.strip().split('\n')
    for line in lines:
        if ',' in line:
            parts = line.split(',')
            ticker = parts[0].strip().upper()
            try:
                value = float(parts[1].strip())
            except:
                value = 0.0
            if ticker:
                portfolio[ticker] = value
        else:
            # 只有代號的情況，預設金額為 0
            ticker = line.strip().upper()
            if ticker:
                portfolio[ticker] = 0.0
    return portfolio

# --- MAIN: 儀表板介面 ---
def main():
    st.title("Alpha 2.0 Pro: 戰略資產中控台")
    st.markdown("---")

    # --- 側邊欄：資產輸入 ---
    with st.sidebar:
        st.header("⚙️ 資產配置輸入")
        st.caption("格式：代號, 持倉金額 (換行分隔)")
        
        default_input = """BTC-USD, 50000
QQQ, 30000
BNSOL-USD, 15000
0050.TW, 20000
NVDA, 10000"""
        
        user_input = st.text_area("持倉清單", default_input, height=200)
        
        # 解析輸入
        portfolio_dict = parse_input(user_input)
        tickers_list = list(portfolio_dict.keys())
        
        total_value = sum(portfolio_dict.values())
        st.metric("總資產估值 (Est.)", f"${total_value:,.0f}")
        
        if st.button("🚀 啟動量化審計", type="primary"):
            st.session_state['run_analysis'] = True
        
    # 如果還沒按按鈕，就停在這
    if not st.session_state.get('run_analysis', False):
        st.info("👈 請在左側輸入您的持倉，並點擊『啟動量化審計』。")
        return

    # --- 開始分析 ---
    with st.spinner("Alpha 正在連接交易所數據庫並計算模型..."):
        df = fetch_data(tickers_list)
            
    if df.empty:
        st.error("無法獲取數據，請檢查代號是否正確。")
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

    # --- B. 資產整合總表 (Integrated Portfolio) ---
    st.subheader("2. 資產整合總表 (Portfolio Overview)")
    
    # 準備表格數據
    table_data = []
    
    for ticker in tickers_list:
        if ticker not in df.columns: continue
        
        # 獲取各項指標
        trend = analyze_trend(df[ticker])
        levels, vol_status = calc_volatility_shells(df[ticker])
        kelly_pct, win_prob = calc_kelly_position(trend)
        
        current_val = portfolio_dict.get(ticker, 0)
        weight = (current_val / total_value) if total_value > 0 else 0
        
        # 建議動作邏輯
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
    
    # 顯示表格與圖表
    p_col1, p_col2 = st.columns([2, 1])
    
    with p_col1:
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)
    
    with p_col2:
        if total_value > 0:
            pie_df = pd.DataFrame(list(portfolio_dict.items()), columns=['Ticker', 'Value'])
            fig = px.pie(pie_df, values='Value', names='Ticker', title='資產配置分布', hole=0.4)
            fig.update_layout(margin=dict(t=30, b=0, l=0, r=0))
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- C. 個股深度戰術卡片 (Tactical Cards) ---
    st.subheader("3. 深度戰術審計 (Deep Dive)")
    
    # 這裡只顯示用戶持倉的詳細六維數據
    cols = st.columns(3)
    for i, ticker in enumerate(tickers_list):
        if ticker not in df.columns: continue
        
        trend = analyze_trend(df[ticker])
        levels, vol_status = calc_volatility_shells(df[ticker])
        
        with cols[i % 3]:
            with st.container(border=True):
                st.markdown(f"#### 🎯 {ticker}")
                st.markdown(f"<span class='{trend['color']}'>{trend['status']}</span>", unsafe_allow_html=True)
                
                # 迷你數據區
                sub_c1, sub_c2 = st.columns(2)
                with sub_c1:
                    st.caption("支撐位 (L2)")
                    st.markdown(f"**{levels['L2']:.2f}**")
                with sub_c2:
                    st.caption("壓力位 (H2)")
                    st.markdown(f"**{levels['H2']:.2f}**")
                
                # 波動區間視覺化 (簡單文字版)
                st.progress((trend['p_now'] - levels['L3']) / (levels['H3'] - levels['L3']), text=f"區間位置 ({vol_status})")
                
                st.caption(f"AI 預測目標: ${trend['p_1m']:.2f}")

if __name__ == "__main__":
    main()