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
st.set_page_config(page_title="Alpha 2.0 Pro: 資金雷達戰情室", layout="wide", page_icon="📡")

# 自定義 CSS
st.markdown("""
<style>
    .metric-card {background-color: #0E1117; border: 1px solid #262730; border-radius: 5px; padding: 15px; color: white;}
    .bullish {color: #00FF7F; font-weight: bold;}
    .bearish {color: #FF4B4B; font-weight: bold;}
    .neutral {color: #FFD700; font-weight: bold;}
    .liquidity-box {border-left: 5px solid #00BFFF; background-color: #001f3f; padding: 10px;}
</style>
""", unsafe_allow_html=True)

# --- 1. 核心數據引擎 (OHLCV + FRED) ---
@st.cache_data(ttl=3600)
def fetch_market_data(tickers):
    """
    抓取 OHLCV (含成交量) 用於計算資金流
    """
    benchmarks = ['QQQ', 'QLD', 'TQQQ', 'BTC-USD', '^VIX', '^TNX', 'HYG']
    all_tickers = list(set(tickers + benchmarks))
    
    data = {col: {} for col in ['Close', 'Open', 'High', 'Low', 'Volume']}
    
    progress_bar = st.progress(0, text="Alpha 正在掃描全市場資金流向...")
    
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
    """
    抓取真實美元流動性 (Fed Balance Sheet - TGA - RRP)
    """
    if not api_key: return None
    try:
        fred = Fred(api_key=api_key)
        # WALCL: Fed總資產, WTREGEN: 財政部TGA帳戶, RRPONTSYD: 逆回購
        walcl = fred.get_series('WALCL', observation_start='2024-01-01')
        tga = fred.get_series('WTREGEN', observation_start='2024-01-01')
        rrp = fred.get_series('RRPONTSYD', observation_start='2024-01-01')
        
        # 數據頻率不同，需對齊 (以週為單位 forward fill)
        df = pd.DataFrame({'WALCL': walcl, 'TGA': tga, 'RRP': rrp}).ffill().dropna()
        
        # 計算淨流動性 (單位：十億美元)
        df['Net_Liquidity'] = (df['WALCL'] - df['TGA'] - df['RRP']) / 1000 
        return df
    except: return None

# --- 2. 資金流向指標 (OBV & MFI) ---
def calc_fund_flow(close, high, low, volume):
    if volume is None or volume.empty: return None
    
    # 1. OBV (On-Balance Volume)
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    
    # 計算 OBV 趨勢 (斜率)
    y = obv.values[-20:].reshape(-1, 1) # 看過去 20 天
    x = np.arange(len(y)).reshape(-1, 1)
    obv_slope = LinearRegression().fit(x, y).coef_[0].item()
    
    # 2. MFI (Money Flow Index)
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    
    positive_flow = np.where(typical_price > typical_price.shift(1), money_flow, 0)
    negative_flow = np.where(typical_price < typical_price.shift(1), money_flow, 0)
    
    # 14天週期
    pos_sum = pd.Series(positive_flow).rolling(14).sum().iloc[-1]
    neg_sum = pd.Series(negative_flow).rolling(14).sum().iloc[-1]
    
    if neg_sum == 0: mfi = 100
    else:
        mfi_ratio = pos_sum / neg_sum
        mfi = 100 - (100 / (1 + mfi_ratio))
        
    return {"obv_slope": obv_slope, "mfi": mfi, "obv_series": obv}

# --- 3. 趨勢與估值 ---
def analyze_trend(series):
    if series is None: return None
    series = series.dropna()
    if len(series) < 200: return None

    y = series.values.reshape(-1, 1)
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    
    p_now = series.iloc[-1].item()
    p_1m = model.predict([[len(y) + 22]])[0].item()
    k = model.coef_[0].item()
    
    ema20 = series.ewm(span=20).mean().iloc[-1].item()
    sma200 = series.rolling(200).mean().iloc[-1].item()
    
    status = "🛡️ 區間盤整"
    if p_now < sma200: status = "🛑 熊市防禦"
    elif p_now > ema20 and k > 0: status = "🔥 加速進攻"
    elif p_now < ema20: status = "⚠️ 動能減弱"
        
    is_overheated = (k > 0 and p_1m < p_now)
    
    return {"k": k, "p_now": p_now, "p_1m": p_1m, "ema20": ema20, "sma200": sma200, 
            "status": status, "is_overheated": is_overheated}

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

# --- 4. 決策金字塔 ---
def determine_strategy_gear(qqq_trend, vix_now, qqq_pe, hyg_trend, net_liquidity_trend):
    if not qqq_trend: return "N/A", "數據不足"
    price = qqq_trend['p_now']
    sma200 = qqq_trend['sma200']
    ema20 = qqq_trend['ema20']
    vix = vix_now if vix_now else 20
    pe = qqq_pe if qqq_pe else 25 
    
    # 1. 真實流動性濾網 (FED Net Liquidity)
    if net_liquidity_trend == "收縮":
        return "檔位 1 (QQQ)", "💧 聯準會縮表：淨流動性下降，市場缺乏燃料。禁止高槓桿。"

    # 2. 替代流動性濾網 (HYG)
    if hyg_trend and hyg_trend['p_now'] < hyg_trend['sma200']:
        return "檔位 0 (現金)", "💔 信用破裂：高收益債跌破年線，系統性風險極高。"

    # 3. 趨勢與估值
    if price < sma200: return "檔位 0 (現金)", "🛑 熊市：跌破年線。"
    if pe > 32: return "檔位 1 (QQQ)", "⚠️ 估值天花板：PE > 32。"
    if vix > 22: return "檔位 1 (QQQ)", "🌩️ VIX 恐慌模式。"
    
    # 4. 進攻
    if price > ema20: return "檔位 3 (TQQQ)", "🚀 完美風口：流動性充裕 + 趨勢向上。"
    return "檔位 2 (QLD)", "🛡️ 牛市回調：保持中度槓桿。"

# --- 5. 繪圖 ---
def plot_combo_chart(ticker, df_close, df_vol, trend_data, fund_flow):
    if ticker not in df_close.columns: return None
    
    dates = df_close.index[-150:]
    closes = df_close[ticker].iloc[-150:]
    obv = fund_flow['obv_series'].iloc[-150:]
    
    fig = go.Figure()
    
    # 主圖：K線
    fig.add_trace(go.Scatter(x=dates, y=closes, name='Price', line=dict(color='#00FF7F', width=2)))
    fig.add_trace(go.Scatter(x=dates, y=df_close[ticker].ewm(span=20).mean().iloc[-150:], name='20 EMA', line=dict(color='#FFD700', width=1)))
    
    # 副圖：OBV
    fig.add_trace(go.Scatter(x=dates, y=obv, name='OBV (資金)', line=dict(color='#00BFFF', width=2), yaxis='y2'))
    
    fig.update_layout(
        title=f"{ticker} - 量價關係圖",
        height=400,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
        xaxis=dict(showgrid=False),
        yaxis=dict(title="Price", showgrid=True, gridcolor='#333'),
        yaxis2=dict(title="OBV", overlaying='y', side='right', showgrid=False)
    )
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
    st.caption("v19.0 | 自動載入 Secrets API Key")
    st.markdown("---")

    with st.sidebar:
        st.header("⚙️ 參數設定")
        
        # [升級] 自動從 Secrets 讀取 Key，若無則顯示輸入框
        fred_key = None
        if "FRED_API_KEY" in st.secrets:
            fred_key = st.secrets["FRED_API_KEY"]
            st.success("🔑 FRED API Key 已從 Secrets 載入")
        else:
            fred_key = st.text_input("FRED API Key (選填)", type="password", help="輸入後可解鎖真實流動性數據")
        
        st.header("💼 資產配置")
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
        
        if st.button("🚀 啟動全域掃描", type="primary"): st.session_state['run'] = True

    if not st.session_state.get('run', False):
        if fred_key:
            st.info("👈 API Key 已就緒，請點擊『啟動全域掃描』。")
        else:
            st.info("👈 請輸入 FRED Key (可選) 並點擊啟動。")
        return

    # 下載數據
    with st.spinner("正在建立雙引擎連線 (FRED + Market)..."):
        df_close, df_open, df_high, df_low, df_vol = fetch_market_data(tickers_list)
        df_liquidity = fetch_fred_liquidity(fred_key)
        qqq_pe = get_valuation_metrics('QQQ')

    if df_close.empty: st.error("市場數據獲取失敗"); return

    # --- A. 宏觀與流動性 (The Engine Room) ---
    st.subheader("1. 宏觀與流動性引擎 (The Engine Room)")
    
    # 計算宏觀指標
    vix = df_close.get('^VIX').iloc[-1] if '^VIX' in df_close else None
    hyg_trend = analyze_trend(df_close.get('HYG'))
    
    # 計算真實流動性狀態
    liq_status = "未知 (無 Key)"
    liq_trend_val = "N/A"
    if df_liquidity is not None:
        current_liq = df_liquidity['Net_Liquidity'].iloc[-1]
        prev_liq = df_liquidity['Net_Liquidity'].iloc[-5] # 一週前
        if current_liq > prev_liq: 
            liq_status = "擴張 (印鈔中)"
            liq_trend_val = "擴張"
        else: 
            liq_status = "收縮 (抽水中)"
            liq_trend_val = "收縮"
    
    # 決策
    qqq_trend = analyze_trend(df_close.get('QQQ'))
    gear, reason = determine_strategy_gear(qqq_trend, vix, qqq_pe, hyg_trend, liq_trend_val)
    
    # 顯示儀表
    c1, c2, c3, c4 = st.columns(4)
    with c1: 
        if df_liquidity is not None:
            st.metric("美元淨流動性 (Fed)", liq_status, f"${df_liquidity['Net_Liquidity'].iloc[-1]:.2f}T")
        else:
            st.metric("美元淨流動性", "N/A", "未偵測到 API Key")
            
    with c2: 
        h_stat = "充裕" if hyg_trend and hyg_trend['p_now'] > hyg_trend['sma200'] else "枯竭"
        st.metric("信用市場 (HYG)", h_stat, delta="垃圾債健康" if h_stat=="充裕" else "違約風險升", delta_color="inverse")
    with c3:
        st.metric("VIX 恐慌指數", f"{vix:.2f}" if vix else "N/A", delta="風暴" if vix and vix>22 else "平靜", delta_color="inverse")
    with c4:
        st.metric("Alpha 指令", gear)

    if "收縮" in liq_status or "枯竭" in h_stat:
        st.warning(f"⚠️ **流動性警報：** {reason}")
    else:
        st.success(f"✅ **系統狀態：** {reason}")

    # 流動性圖表
    if df_liquidity is not None:
        fig_liq = px.line(df_liquidity, y='Net_Liquidity', title='聯準會淨流動性趨勢 (Net Liquidity = Fed Assets - TGA - RRP)')
        st.plotly_chart(fig_liq, use_container_width=True)

    st.markdown("---")

    # --- B. 資金流向深度審計 (Fund Flow Radar) ---
    st.subheader("2. 持倉資金流向雷達 (Fund Flow Radar)")
    st.markdown("偵測「量價背離」與「主力吸籌」跡象：")
    
    for ticker in tickers_list:
        if ticker not in df_close.columns: continue
        trend = analyze_trend(df_close[ticker])
        ff = calc_fund_flow(df_close[ticker], df_high[ticker], df_low[ticker], df_vol[ticker])
        
        if not trend or not ff: continue
        
        # 資金流訊號判斷
        obv_signal = "吸籌 (量先價行)" if ff['obv_slope'] > 0 else "出貨 (量縮/背離)"
        mfi_signal = "過熱 (>80)" if ff['mfi'] > 80 else ("超賣 (<20)" if ff['mfi'] < 20 else "中性")
        
        with st.expander(f"📡 {ticker} - 資金訊號: {obv_signal} | MFI: {ff['mfi']:.1f}", expanded=True):
            k1, k2 = st.columns([3, 1])
            with k1:
                st.plotly_chart(plot_combo_chart(ticker, df_close, df_vol, trend, ff), use_container_width=True, key=f"ff_{ticker}")
            with k2:
                st.markdown("#### 資金數據")
                st.metric("OBV 趨勢", "向上" if ff['obv_slope'] > 0 else "向下", delta=f"斜率: {ff['obv_slope']:.2f}")
                st.metric("MFI 資金流", f"{ff['mfi']:.1f}", delta=mfi_signal, delta_color="inverse")
                
                # 乖離警示
                if trend['is_overheated']:
                    st.error("🔥 價格乖離過大！(可能利好出盡)")
                elif ff['mfi'] > 80:
                    st.warning("⚠️ 資金極度過熱")
                else:
                    st.info("✅ 資金結構健康")
                
                st.divider()
                st.caption(f"1個月預測: ${trend['p_1m']:.2f}")

    st.markdown("---")
    
    # --- C. 資產總表 ---
    st.subheader("3. 資產配置總表")
    table_data = []
    for ticker in tickers_list:
        if ticker not in df_close.columns: continue
        trend = analyze_trend(df_close[ticker])
        vol_levels, vol_status = calc_volatility_shells(df_close[ticker])
        ff = calc_fund_flow(df_close[ticker], df_high[ticker], df_low[ticker], df_vol[ticker])
        
        current_val = portfolio_dict.get(ticker, 0)
        weight = (current_val / total_value) if total_value > 0 else 0
        
        # 綜合建議
        action = "持有"
        if trend['is_overheated'] or (ff and ff['mfi']>85): action = "止盈 (過熱)"
        elif trend['status'] == "🛑 熊市防禦": action = "清倉/避險"
        elif ff and ff['obv_slope'] > 0 and vol_status == "💎 超賣機會 (L2)": action = "強力買進 (吸籌)"
        
        table_data.append({
            "代號": ticker,
            "權重": f"{weight:.1%}",
            "現價": f"${trend['p_now']:.2f}",
            "趨勢": trend['status'],
            "資金流 (OBV)": "流入 🟢" if ff and ff['obv_slope']>0 else "流出 🔴",
            "MFI狀態": f"{ff['mfi']:.0f}" if ff else "N/A",
            "乖離警示": "🔥" if trend['is_overheated'] else "-",
            "建議": action
        })
        
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.header("4. 量化模型白皮書 (v19.0)")
    st.info("""
    **新增模組說明：**
    1. **淨流動性 (Net Liquidity):** 這是美股的「燃料」。公式 = Fed資產 - TGA帳戶 - 逆回購。水位上升=牛市引擎；水位下降=熊市壓力。
    2. **OBV (能量潮):** 累計成交量指標。當股價盤整但 OBV 創新高，代表主力在「吸籌」，是暴漲前兆。
    3. **MFI (資金流指標):** 結合價格與成交量的 RSI。MFI > 80 代表資金過熱，通常是利好出盡的賣點。
    """)

if __name__ == "__main__":
    main()