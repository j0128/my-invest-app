import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from fredapi import Fred
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import plotly.express as px

# --- 1. 系統初始化 (功能 20: Session State) ---
st.set_page_config(page_title="Posa Alpha 4.4 Flagship", layout="wide")

if 'prices' not in st.session_state:
    st.session_state.prices = None
if 'audit_results' not in st.session_state:
    st.session_state.audit_results = {}

# Seeking Alpha 深度指標 (功能 9)
SA_INSIGHTS = {
    'MU': 'HBM 領先, PEG 0.20x', 
    'CLS': '15次盈餘上修, 0次下修', 
    'AMD': 'M1400 加速器, OpenAI 夥伴', 
    'URA': '鈾實物需求週期, 長線供應缺口'
}

# --- 2. 數據抓取引擎 (功能 1, 2, 3, 4, 5, 6, 17, 18) ---
@st.cache_data(ttl=600)
def fetch_everything(tickers):
    """強化版數據抓取：修復 URA/0050 並獲取所有指標"""
    processed = [t.strip().upper() for t in tickers if t]
    # 基準與宏觀標的
    benchmarks = ['QQQ', 'QLD', 'TQQQ', '0050.TW', 'BTC-USD', '^VIX', '^MOVE']
    all_list = list(set(processed + benchmarks))
    
    # 抓取 2 年資料支援 Pi Cycle (功能 6)
    df = yf.download(all_list, period="2y", auto_adjust=True, progress=False)
    prices = df['Close'].ffill().bfill() # 功能 20: 補點技術
    
    earnings, news_data = {}, {}
    for t in processed:
        try:
            tk = yf.Ticker(t)
            # 財報倒數修復 (功能 17: 取消 999)
            cal = tk.calendar
            if cal is not None and not cal.empty:
                earnings[t] = cal.loc['Earnings Date'].iloc[0]
            # 新聞抓取 (功能 18)
            news_data[t] = tk.news[:3]
        except: earnings[t] = None
    return prices, earnings, news_data

# --- 3. 凱利與三維預測核心 (功能 10, 11, 14, 15) ---
def run_quantum_audit(series, qld_prices):
    """
    功能 10: 60日自適應 Half-Kelly
    功能 11: 7% 移動止損
    功能 14, 15: 1w 與 1m 預測
    """
    curr = series.iloc[-1]
    
    # A. 60日自適應 Half-Kelly
    # 理論依據: $K = (W - (1-W)/R) \times 0.5$
    rets = series.pct_change().shift(-5)
    ema20 = series.ewm(span=20).mean()
    sig = series > ema20
    # 僅取最近 60 個符合信號的樣本，確保動態性
    v_rets = rets[sig].tail(60).dropna()
    
    if len(v_rets) > 5:
        win_p = (v_rets > 0).mean()
        pos_avg = v_rets[v_rets > 0].mean() if not v_rets[v_rets > 0].empty else 0.01
        neg_avg = abs(v_rets[v_rets < 0].mean()) if not v_rets[v_rets < 0].empty else 0.01
        r_ratio = pos_avg / neg_avg
        kelly = max(0, (win_p - (1 - win_p) / r_ratio) * 0.5)
    else: kelly, win_p = 0.0, 0.5

    # B. 1w Expected Move (IV 邏輯)
    # $Price \times \sigma \times \sqrt{7/365}$
    vol = series.pct_change().tail(30).std() * np.sqrt(252)
    move_1w = curr * vol * np.sqrt(7/365)
    
    # C. 1m 線性回歸預測
    y = series.tail(60).values.reshape(-1, 1)
    x = np.array(range(len(y))).reshape(-1, 1)
    reg = LinearRegression().fit(x, y)
    pred_1m = reg.predict([[len(y) + 22]])[0][0]
    
    # D. 7% 移動止損 (功能 11)
    peak = series.tail(252).max()
    t_stop = peak * 0.93
    
    # E. 效率判定 (功能 7)
    efficiency = "🚀 高效" if (series/qld_prices).iloc[-1] > (series/qld_prices).iloc[-20] else "🐌 低效"
    
    return {
        "kelly": kelly, "win_p": win_p, 
        "range_1w": (curr - move_1w, curr + move_1w),
        "pred_1m": pred_1m, "t_stop": t_stop, "eff": efficiency
    }

# --- 4. 側邊欄：實戰輸入 Form (功能 5, 20) ---
with st.sidebar.form("posa_input_form"):
    st.header("💰 12.7萬實戰資產配置")
    # 初始化預設持倉
    if 'portfolio_df' not in st.session_state:
        st.session_state.portfolio_df = pd.DataFrame([
            {"代號": "MU", "金額": 30000}, {"代號": "AMD", "金額": 25000},
            {"代號": "URA", "金額": 15000}, {"代號": "0050.TW", "金額": 40000},
            {"代號": "BTC-USD", "金額": 57000}
        ])
    
    edited_df = st.data_editor(st.session_state.portfolio_df, num_rows="dynamic")
    submit_btn = st.form_submit_button("🚀 確認並執行 20 項全方位審計")

# --- 5. 數據執行鎖定邏輯 (功能 20) ---
if submit_btn or st.session_state.prices is not None:
    if submit_btn:
        # 當按下按鈕，執行模組一的 fetch_everything
        with st.spinner('會計師正在查核數據中...'):
            st.session_state.user_tickers = edited_df["代號"].dropna().tolist()
            p, e, n = fetch_everything(st.session_state.user_tickers)
            st.session_state.prices = p
            st.session_state.earnings = e
            st.session_state.news = n
            # 抓取宏觀數據
            liq_val, btcd_val, mvrv_val = fetch_macro_onchain()
            st.session_state.macro = {"liq": liq_val, "btcd": btcd_val, "mvrv": mvrv_val}

    # 讀取緩存數據，防止切換當機
    prices = st.session_state.prices
    earnings = st.session_state.earnings
    macro = st.session_state.macro
    tickers = st.session_state.user_tickers

    # --- 6. 頂部看板：宏觀地基 (功能 1, 2, 3, 4) ---
    st.subheader("🌐 全球週期與地基審計 (Macro Ground)")
    col1, col2, col3, col4 = st.columns(4)
    
    # 功能 1: 淨流動性定義說明
    col1.metric("淨流動性 (Net Liquidity)", f"${macro['liq']:,.2f}B")
    with col1:
        st.caption("📖 **定義**：聯準會總資產(WALCL) - 財政部帳戶(TGA) - 逆回購(RRP)。這是市場真實的「含氧量」。")

    # 功能 4: MVRV 真實數據
    col2.metric("MVRV 週期溫度", f"{macro['mvrv']:.2f}", 
                delta="⚠️ 利好出盡" if macro['mvrv'] > 3.0 else "✅ 週期安全")
    
    # 功能 2, 3: 股債雙天氣
    col3.metric("VIX / MOVE (股債天氣)", f"{prices['^VIX'].iloc[-1]:.1f} / {prices['^MOVE'].iloc[-1]:.0f}")
    
    # 功能 5: BTC.D
    col4.metric("BTC.D (資金羅盤)", f"{macro['btcd']:.1f}%")

    # --- 7. 比特幣週期警報 (功能 6: Pi Cycle Top) ---
    st.divider()
    btc_series = prices['BTC-USD']
    ma111 = btc_series.rolling(111).mean()
    ma350_x2 = btc_series.rolling(350).mean() * 2
    
    st.subheader("🔮 週期逃命指標：Pi Cycle Top Indicator")
    c_p1, c_p2, c_p3 = st.columns([1, 1, 2])
    c_p1.metric("比特幣當前價格", f"${btc_series.iloc[-1]:,.0f}")
    c_p2.metric("Pi 頂部壓力線", f"${ma350_x2.iloc[-1]:,.0f}")
    
    if ma111.iloc[-1] > ma350_x2.iloc[-1]:
        c_p3.error("🚨 **終極警報：PI CYCLE TOP 交叉！** 牛市可能已見頂，建議執行大規模獲利了結。")
    else:
        c_p3.success("✅ **週期運行中**：目前 111DMA 尚未交叉 350DMA*2，預期頂部仍有空間。")

    # --- 8. 即時市場脈搏 (功能 6, 11) ---
    st.subheader("⚡ 即時市場脈搏 (Real-time Pulse)")
    display_list = [t for t in tickers if t in prices.columns]
    for i in range(0, len(display_list), 4):
        cols = st.columns(4)
        for j, t in enumerate(display_list[i:i+4]):
            cp = prices[t].iloc[-1]
            chg = (prices[t].iloc[-1]/prices[t].iloc[-2]-1)*100
            cols[j].metric(t, f"${cp:,.2f}", f"{chg:.2f}%")

# --- 9. 深度審計大表 (功能 7, 8, 10, 13, 14, 17) ---
    st.divider()
    st.subheader("📋 跨市場深度審計 (凱利權重與三維預測)")
    
    audit_results = []
    today = datetime.now().date()
    
    # 從緩存中讀取數據
    p_data = st.session_state.prices
    e_data = st.session_state.earnings
    n_data = st.session_state.news
    
    for t in tickers:
        if t in p_data.columns and t not in ['QQQ', 'QLD', 'TQQQ']:
            # 呼叫模組一的運算引擎
            res = run_quantum_audit(p_data[t], p_data['QLD'])
            
            # 處理財報倒數 (修正功能 17: 取消 999)
            ed = e_data.get(t)
            if ed:
                # 確保 ed 是 date 物件
                target_date = ed.date() if hasattr(ed, 'date') else ed
                days_rem = (target_date - today).days
                e_display = f"⚠️ {days_rem}d" if days_rem <= 7 else f"{days_rem}d"
            else:
                e_display = "無資料" # 徹底修復 999 問題

            audit_results.append({
                "標的": t,
                "SA觀點": SA_INSIGHTS.get(t, "實務資產"),
                "效率": res['eff'],
                "20EMA": "🟢 站穩" if p_data[t].iloc[-1] > p_data[t].ewm(span=20).mean().iloc[-1] else "🔴 跌破",
                "凱利權重": f"{res['kelly']*100:.1f}%",
                "1w預期區間": f"{res['range_1w'][0]:.1f} - {res['range_1w'][1]:.1f}",
                "1m回歸目標": f"{res['pred_1m']:.1f}",
                "移動止損": f"${res['t_stop']:.1f}",
                "財報": e_display
            })
    
    st.table(pd.DataFrame(audit_results))

    # --- 10. 視覺化分析：熱力圖與生命線 (功能 9, 13, 15, 20) ---
    st.divider()
    col_left, col_right = st.columns([1, 1.2])
    
    with col_left:
        st.subheader("🤝 板塊相關性與風險分散審計")
        # 計算相關性
        corr_matrix = p_data[tickers].corr()
        st.plotly_chart(px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale='RdBu_r'), use_container_width=True)
        
        # 功能 13: 相關性文字解釋
        st.markdown("#### 📖 審計分析結論")
        # 簡單邏輯判定：找出一對最高相關
        high_corr_pairs = []
        for i in range(len(tickers)):
            for j in range(i+1, len(tickers)):
                if corr_matrix.iloc[i,j] > 0.8:
                    high_corr_pairs.append(f"{tickers[i]} & {tickers[j]}")
        
        if high_corr_pairs:
            st.warning(f"🚨 **過度集中警告**：{', '.join(high_corr_pairs)} 相關性過高 (>0.8)，代表風險高度重疊，凱利配置應進一步縮減。")
        else:
            st.success("✅ **配置健康**：目前持倉標的分散度良好，受單一板塊崩跌影響較低。")

    with col_right:
        st.subheader("📈 20EMA 趨勢審核 (圖表鎖定版)")
        # 功能 20: 增加 key 避免與其他組件衝突，確保切換不當機
        pick = st.selectbox("選擇要深度審核的標的", tickers, key="posa_chart_selector")
        if pick in p_data.columns:
            fig = go.Figure()
            # 股價與 20EMA
            fig.add_trace(go.Scatter(x=p_data.index, y=p_data[pick], name="現價", line=dict(color='gold', width=2)))
            fig.add_trace(go.Scatter(x=p_data.index, y=p_data[pick].ewm(span=20).mean(), name="20EMA生命線", line=dict(color='white', dash='dash')))
            # 標註移動止損位 (功能 11)
            t_stop_val = p_data[pick].tail(252).max() * 0.93
            fig.add_hline(y=t_stop_val, line_dash="dot", line_color="red", annotation_text="7% 移動止損位")
            
            fig.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True)

    # --- 11. 新增：標的新聞整合 (功能 16) ---
    st.divider()
    st.subheader("📰 標的即時核心情報 (量化消息過濾)")
    n_cols = st.columns(len(tickers))
    for i, t in enumerate(tickers):
        with n_cols[i]:
            st.write(f"**{t}**")
            if t in n_data:
                for news in n_data[t][:2]: # 僅顯示前兩則最核心新聞
                    st.caption(f"🔗 [{news['title']}]({news['link']})")

    # --- 12. 旗艦決策手冊 (功能 15) ---
    st.divider()
    with st.expander("📚 Posa 旗艦審計決策手冊"):
        st.markdown(f"""
        ### 1. 趨勢與預判邏輯 (Future Forecast)
        * **1w Expected Move (一週預測)**：基於 Black-Scholes 模型：$Price \pm (Price \times \sigma \times \sqrt{{7/365}})$. 
          這代表統計學上 68% 的正常波動邊界，跌破則視為異常趨勢。
        * **1m Regression (一個月預測)**：利用 60 交易日線性回歸：$y = ax + b$. 推估價格在慣性下的運動路徑。

        ### 2. 風險控制 (Risk Control)
        * **動態凱利 (60d Adaptive)**：$K = (W - \\frac{{1-W}}{{R}}) \\times 0.5$. 
          採用 60 天短視窗以適應市場快速變遷，0.5 係數用於對抗市場非正態分佈的肥尾風險。
        * **移動止損 (Trailing Stop)**：取過去 252 交易日（約一年）之最高收盤價，向下回撤 7% 為強制撤退點。

        ### 3. 利好出盡與週期 (Cycle Top)
        * **Pi Cycle Top Indicator**：當 $111-day DMA > 350-day DMA \\times 2$。
          歷史證明這是比特幣在消息瘋狂（利好出盡）時的終極頂部訊號。
        * **MVRV Ratio**：當數值 $> 3.0$ 時，代表持有者獲利豐厚，市場隨時可能發生集體踐踏式撤退。
        """)
else:
    # 初始進入頁面
    st.info("💡 請在左側輸入 資產配置代號，並點擊『🚀 確認並執行全方位審計』。")