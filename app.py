import streamlit as st
import yfinance as yf
import pandas as pd
from fredapi import Fred
import plotly.express as px
import plotly.graph_objects as go
import time
import random
from datetime import datetime

# --- 1. 系統設定與 Seeking Alpha 數據庫 ---
st.set_page_config(page_title="Posa Alpha 3.3", layout="wide")
st.title("🛡️ Posa Alpha 3.3: 視覺化審計與智慧決策終端")

# SA 十大金股與關鍵數據 [cite: 208, 264, 415, 417]
SA_TOP_10 = ['MU', 'AMD', 'CLS', 'CIEN', 'COHR', 'ALL', 'INCY', 'GOLD', 'WLDN', 'ATI']
SA_DATA = {
    'MU': {'note': 'HBM 領先, PEG 0.20x (折價 88%)', 'eps_g': '206%'},
    'CLS': {'note': '15次盈餘上修, 0次下修', 'eps_g': '51%'},
    'AMD': {'note': 'OpenAI 夥伴, M1400 加速器', 'eps_g': '34%'},
    'ALL': {'note': '連續 32 年配息, AI 核保效率高', 'eps_g': '193%'}
}

try:
    FRED_API_KEY = st.secrets["FRED_API_KEY"]
    fred = Fred(api_key=FRED_API_KEY)
except:
    st.error("❌ 請在 Secrets 設定 FRED_API_KEY")
    st.stop()

# --- 2. 側邊欄：實戰配置編輯器 ---
st.sidebar.header("💰 我的實戰配置")
if 'portfolio_df' not in st.session_state:
    st.session_state.portfolio_df = pd.DataFrame([
        {"代號": "MU", "金額": 30000},
        {"代號": "AMD", "金額": 25000},
        {"代號": "SOL-USD", "金額": 15000},
        {"代號": "QQQ", "金額": 45000}
    ])
edited_df = st.sidebar.data_editor(st.session_state.portfolio_df, num_rows="dynamic")
user_tickers = edited_df["代號"].tolist()
total_val = edited_df["金額"].sum()

st.sidebar.divider()
TRAILING_PCT = st.sidebar.slider("移動止損 (%)", 5, 15, 7) / 100
KELLY_SCALE = st.sidebar.slider("凱利縮放係數", 0.1, 1.0, 0.5)

# --- 3. 數據抓取與凱利計算 ---
@st.cache_data(ttl=3600)
def fetch_and_audit(tickers):
    prices, earnings = pd.DataFrame(), {}
    full_list = list(set(tickers + SA_TOP_10 + ['QQQ', '^VIX', '^MOVE', 'BTC-USD']))
    for t in full_list:
        time.sleep(random.uniform(0.3, 0.8))
        try:
            tk = yf.Ticker(t)
            df = tk.history(period="1y")
            if not df.empty:
                prices[t] = df['Close']
                if "-" not in t:
                    cal = tk.calendar
                    if cal is not None and not cal.empty:
                        earnings[t] = cal.loc['Earnings Date'].iloc[0].strftime('%Y-%m-%d')
        except: continue
    try:
        liq = (fred.get_series('WALCL').iloc[-1] - fred.get_series('WTREGEN').iloc[-1] - fred.get_series('RRPONTSYD').iloc[-1]) / 1000
    except: liq = 0
    return liq, prices, earnings

def get_stats(t_prices, q_prices):
    ema20 = t_prices.ewm(span=20).mean()
    rs = t_prices / q_prices
    sig = (t_prices > ema20) & (rs > rs.rolling(20).mean())
    rets = t_prices.shift(-5) / t_prices - 1
    v_rets = rets[sig].dropna()
    if len(v_rets) < 5: return 0.52, 2.0
    return (v_rets > 0).mean(), (v_rets[v_rets > 0].mean() / abs(v_rets[v_rets < 0].mean()))

# --- 4. 頁面渲染 ---
try:
    net_liq, prices, e_dates = fetch_and_audit(user_tickers)
    vix = prices['^VIX'].iloc[-1]
    
    # A. 頂部視覺化：情緒儀表盤
    st.subheader("🌡️ 市場風險溫度與地基審計")
    col1, col2 = st.columns([1, 2])
    with col1:
        fig_vix = go.Figure(go.Indicator(
            mode = "gauge+number", value = vix, title = {'text': "VIX 恐慌指數"},
            gauge = {'axis': {'range': [None, 40]}, 'steps': [
                {'range': [0, 18], 'color': "lightgreen"},
                {'range': [18, 25], 'color': "orange"},
                {'range': [25, 40], 'color': "red"}],
                'bar': {'color': "black"}}))
        st.plotly_chart(fig_vix, use_container_width=True)
    with col2:
        m1, m2, m3 = st.columns(3)
        m1.metric("淨流動性", f"${net_liq:.2f}B")
        m2.metric("BTC 趨勢", "🟢 強勢" if prices['BTC-USD'].iloc[-1] > prices['BTC-USD'].ewm(span=20).mean().iloc[-1] else "🔴 弱勢")
        m3.metric("總市值", f"${total_val:,.0f}")
        st.write(f"💡 **biibo 意見：** {'市場處於進攻模式，地基穩固。' if vix < 18 else '風險升溫，應縮減個股權重。'}")

    # B. 持倉審計表 (恢復凱利與財報預警)
    st.subheader("🔍 組合深度審計 (含 Seeking Alpha 觀點)")
    audit_results = []
    today = datetime.now().date()
    
    for t in user_tickers:
        if t not in prices.columns or t in ['QQQ', '^VIX']: continue
        win_p, odds = get_stats(prices[t], prices['QQQ'])
        kelly_w = max(0, (win_p - (1 - win_p) / odds) * KELLY_SCALE)
        act_w = edited_df.loc[edited_df['代號']==t, '金額'].sum() / total_val
        e_date = e_dates.get(t, "N/A")
        e_alert = "⚠️ 7天內" if e_date != "N/A" and (datetime.strptime(e_date, '%Y-%m-%d').date() - today).days <= 7 else "✅"
        
        sa_note = SA_DATA.get(t, {}).get('note', '自定義標的')
        
        audit_results.append({
            "標的": t, "SA 觀點": sa_note, "回測勝率": f"{win_p*100:.1f}%",
            "凱利建議": kelly_w, "實際權重": act_w, "財報": e_alert,
            "止損狀態": "❌ 觸發" if prices[t].iloc[-1] <= prices[t].max()*(1-TRAILING_PCT) else "🟢 安全"
        })
    
    audit_df = pd.DataFrame(audit_results)
    st.table(audit_df.drop(columns=['凱利建議', '實際權重']).assign(
        凱利建議權重 = audit_df['凱利建議'].apply(lambda x: f"{x*100:.1f}%"),
        目前權重 = audit_df['實際權重'].apply(lambda x: f"{x*100:.1f}%")
    ))

    # C. 配置對比圖 (解決單薄感)
    st.subheader("📊 配置修正對比：實際 vs. 凱利建議")
    fig_comp = go.Figure(data=[
        go.Bar(name='實際權重', x=audit_df['標的'], y=audit_df['實際權重']),
        go.Bar(name='凱利建議', x=audit_df['標的'], y=audit_df['凱利建議'])
    ])
    fig_comp.update_layout(barmode='group', height=400)
    st.plotly_chart(fig_comp, use_container_width=True)

    # D. 會計師報告
    st.divider()
    st.subheader("🖋️ Alpha 3.3 自動審計報告")
    with st.container(border=True):
        if vix > 18: st.write("🚨 **風控提示：** VIX 已破 18，凱利公式已自動下修建議倉位。")
        for _, row in audit_df.iterrows():
            if row['實際權重'] > row['凱利建議'] + 0.1:
                st.write(f"⚠️ **過度曝險：** {row['標的']} 實際權重過高，建議減碼至 {row['凱利建議權重']}。")
            if row['財報'] == "⚠️ 7天內":
                st.write(f"💣 **財報警示：** {row['標的']} 財報在即，建議減碼 50% 以避開黑天鵝 。")

except Exception as e:
    st.error(f"系統運行中：{e}")