import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from sklearn.linear_model import LinearRegression
from datetime import datetime, date

# --- 頁面配置 ---
st.set_page_config(page_title="Alpha 2.0 Pro Quant", layout="wide")

# --- 1. 專業級數據清洗 (防彈核心) ---
def module_integrity_pro(df_raw):
    """
    功能：強制扁平化 yfinance 的 MultiIndex，確保數據路徑暢通。
    """
    df = df_raw.copy()
    
    # 偵測並處理多層索引
    if isinstance(df.columns, pd.MultiIndex):
        # 如果第一層包含 'Adj Close'，只取這一層
        if 'Adj Close' in df.columns.levels[0]:
            df = df['Adj Close']
        # 否則嘗試取最後一層 (通常是 Ticker)
        else:
            df.columns = df.columns.get_level_values(-1)
    
    # 移除全空列並填補數據
    df = df.ffill().dropna(how='all')
    
    # 基準資產檢查
    if 'QQQ' not in df.columns:
        return None, "❌ 嚴重錯誤：數據中找不到 QQQ。請務必在側邊欄勾選 QQQ 作為 Beta 基準。"
    
    # 清除無限大與非數值
    clean_df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    if clean_df.empty:
        return None, "❌ 數據清洗後為空，請檢查網路連線或資產代號。"
        
    return clean_df, None

# --- 2. 趨勢預測與線性回歸 (Core Projection) ---
def module_core_pro(df):
    """
    依據 邏輯計算 k 值、效率係數與 1M/3M 目標價。
    """
    # 準備 X, Y 數據
    y = df['QQQ'].values.reshape(-1, 1)
    x = np.arange(len(y)).reshape(-1, 1)
    
    # 線性回歸模型
    model = LinearRegression().fit(x, y)
    
    # 提取關鍵指標 (強制轉為 float 標量，防止 ValueError)
    k = float(model.coef_[0][0])
    eff = float(model.score(x, y))
    
    # 時間序列預測
    ts_p = model.predict(x).flatten()
    
    # 未來價格預測 
    p1 = float(model.predict([[len(y) + 22]])[0][0]) # 1 Month
    p3 = float(model.predict([[len(y) + 66]])[0][0]) # 1 Quarter
    
    # 20EMA 趨勢判定 
    ema20_series = df['QQQ'].ewm(span=20).mean()
    current_price = float(df['QQQ'].iloc[-1])
    last_ema = float(ema20_series.iloc[-1])
    
    if current_price > last_ema and k > 0:
        status = "🔥 加速上升"
    elif current_price < last_ema:
        status = "🛑 趨勢損毀"
    else:
        status = "🛡️ 區間盤整"
        
    return {
        "k": k, "eff": eff, "p1": p1, "p3": p3, 
        "ts_p": ts_p, "status": status, "ema20": ema20_series
    }

# --- 3. 組合審計與資金權重 (Portfolio Logic) ---
def module_portfolio_pro(df, core, investments):
    """
    依據 邏輯計算選股等級與 Kelly 倉位。
    """
    # 計算總回報 (轉為 Dictionary 以避開 Series 比較錯誤)
    rets_dict = df.pct_change().dropna().sum().to_dict()
    eps = 1e-12 # 極小值防禦除以零
    
    # 找出主要對標資產 (非指數類)
    indices = ['QQQ', 'QLD', 'TQQQ', 'BTC-USD']
    target = next((a for a in investments.keys() if a not in indices), 'QQQ')
    
    # 提取標量數值
    v_target = float(rets_dict.get(target, 0))
    v_qld = float(rets_dict.get('QLD', eps))
    v_tqqq = float(rets_dict.get('TQQQ', eps))
    
    # 選股等級對標 
    if v_target > v_tqqq:
        grade = "Alpha+ (跑贏 3 倍)"
    elif v_target > v_qld:
        grade = "Beta+ (跑贏 2 倍)"
    else:
        grade = "Underperform (落後槓桿)"
        
    # 計算總資產
    total_cap = sum(investments.values()) if sum(investments.values()) > 0 else 1.0
    
    # Kelly 公式建議 
    win_rate = 0.6 if core['k'] > 0 else 0.4
    kelly = np.clip((win_rate * 2 - 1), 0, 0.75)
    
    return {
        "grade": grade, "target": target, "total": total_cap, 
        "kelly": kelly, "weights": {k: v/total_cap for k, v in investments.items()}
    }

# --- 4. FRED 宏觀與外部因子 (External Audit) ---
def module_fred_audit(df, api_key, exit_date):
    """
    依據 計算 Importance Score 與 Pi Cycle。
    """
    res = {"imp_score": 0.0, "fed_rate": 0.0, "pi_top": False}
    
    # FRED API 數據獲取
    if api_key:
        try:
            fred = Fred(api_key=api_key)
            # 獲取聯邦基金利率 (FEDFUNDS)
            fed_data = fred.get_series('FEDFUNDS', limit=1)
            if not fed_data.empty:
                rate = float(fed_data.iloc[-1])
                res['fed_rate'] = rate
                # Importance Score 公式 
                # 假設市場共識 4.5%，波動敏感度 1.5
                res['imp_score'] = abs(rate - 4.5) * 1.5
        except Exception as e:
            # 靜默失敗，不讓主程式崩潰，但記錄錯誤
            print(f"FRED API Error: {e}")
            
    # Pi Cycle Top (BTC) 
    if 'BTC-USD' in df.columns:
        ma111 = float(df['BTC-USD'].rolling(111).mean().iloc[-1])
        ma350_x2 = float(df['BTC-USD'].rolling(350).mean().iloc[-1]) * 2
        res['pi_top'] = ma111 > ma350_x2
        
    # 撤退倒數計時 
    today = date(2026, 1, 15) # 模擬當前時間
    days_left = (exit_date - today).days
    res['exit_factor'] = np.clip(days_left / 136, 0.0, 1.0)
    res['days_left'] = days_left
    
    return res

# --- 5. 自動財報日 (Earnings) ---
def get_auto_earnings(ticker):
    """
    依據 邏輯，自動鎖定 2026 Q1 財報日。
    """
    # 模擬 2026 年初的財報行事曆
    calendar = {
        'AMD': '2026-01-27', 'NVDA': '2026-02-25', 'TSM': '2026-01-16',
        'QQQ': '2026-01-29', 'AAPL': '2026-01-30', 'MSFT': '2026-01-27'
    }
    return calendar.get(ticker.upper(), "N/A")

# --- UI 介面層 ---
st.sidebar.header("🎯 Alpha 2.0 調度中心")

# 使用 Form 避免重複刷新
with st.sidebar.form("pro_form"):
    # FRED API 輸入
    fred_key_input = st.text_input("FRED API Key (選填)", type="password", help="用於計算宏觀驚奇指數")
    
    # 資產選擇
    monitored = st.multiselect(
        "核心資產 (必須含 QQQ)", 
        ["QQQ","QLD","TQQQ","BTC-USD","AMD","NVDA","TSM","AAPL"], 
        default=["QQQ","QLD","TQQQ","AMD"]
    )
    
    st.markdown("---")
    st.write("💰 **持倉金額配置 (USD)**")
    
    # 動態生成金額輸入框
    invest_map = {}
    for asset in monitored:
        invest_map[asset] = st.number_input(f"{asset} 持倉", min_value=0, value=1000, step=100)
        
    # 日期設定
    exit_date_in = st.date_input("2026 清倉目標日", value=date(2026, 5, 31))
    
    # 執行按鈕
    submit_btn = st.form_submit_button("🚀 啟動 Alpha 2.0 審計")

# 主畫面渲染
st.title("🚀 Alpha 2.0 Pro: 進攻型深度審計 (2026 旗艦版)")

if submit_btn:
    # 1. 抓取數據
    with st.spinner('正在連線 Yahoo Finance 與 FRED 資料庫...'):
        try:
            # 下載 2024-2026 數據
            raw_data = yf.download(monitored, start="2024-01-01", end="2026-01-16", progress=False)
            
            if raw_data.empty:
                st.error("❌ 無法獲取數據，請檢查資產代號或網絡連線。")
            else:
                # 2. 數據清洗
                clean_df, err_msg = module_integrity_pro(raw_data)
                
                if err_msg:
                    st.error(err_msg)
                else:
                    # 3. 執行三大核心模組
                    core = module_core_pro(clean_df)
                    port = module_portfolio_pro(clean_df, core, invest_map)
                    ext = module_fred_audit(clean_df, fred_key_input, exit_date_in)
                    
                    # 4. 財報風險計算 
                    earn_date = get_auto_earnings(port['target'])
                    risk_tag = "SAFE"
                    if earn_date != "N/A":
                        days_to_earn = (datetime.strptime(earn_date, "%Y-%m-%d").date() - date(2026, 1, 15)).days
                        if days_to_earn <= 7: risk_tag = "⚠️ 高波動禁區"
                        elif days_to_earn <= 14: risk_tag = "🛡️ 觀察窗口"

                    # --- 儀表板呈現 ---
                    # 第一排：核心量化指標
                    k_col, mac_col, p1_col, tot_col = st.columns(4)
                    k_col.metric("進攻斜率 (k)", f"{core['k']:.2f}", delta=core['status'])
                    
                    # FRED 數據展示
                    fred_label = "FRED 驚奇指數"
                    if ext['imp_score'] == 0.0: fred_label += " (未連線)"
                    mac_col.metric(fred_label, f"{ext['imp_score']:.2f}", delta=f"利率: {ext['fed_rate']}%")
                    
                    p1_col.metric("1M 預測目標 (p1)", f"${core['p1']:.2f}")
                    tot_col.metric("總持倉價值", f"${port['total']:,.0f}")
                    
                    st.divider()
                    
                    # 第二排：資產與趨勢分析
                    left_c, right_c = st.columns(2)
                    
                    with left_c:
                        st.subheader(f"📊 選股等級：{port['grade']}")
                        st.caption(f"當前審計對象：**{port['target']}** (財報日: {earn_date} | 狀態: {risk_tag})")
                        
                        # 繪製權重分佈
                        w_df = pd.DataFrame(port['weights'].items(), columns=['Asset', 'Weight']).set_index('Asset')
                        st.bar_chart(w_df)
                        
                        if ext['pi_top']:
                            st.error("🚨 BTC Pi Cycle 觸發頂部信號 (111DMA > 350DMA*2)！建議開始減碼。")
                            
                    with right_c:
                        st.subheader("📈 QQQ 20EMA 趨勢生命線")
                        # 準備繪圖數據 (確保索引對齊)
                        chart_df = pd.DataFrame({
                            "實際價格": clean_df['QQQ'][-60:].values,
                            "20EMA": core['ema20'][-60:].values,
                            "線性預測": core['ts_p'][-60:]
                        })
                        st.line_chart(chart_df, color=["#FF4B4B", "#1F77B4", "#FFA500"])
                    
                    # 底部：撤退倒數
                    st.success(f"📍 戰略結論：Kelly 建議最大倉位 {port['kelly']:.0%}。距離 2026/05 撤退日剩餘 {ext['days_left']} 天。")
                    
        except Exception as e:
            st.error(f"系統運行時發生未預期的錯誤：{str(e)}")
            st.code(f"錯誤詳情：{e}") # 方便除錯
else:
    st.info("👋 請在左側輸入 FRED API Key (可選)、選擇資產並配置金額，最後按下 **「🚀 啟動 Alpha 2.0 審計」**。")