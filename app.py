import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import datetime

def module_data_integrity(data_dict):
    """
    功能：殘差審計 (residual_audit) 與 缺口風險因子 (gap_risk_factor)
    """
    # 建立 DataFrame 並進行前值填充，解決 2026 年連假後的數據斷點
    df = pd.DataFrame(data_dict).ffill()
    
    # 檢測跳空缺口 (Gap Risk)
    df['gap_risk'] = df['QQQ'].pct_change().abs() > 0.03 # 漲跌幅 > 3% 定義為大跳空
    
    # 執行殘差審計：刪除無法計算的行
    clean_df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    # 樣本數檢查
    if len(clean_df) < 60:
        raise ValueError("Alpha 2.0 警告：有效樣本不足 60 日，審計無法啟動。")
        
    return clean_df

def module_core_projection(df):
    """
    功能：k, eff, p1, p3, ts_p, 以及未來數值預測 (1w, 1m, 1q)
    """
    y = df['QQQ'].values.reshape(-1, 1)
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    
    k = model.coef_[0][0] # 斜率
    eff = model.score(x, y) # R2 決定係數
    
    # 預測路徑生成 (ts_p)
    ts_p = model.predict(x).flatten()
    
    # 未來預測
    p_1w = model.predict([[len(y) + 5]])[0][0]  # 1-Week
    p_1m = model.predict([[len(y) + 22]])[0][0] # 1-Month (p1)
    p_1q = model.predict([[len(y) + 66]])[0][0] # 1-Quarter (p3)
    
    return {"k": k, "eff": eff, "p1": p_1m, "p3": p_1q, "ts_p": ts_p, "p_1w": p_1w}

def module_volatility_trend(df, core_results):
    """
    功能：l1~l3, h1~h3, 20EMA, Sparklines 視覺化數據
    """
    current_p = df['QQQ'].iloc[-1]
    ts_p = core_results['ts_p']
    std = np.std(df['QQQ'].values - ts_p)
    
    # 六維殼層
    shells = {
        'l1': core_results['p1'] - std,   'h1': core_results['p1'] + std,
        'l2': core_results['p1'] - 2*std, 'h2': core_results['p1'] + 2*std,
        'l3': core_results['p1'] - 3*std, 'h3': core_results['p1'] + 3*std
    }
    
    # 20EMA 與 生命線判定
    ema20 = df['QQQ'].ewm(span=20).mean().iloc[-1]
    
    trend_status = "🛡️ 區間盤整"
    if current_p > ema20:
        trend_status = "🔥 加速上升" if core_results['k'] > 0 else "🛡️ 盤整偏多"
    elif current_p < ema20:
        trend_status = "🛑 趨勢損毀"
        
    return {"shells": shells, "ema20": ema20, "status": trend_status}

def module_portfolio_logic(df, core_results):
    """
    功能：pQ, kelly_f, Alpha/Beta+ 判定
    """
    returns = df.pct_change().dropna()
    
    # 對標 QLD/TQQQ 效率
    # 如果個股(如AMD)回報 / QLD回報 < 1，代表效率低
    bench_qld = (returns['AMD'].sum() / returns['QLD'].sum()) if 'AMD' in df else 0
    bench_tqqq = (returns['AMD'].sum() / returns['TQQQ'].sum()) if 'AMD' in df else 0
    
    alpha_status = "Underperform"
    if bench_tqqq > 1: alpha_status = "Alpha+"
    elif bench_qld > 1: alpha_status = "Beta+"
    
    # pQ 因子：結合趨勢純度與槓桿效率
    pQ = core_results['eff'] * bench_qld
    
    # Kelly 倉位 (2026 修正版：考慮勝率與盈虧比)
    win_rate = 0.6 if core_results['k'] > 0 else 0.4
    kelly_f = np.clip((win_rate - (1 - win_rate)) / 1, 0, 0.7) # 最高 70% 倉位限制
    
    return {"pQ": pQ, "kelly": kelly_f, "alpha_grade": alpha_status}

def module_external_audit(df):
    """
    功能：btc_corr, Pi Cycle Top, MVRV 預判
    """
    # 1. Pi Cycle Top Indicator
    ma111 = df['BTC'].rolling(window=111).mean()
    ma350_2 = df['BTC'].rolling(window=350).mean() * 2
    pi_top_signal = ma111.iloc[-1] > ma350_2.iloc[-1]
    
    # 2. 跨資產相關性
    btc_corr = df['QQQ'].pct_change().corr(df['BTC'].pct_change())
    
    # 3. 2026 五月撤退倒數
    target_date = datetime(2026, 5, 31)
    current_date = datetime(2026, 1, 15)
    days_left = (target_date - current_date).days
    exit_countdown = np.clip(days_left / 136, 0, 1) # 權重衰減
    
    return {"pi_top": pi_top_signal, "btc_corr": btc_corr, "exit_factor": exit_countdown}

def module_risk_monitoring(earnings_date_str, macro_data=None):
    """
    功能：Earnings Countdown, Importance Score
    """
    # 1. 財報監控
    today = datetime(2026, 1, 15)
    earn_date = datetime.strptime(earnings_date_str, "%Y-%m-%d")
    days_to_earn = (earn_date - today).days
    
    earn_risk = "SAFE"
    if days_to_earn <= 7: earn_risk = "⚠️ 高波動風險 (禁區)"
    elif days_to_earn <= 14: earn_risk = "🛡️ 觀察窗口 (準備減碼)"
    
    # 2. 消息量化 (Importance Score)
    # 公式：|Actual - Consensus| / Std * MarketSensitivity
    importance_score = 0
    if macro_data:
        surprise = abs(macro_data['actual'] - macro_data['consensus'])
        importance_score = (surprise / macro_data['std']) * macro_data['sensitivity']
        
    return {"earn_days": days_to_earn, "earn_risk": earn_risk, "news_score": importance_score}

