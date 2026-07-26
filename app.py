import os
import datetime
import yfinance as yf
import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, HTMLResponse, JSONResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

# ---------------------------------------------------------
# 1. データ取得・テクニカル分析ロジック
# ---------------------------------------------------------
def fetch_jpx_data(ticker_symbol: str, interval: str = "1d", period: str = "3mo") -> pd.DataFrame:
    """東証銘柄の株価データを取得"""
    formatted_ticker = f"{ticker_symbol}.T" if not ticker_symbol.endswith(".T") else ticker_symbol
    try:
        stock = yf.Ticker(formatted_ticker)
        df = stock.history(period=period, interval=interval)
        return df if not df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def check_daily_overbought(df_daily: pd.DataFrame, dev_threshold: float = 15.0, rsi_threshold: float = 70.0):
    """日足の過熱感（25日乖離率・RSI）を判定"""
    if df_daily.empty or len(df_daily) < 25:
        return False, 0.0, 0.0

    df = df_daily.copy()
    df['SMA25'] = df['Close'].rolling(window=25).mean()
    latest_close = df['Close'].iloc[-1]
    latest_sma25 = df['SMA25'].iloc[-1]
    dev_rate = ((latest_close - latest_sma25) / latest_sma25) * 100

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    latest_rsi = rsi.iloc[-1]

    is_overbought = (dev_rate >= dev_threshold) and (latest_rsi >= rsi_threshold)
    return is_overbought, round(dev_rate, 2), round(latest_rsi, 2)

def detect_chart_patterns(df_5m: pd.DataFrame) -> list:
    """ローソク足パターン検出"""
    patterns = []
    if df_5m.empty or len(df_5m) < 5:
        return patterns

    latest = df_5m.iloc[-1]
    prev = df_5m.iloc[-2]

    body = abs(latest['Close'] - latest['Open'])
    upper_shade = latest['High'] - max(latest['Open'], latest['Close'])
    if upper_shade > (body * 2) and latest['Close'] < latest['Open']:
        patterns.append("上ヒゲ大陰線（上値抵抗感）")

    if prev['High'] > latest['High'] and latest['Close'] < prev['Close']:
        patterns.append("高値切り下げ（失速）")

    return patterns

def analyze_5m_short_signal(ticker_symbol: str, name: str):
    """VWAP下抜けおよび前日・当日の成功・失敗判定"""
    df_5m = fetch_jpx_data(ticker_symbol, interval="5m", period="5d")
    if df_5m.empty or len(df_5m) < 10:
        return None

    df_5m['TP'] = (df_5m['High'] + df_5m['Low'] + df_5m['Close']) / 3
    df_5m['PV'] = df_5m['TP'] * df_5m['Volume']
    df_5m['Date'] = df_5m.index.date
    cum_pv = df_5m.groupby('Date')['PV'].cumsum()
    cum_vol = df_5m.groupby('Date')['Volume'].cumsum()
    df_5m['VWAP'] = cum_pv / cum_vol

    # VWAP下抜けシグナルの検出
    df_5m['is_break'] = (df_5m['Close'].shift(1) >= df_5m['VWAP'].shift(1)) & (df_5m['Close'] < df_5m['VWAP'])

    dates = sorted(list(set(df_5m['Date'])))
    if len(dates) < 1:
        return None

    latest_date = dates[-1]
    prev_date = dates[-2] if len(dates) >= 2 else None

    today_breaks = df_5m[df_5m['Date'] == latest_date]['is_break'].any()
    prev_breaks = df_5m[df_5m['Date'] == prev_date]['is_break'].any() if prev_date else False

    if not today_breaks and not prev_breaks:
        return None

    signal_timing = "当日検出" if today_breaks else "前日検出"

    # --- 成功・結果判定ロジック ---
    target_date = latest_date if today_breaks else prev_date
    df_target = df_5m[df_5m['Date'] == target_date]
    
    # シグナルが出た最初の足
    break_rows = df_target[df_target['is_break']]
    if not break_rows.empty:
        break_price = break_rows.iloc[0]['Close']
        target_price = break_price * 0.98  # -2%下落を目標とする
        stop_price = break_price * 1.015   # +1.5%上昇で損切

        # シグナル発生以降の安値と高値
        after_break_df = df_target.loc[break_rows.index[0]:]
        min_price = after_break_df['Low'].min()
        max_price = after_break_df['High'].max()

        if min_price <= target_price:
            status_result = "【成功】利確達成"
        elif max_price >= stop_price:
            status_result = "【失敗】損切"
        else:
            status_result = "【継続中】含み益/推移中"
    else:
        status_result = "判定中"

    latest = df_5m.iloc[-1]
    patterns = detect_chart_patterns(df_5m)
    latest_close = latest['Close']

    return {
        "code": ticker_symbol,
        "name": name,
        "price": round(latest_close, 1),
        "vwap": round(latest['VWAP'], 1),
        "signal_timing": signal_timing,
        "status_result": status_result,
        "shape_patterns": patterns,
        "stop_loss": round(latest_close * 1.015),
        "take_profit": round(latest_close * 0.98)
    }

# ---------------------------------------------------------
# 2. Web API サーバー（FastAPI）
# ---------------------------------------------------------
JAPAN_STOCKS = {
    "6857": "アドバンテスト",
    "8035": "東京エレクトロン",
    "6146": "ディスコ",
    "9984": "ソフトバンクグループ",
    "7011": "三菱重工業",
    "6526": "ソシオネクスト"
}

@app.get("/api/signals")
def get_signals():
    results = []
    for code, name in JAPAN_STOCKS.items():
        df_daily = fetch_jpx_data(code, interval="1d", period="3mo")
        is_candidate, dev, rsi = check_daily_overbought(df_daily)
        if is_candidate:
            signal_data = analyze_5m_short_signal(code, name)
            if signal_data:
                signal_data["daily_dev"] = dev
                signal_data["daily_rsi"] = rsi
                results.append(signal_data)
    return JSONResponse(content=results)

@app.get("/api/chart/{code}")
def get_chart_data(code: str):
    df_5m = fetch_jpx_data(code, interval="5m", period="5d")
    if df_5m.empty:
        return JSONResponse(content={"candles": [], "vwap": []})

    df_5m['TP'] = (df_5m['High'] + df_5m['Low'] + df_5m['Close']) / 3
    df_5m['PV'] = df_5m['TP'] * df_5m['Volume']
    df_5m['Date'] = df_5m.index.date
    cum_pv = df_5m.groupby('Date')['PV'].cumsum()
    cum_vol = df_5m.groupby('Date')['Volume'].cumsum()
    df_5m['VWAP'] = cum_pv / cum_vol

    candles, vwap_list = [], []
    for index, row in df_5m.tail(80).iterrows():
        timestamp = int(index.timestamp())
        candles.append({
            "time": timestamp,
            "open": round(row['Open'], 1),
            "high": round(row['High'], 1),
            "low": round(row['Low'], 1),
            "close": round(row['Close'], 1)
        })
        vwap_list.append({
            "time": timestamp,
            "value": round(row['VWAP'], 1)
        })

    return JSONResponse(content={"candles": candles, "vwap": vwap_list})

@app.get("/")
def get_index():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    return HTMLResponse(content="<h1>index.html が見つかりません</h1>", status_code=404)
   

    
