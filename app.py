import os
import math
import pandas as pd
import yfinance as yf
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

def safe_float(val, default=0.0):
    try:
        f = float(val)
        return default if math.isnan(f) or math.isinf(f) else f
    except Exception:
        return default

def get_jpx_stock_list(limit: int = 100):
    url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
    try:
        df = pd.read_excel(url)
        df_stocks = df[['コード', '銘柄名', '市場・商品区分']].dropna()
        
        stock_dict = {}
        for _, row in df_stocks.iterrows():
            code = str(int(row['コード']))
            name = str(row['銘柄名'])
            market = str(row['市場・商品区分'])
            
            if any(m in market for m in ['プライム', 'スタンダード', 'グロース']):
                stock_dict[code] = name
                if len(stock_dict) >= limit:
                    break
                    
        return stock_dict
    except Exception as e:
        print(f"JPX銘柄リスト取得エラー: {e}")
        return {
            "6857": "アドバンテスト", "8035": "東京エレクトロン", "6146": "ディスコ",
            "9984": "ソフトバンクG", "7011": "三菱重工", "6526": "ソシオネクスト",
            "7203": "トヨタ自動車", "8306": "三菱UFJ", "6758": "ソニーG", "6723": "ルネサス"
        }

def fetch_jpx_data(ticker_symbol: str, interval: str = "5m", period: str = "5d") -> pd.DataFrame:
    formatted_ticker = f"{ticker_symbol}.T" if not ticker_symbol.endswith(".T") else ticker_symbol
    try:
        stock = yf.Ticker(formatted_ticker)
        df = stock.history(period=period, interval=interval)
        return df if not df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

@app.get("/api/signals")
def get_signals():
    stock_targets = get_jpx_stock_list(limit=100)
    results = []
    
    for code, name in stock_targets.items():
        try:
            # 5分足データの取得
            df_5m = fetch_jpx_data(code, interval="5m", period="5d")
            if df_5m.empty or len(df_5m) < 10:
                continue

            # VWAPの計算
            df_5m['TP'] = (df_5m['High'] + df_5m['Low'] + df_5m['Close']) / 3
            df_5m['PV'] = df_5m['TP'] * df_5m['Volume']
            df_5m['Date'] = df_5m.index.date
            cum_pv = df_5m.groupby('Date')['PV'].cumsum()
            cum_vol = df_5m.groupby('Date')['Volume'].cumsum()
            df_5m['VWAP'] = cum_pv / cum_vol
            
            # 直近2本のローソク足を取得
            prev_candle = df_5m.iloc[-2]
            curr_candle = df_5m.iloc[-1]

            open_p  = curr_candle['Open']
            close_p = curr_candle['Close']
            high_p  = curr_candle['High']
            low_p   = curr_candle['Low']
            vwap_val = safe_float(curr_candle['VWAP'])

            p_open_p  = prev_candle['Open']
            p_close_p = prev_candle['Close']

            # --- チャートの形 判定ロジック ---
            body_size = abs(close_p - open_p)              # 実体の長さ
            upper_wick = high_p - max(open_p, close_p)    # 上ヒゲの長さ

            # パターン1: 長い上ヒゲ（実体の2倍以上の上ヒゲ）
            is_upper_wick = (upper_wick >= body_size * 2) and (upper_wick > 0)

            # パターン2: 陰線包み足（前回の陽線を今回の大きな陰線が包み込む）
            is_prev_bull = p_close_p > p_open_p
            is_curr_bear = close_p < open_p
            is_engulfing = is_prev_bull and is_curr_bear and (open_p >= p_close_p) and (close_p <= p_open_p)

            # 条件: VWAPより下に位置しているか
            is_below_vwap = close_p < vwap_val

            # --- 総合判定 ---
            if is_below_vwap and is_upper_wick:
                judgment = "🚨 天井サイン（長い上ヒゲ＋VWAP割れ）"
            elif is_below_vwap and is_engulfing:
                judgment = "🚨 転換サイン（包み足＋VWAP割れ）"
            elif is_below_vwap:
                judgment = "🔹 VWAP下回り"
            elif is_upper_wick:
                judgment = "🟡 天井気配（上ヒゲあり・VWAP上）"
            else:
                judgment = "⚪️ 通常"

            results.append({
                "code": code,
                "name": name,
                "price": round(safe_float(close_p), 1),
                "vwap": round(vwap_val, 1),
                "daily_dev": 0,  # 非表示用互換
                "daily_rsi": 0,  # 非表示用互換
                "signal_timing": "VWAP下回り" if is_below_vwap else "VWAP上回り",
                "daily_judgment": judgment,
                "stop_loss": round(safe_float(close_p) * 1.015),
                "take_profit": round(safe_float(close_p) * 0.98)
            })
        except Exception as e:
            print(f"銘柄スキップ {code}: {e}")
            continue

    return JSONResponse(content=results)

@app.get("/api/chart/{code}")
def get_chart_data(code: str):
    try:
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
                "open": round(safe_float(row['Open']), 1),
                "high": round(safe_float(row['High']), 1),
                "low": round(safe_float(row['Low']), 1),
                "close": round(safe_float(row['Close']), 1)
            })
            vwap_list.append({
                "time": timestamp,
                "value": round(safe_float(row['VWAP']), 1)
            })

        return JSONResponse(content={"candles": candles, "vwap": vwap_list})
    except Exception as e:
        print(f"チャートAPIエラー ({code}): {e}")
        return JSONResponse(content={"candles": [], "vwap": []})

@app.get("/")
def get_index():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>index.html が見つかりません</h1>", status_code=404)
