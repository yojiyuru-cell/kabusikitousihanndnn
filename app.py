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

def fetch_jpx_data(ticker_symbol: str, interval: str = "1d", period: str = "3mo") -> pd.DataFrame:
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
            # --- 1. 日足分析 ---
            df_daily = fetch_jpx_data(code, interval="1d", period="3mo")
            if df_daily.empty or len(df_daily) < 25:
                continue

            df_daily['SMA25'] = df_daily['Close'].rolling(window=25).mean()
            latest_close = df_daily['Close'].iloc[-1]
            latest_sma25 = df_daily['SMA25'].iloc[-1]
            
            dev_rate = ((latest_close - latest_sma25) / latest_sma25) * 100 if latest_sma25 else 0

            delta = df_daily['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            latest_rsi = rsi.iloc[-1]

            # --- 2. 5分足分析 ---
            df_5m = fetch_jpx_data(code, interval="5m", period="5d")
            vwap_val = 0
            vwap_status = "通常"
            
            if not df_5m.empty and len(df_5m) >= 5:
                df_5m['TP'] = (df_5m['High'] + df_5m['Low'] + df_5m['Close']) / 3
                df_5m['PV'] = df_5m['TP'] * df_5m['Volume']
                df_5m['Date'] = df_5m.index.date
                cum_pv = df_5m.groupby('Date')['PV'].cumsum()
                cum_vol = df_5m.groupby('Date')['Volume'].cumsum()
                df_5m['VWAP'] = cum_pv / cum_vol
                
                latest_5m = df_5m.iloc[-1]
                vwap_val = safe_float(latest_5m['VWAP'])
                
                if latest_5m['Close'] < vwap_val:
                    vwap_status = "VWAP下回り"
                else:
                    vwap_status = "VWAP上回り"

            # --- 3. 総合判定 ---
            is_overheated = (dev_rate >= 8.0) or (latest_rsi >= 65.0)
            is_vwap_below = (vwap_status == "VWAP下回り")

            if is_overheated and is_vwap_below:
                judgment = "🚨 空売りシグナル（過熱＋VWAP割れ）"
            elif is_overheated:
                judgment = "🟡 高値圏（監視）"
            elif is_vwap_below:
                judgment = "🔹 VWAP下回り"
            else:
                judgment = "⚪️ 通常"

            results.append({
                "code": code,
                "name": name,
                "price": round(safe_float(latest_close), 1),
                "vwap": round(vwap_val, 1),
                "daily_dev": round(safe_float(dev_rate), 2),
                "daily_rsi": round(safe_float(latest_rsi), 2),
                "signal_timing": vwap_status,
                "daily_judgment": judgment,
                "stop_loss": round(safe_float(latest_close) * 1.015),
                "take_profit": round(safe_float(latest_close) * 0.98)
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
