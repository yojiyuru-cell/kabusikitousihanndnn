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

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

@app.get("/api/signals")
def get_signals():
    stock_targets = get_jpx_stock_list(limit=100)
    results = []
    
    for code, name in stock_targets.items():
        try:
            df_5m = fetch_jpx_data(code, interval="5m", period="5d")
            if df_5m.empty or len(df_5m) < 20:
                continue

            # VWAP & 移動平均線 & RSIの計算
            df_5m['TP'] = (df_5m['High'] + df_5m['Low'] + df_5m['Close']) / 3
            df_5m['PV'] = df_5m['TP'] * df_5m['Volume']
            df_5m['Date'] = df_5m.index.date
            cum_pv = df_5m.groupby('Date')['PV'].cumsum()
            cum_vol = df_5m.groupby('Date')['Volume'].cumsum()
            df_5m['VWAP'] = cum_pv / cum_vol

            df_5m['SMA5'] = df_5m['Close'].rolling(window=5).mean()
            df_5m['SMA20'] = df_5m['Close'].rolling(window=20).mean()
            df_5m['RSI'] = calculate_rsi(df_5m['Close'], 14)

            curr = df_5m.iloc[-1]
            prev = df_5m.iloc[-2]

            close_p = curr['Close']
            open_p = curr['Open']
            high_p = curr['High']
            low_p = curr['Low']
            vwap_val = safe_float(curr['VWAP'])
            rsi_val = safe_float(curr['RSI'], 50.0)

            # --- 下落シグナル判定 ---
            drop_reasons = []
            score = 0

            # 条件1: 急騰後のVWAP割れ
            if close_p < vwap_val and prev['Close'] >= safe_float(prev['VWAP']):
                drop_reasons.append("急騰後VWAP下抜け")
                score += 3

            # 条件2: デッドクロス（SMA5がSMA20を下抜け）
            if prev['SMA5'] >= prev['SMA20'] and curr['SMA5'] < curr['SMA20']:
                drop_reasons.append("5分足デッドクロス")
                score += 2

            # 条件3: RSI高値警戒からの反落
            if safe_float(prev['RSI']) > 65 and rsi_val < safe_float(prev['RSI']):
                drop_reasons.append("RSI高値反落")
                score += 2

            # 条件4: 長い上ヒゲ（実体の1.5倍以上のヒゲ）
            body_size = abs(close_p - open_p)
            upper_wick = high_p - max(open_p, close_p)
            if upper_wick >= body_size * 1.5 and upper_wick > 0:
                drop_reasons.append("上ヒゲ（売り圧力強）")
                score += 2

            # シグナルが出ている銘柄のみ抽出
            if score > 0 or close_p < vwap_val:
                judgment = "🚨 強力下落サイン" if score >= 4 else ("⚠️ 下落注意" if score >= 2 else "🔹 弱含み")
                
                results.append({
                    "code": code,
                    "name": name,
                    "price": round(safe_float(close_p), 1),
                    "vwap": round(vwap_val, 1),
                    "rsi": round(rsi_val, 1),
                    "score": score,
                    "reasons": " / ".join(drop_reasons) if drop_reasons else "VWAP下回り",
                    "judgment": judgment,
                    "stop_loss": round(safe_float(close_p) * 1.015, 1),
                    "take_profit": round(safe_float(close_p) * 0.98, 1)
                })

        except Exception as e:
            print(f"銘柄スキップ {code}: {e}")
            continue

    # 下落スコアが高い順に並び替え
    results = sorted(results, key=lambda x: x['score'], reverse=True)
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
