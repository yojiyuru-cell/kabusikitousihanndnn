import os
import math
import io
import re
import time
import threading
import requests
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

# メモリ内キャッシュ
CACHED_RESULTS = []
IS_SCANNING = False

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

def safe_float(val, default=0.0):
    try:
        f = float(val)
        return default if math.isnan(f) or math.isinf(f) else f
    except Exception:
        return default

def is_etf_or_fund(code: str, name: str) -> bool:
    etf_keywords = ["ETF", "ETN", "上場", "ファンド", "ブル", "ベア", "インデックス", "TOPIX", "日経225", "S&P", "NASDAQ", "225"]
    for kw in etf_keywords:
        if kw.lower() in str(name).lower():
            return True
    if re.match(r"^(13|14|15|16|20|25|26|28)\d{2}$", str(code)):
        return True
    return False

def get_all_jpx_stock_list():
    POPULAR_STOCKS = {
        "6857": "アドバンテスト", "8035": "東京エレクトロン", "6146": "ディスコ", 
        "9984": "ソフトバンクG", "7011": "三菱重工", "6526": "ソシオネクスト", 
        "6758": "ソニーG", "7203": "トヨタ", "8306": "三菱UFJ", "9104": "商船三井",
        "8002": "丸紅", "6367": "ダイキン", "6920": "レーザーテック", "7735": "スクリン",
        "4528": "小野薬品", "2413": "エムスリー", "4385": "メルカリ", "5253": "カバー",
        "9166": "GENDA", "5595": "QPS研究所", "7267": "ホンダ", "8316": "三井住友"
    }

    try:
        jpx_url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
        res = requests.get(jpx_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        res.raise_for_status()
        
        df = pd.read_excel(io.BytesIO(res.content))
        stock_dict = {}
        if 'コード' in df.columns and '銘柄名' in df.columns:
            for _, row in df.dropna(subset=['コード', '銘柄名']).iterrows():
                code_raw = str(row['コード']).split('.')[0].strip()
                name_raw = str(row['銘柄名']).strip()

                if len(code_raw) == 4 and code_raw.isdigit():
                    if not is_etf_or_fund(code_raw, name_raw):
                        stock_dict[code_raw] = name_raw

        if len(stock_dict) > 100:
            return stock_dict
    except Exception as e:
        print(f"⚠️ JPX取得失敗。標準リストを使用します: {e}")

    return POPULAR_STOCKS

def run_scan():
    global CACHED_RESULTS, IS_SCANNING
    if IS_SCANNING:
        return
    
    IS_SCANNING = True
    print("🔄 スキャンを開始します...")
    
    stock_targets = get_all_jpx_stock_list()
    codes = list(stock_targets.keys())
    results = []

    # 100銘柄ごとのバッチで安全に取得
    batch_size = 100
    for i in range(0, len(codes), batch_size):
        batch_codes = codes[i:i + batch_size]
        ticker_symbols = [f"{code}.T" for code in batch_codes]

        try:
            batch_data = yf.download(
                tickers=ticker_symbols,
                period="5d",
                interval="5m",
                group_by="ticker",
                threads=True,
                progress=False
            )
        except Exception as e:
            print(f"❌ バッチ取得失敗 ({i}): {e}")
            continue

        if batch_data.empty:
            continue

        for code in batch_codes:
            name = stock_targets[code]
            symbol = f"{code}.T"
            try:
                if len(ticker_symbols) > 1:
                    if symbol not in batch_data.columns.levels[0]:
                        continue
                    df_5m = batch_data[symbol].dropna(how="all")
                else:
                    df_5m = batch_data.dropna(how="all")

                if df_5m.empty or len(df_5m) < 5:
                    continue

                c0 = df_5m.iloc[-1]
                c1 = df_5m.iloc[-2] if len(df_5m) >= 2 else c0
                c2 = df_5m.iloc[-3] if len(df_5m) >= 3 else c0

                close_p = safe_float(c0['Close'])
                if close_p <= 0:
                    continue

                # VWAP計算
                df_5m['TP'] = (df_5m['High'] + df_5m['Low'] + df_5m['Close']) / 3
                df_5m['PV'] = df_5m['TP'] * df_5m['Volume']
                df_5m['Date'] = df_5m.index.date
                cum_pv = df_5m.groupby('Date')['PV'].cumsum()
                cum_vol = df_5m.groupby('Date')['Volume'].cumsum()
                df_5m['VWAP'] = cum_pv / (cum_vol + 1e-5)

                open_p   = safe_float(c0['Open'])
                high_p   = safe_float(c0['High'])
                low_p    = safe_float(c0['Low'])
                vwap_val = safe_float(df_5m['VWAP'].iloc[-1])
                prev_vwap = safe_float(df_5m['VWAP'].iloc[-2]) if len(df_5m) >= 2 else vwap_val

                patterns = []
                score = 0

                # 判定ロジック
                body_size = abs(close_p - open_p)
                upper_wick = high_p - max(open_p, close_p)
                if upper_wick >= max(body_size * 1.5, 3.0):
                    patterns.append("長い上ヒゲ（天井打）")
                    score += 3

                if (c1['Close'] > c1['Open']) and (close_p < open_p) and (open_p >= c1['Close']) and (close_p <= c1['Open']):
                    patterns.append("陰線包み足（強反落）")
                    score += 3

                if (c1['Close'] >= prev_vwap) and (close_p < vwap_val):
                    patterns.append("VWAP下抜け（デッドクロス）")
                    score += 3
                elif close_p < vwap_val:
                    patterns.append("VWAP下回り")
                    score += 1

                if (c2['Close'] < c2['Open']) and (c1['Close'] < c1['Open']) and (close_p < open_p):
                    patterns.append("3本連続陰線")
                    score += 2

                if score == 0:
                    # デフォルトで全銘柄を表示対象にするためスコア0もVWAP位置等で軽微評価
                    if close_p < vwap_val:
                        patterns.append("VWAP下回り")
                        score = 1
                    else:
                        patterns.append("監視対象")
                        score = 1

                if score >= 4:
                    judgment = "🚨 絶好の空売り好機"
                elif score >= 2:
                    judgment = "⚠️ 空売り検討（監視）"
                else:
                    judgment = "🔹 VWAP下回り"

                results.append({
                    "code": code,
                    "name": name,
                    "price": round(close_p, 1),
                    "investment_required": round(close_p * 100),
                    "vwap": round(vwap_val, 1),
                    "score": score,
                    "patterns": " / ".join(patterns),
                    "judgment": judgment,
                    "stop_loss": round(close_p * 1.012, 1),
                    "take_profit": round(close_p * 0.975, 1)
                })

            except Exception:
                continue

    # スコア順にソートして上位100銘柄を保存
    results = sorted(results, key=lambda x: x['score'], reverse=True)
    if results:
        CACHED_RESULTS = results[:100]
    IS_SCANNING = False
    print(f"✅ スキャン完了: {len(CACHED_RESULTS)}銘柄保存")

@app.on_event("startup")
def startup_event():
    # 起動時にバックグラウンドでスキャン開始
    threading.Thread(target=run_scan, daemon=True).start()

@app.get("/api/signals")
def get_signals():
    # まだ1件も読み込めていない場合は急ぎで1回実行
    if not CACHED_RESULTS and not IS_SCANNING:
        threading.Thread(target=run_scan, daemon=True).start()
    return JSONResponse(content=CACHED_RESULTS)

@app.get("/api/chart/{code}")
def get_chart_data(code: str):
    try:
        formatted_ticker = f"{code}.T"
        stock = yf.Ticker(formatted_ticker)
        df_5m = stock.history(period="5d", interval="5m")
        if df_5m.empty:
            return JSONResponse(content={"candles": [], "vwap": []})

        df_5m['TP'] = (df_5m['High'] + df_5m['Low'] + df_5m['Close']) / 3
        df_5m['PV'] = df_5m['TP'] * df_5m['Volume']
        df_5m['Date'] = df_5m.index.date
        cum_pv = df_5m.groupby('Date')['PV'].cumsum()
        cum_vol = df_5m.groupby('Date')['Volume'].cumsum()
        df_5m['VWAP'] = cum_pv / (cum_vol + 1e-5)

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
    except Exception:
        return JSONResponse(content={"candles": [], "vwap": []})

@app.get("/")
@app.head("/")
def get_index():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>index.html が見つかりません</h1>", status_code=404)
