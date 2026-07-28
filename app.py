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

# ---------------------------------------------------------
# メモリ内キャッシュ (最新のTOP100結果を保持)
# ---------------------------------------------------------
CACHED_RESULTS = []
IS_SCANNING = False
LAST_UPDATED_TIME = ""

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
    """ETF・ETN・投信・インデックス商品かどうかを判定して除外する"""
    etf_keywords = ["ETF", "ETN", "上場", "ファンド", "ブル", "ベア", "インデックス", "TOPIX", "日経225", "S&P", "NASDAQ", "225"]
    for kw in etf_keywords:
        if kw.lower() in name.lower():
            return True

    if re.match(r"^(13|14|15|16|20|25|26|28)\d{2}$", code):
        return True

    return False

def get_all_jpx_stock_list():
    """JPXから全東証上場銘柄リストを取得（個別株のみ全件抽出）"""
    POPULAR_STOCKS = {
        "6857": "アドバンテスト", "8035": "東京エレクトロン", "6146": "ディスコ", 
        "9984": "ソフトバンクG", "7011": "三菱重工", "6526": "ソシオネクスト", 
        "6758": "ソニーG", "7203": "トヨタ", "8306": "三菱UFJ", "9104": "商船三井"
    }

    try:
        jpx_url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
        res = requests.get(jpx_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
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
        print(f"⚠️ JPXリスト取得エラー: {e}")

    return POPULAR_STOCKS

# ---------------------------------------------------------
# バックグラウンド全銘柄スキャン関数
# ---------------------------------------------------------
def scan_all_stocks_background():
    global CACHED_RESULTS, IS_SCANNING, LAST_UPDATED_TIME
    if IS_SCANNING:
        return
    
    IS_SCANNING = True
    print("🔄 バックグラウンドで全銘柄スキャンを開始します...")
    
    stock_targets = get_all_jpx_stock_list()
    codes = list(stock_targets.keys())
    results = []

    # 300銘柄ごとのバッチ処理
    batch_size = 300
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

                # 株価フィルター（1,500円〜10,000円）
                if not (1500.0 <= close_p <= 10000.0):
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

                # --- 空売りシグナル判定 ---
                body_size = abs(close_p - open_p)
                upper_wick = high_p - max(open_p, close_p)
                if upper_wick >= max(body_size * 1.5, 3.0):
                    patterns.append("長い上ヒゲ（天井打）")
                    score += 3

                if (c1['Close'] > c1['Open']) and (close_p < open_p) and (open_p >= c1['Close']) and (close_p <= c1['Open']):
                    patterns.append("陰線包み足（強反落）")
                    score += 3

                if len(df_5m) >= 15:
                    recent_15 = df_5m.tail(15)
                    highs = recent_15['High'].nlargest(2).values
                    if len(highs) == 2 and abs(highs[0] - highs[1]) / (highs[0] + 1e-5) < 0.005:
                        recent_low_mean = recent_15['Low'].mean()
                        if close_p < recent_low_mean:
                            patterns.append("ダブルトップ崩れ")
                            score += 4

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
                    continue

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
                    "patterns": " / ".join(patterns) if patterns else "VWAP下回り",
                    "judgment": judgment,
                    "stop_loss": round(close_p * 1.012, 1),
                    "take_profit": round(close_p * 0.975, 1)
                })

            except Exception:
                continue

    # スコア順にソートして上位100銘柄をキャッシュ保存
    results = sorted(results, key=lambda x: x['score'], reverse=True)
    CACHED_RESULTS = results[:100]
    IS_SCANNING = False
    print(f"✅ スキャン完了！ TOP{len(CACHED_RESULTS)} 銘柄をキャッシュしました。")

# ---------------------------------------------------------
# 定期更新タイマー（バックグラウンドループ）
# ---------------------------------------------------------
def background_loop():
    while True:
        try:
            scan_all_stocks_background()
        except Exception as e:
            print(f"❌ ループエラー: {e}")
        time.sleep(300) # 5分ごとに自動再スキャン

# サーバー起動時にスキャンを開始
threading.Thread(target=background_loop, daemon=True).start()

# ---------------------------------------------------------
# APIエンドポイント
# ---------------------------------------------------------
@app.get("/api/signals")
def get_signals():
    # 初回スキャン中かつキャッシュが無い場合、手動スキャンを実行
    if not CACHED_RESULTS and not IS_SCANNING:
        threading.Thread(target=scan_all_stocks_background, daemon=True).start()
        
    return JSONResponse(content=CACHED_RESULTS)

@app.get("/api/chart/{code}")
def get_chart_data(code: str):
    try:
        formatted_ticker = f"{code}.T"
        stock = yf.Ticker(formatted_ticker)
        df_5m = stock.history(period="5d", interval="5m")
        if df_5m.empty:
            return JSONResponse(content={"candles": [], "vwap": [], "earnings_date": "未定", "ex_dividend_date": "なし/未定"})

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

        return JSONResponse(content={
            "candles": candles,
            "vwap": vwap_list,
            "earnings_date": "未定",
            "ex_dividend_date": "なし/未定"
        })
    except Exception:
        return JSONResponse(content={"candles": [], "vwap": [], "earnings_date": "未定", "ex_dividend_date": "なし/未定"})

@app.get("/")
@app.head("/")
def get_index():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>index.html が見つかりません</h1>", status_code=404)

