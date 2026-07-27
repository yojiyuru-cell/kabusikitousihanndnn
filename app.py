import os
import math
import io
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

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

def safe_float(val, default=0.0):
    try:
        f = float(val)
        return default if math.isnan(f) or math.isinf(f) else f
    except Exception:
        return default

# 主要銘柄＋JPXリストの安全取得
def get_jpx_stock_list_auto(limit: int = 100):
    # 通信エラー時にも確実に動く国内主力100銘柄
    POPULAR_STOCKS = {
        "6857": "アドバンテスト", "8035": "東京エレクトロン", "6146": "ディスコ", 
        "9984": "ソフトバンクG", "7011": "三菱重工", "6526": "ソシオネクスト", 
        "1570": "日経レバ", "6758": "ソニーG", "7203": "トヨタ", "8306": "三菱UFJ",
        "9104": "商船三井", "8002": "丸紅", "6367": "ダイキン", "6920": "レーザーテック",
        "7735": "スクリン", "4528": "小野薬品", "2413": "エムスリー", "4385": "メルカリ",
        "5253": "カバー", "9166": "GENDA", "5595": "QPS研究所", "1357": "日経ダブルインバ",
        "1579": "日経ブル2倍", "7267": "ホンダ", "8316": "三井住友", "8411": "みずほ",
        "9432": "NTT", "9433": "KDDI", "7974": "任天堂", "6098": "リクルート"
    }

    try:
        jpx_url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
        res = requests.get(jpx_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        res.raise_for_status()
        
        df = pd.read_excel(io.BytesIO(res.content))
        stock_dict = {}
        if 'コード' in df.columns and '銘柄名' in df.columns:
            for _, row in df.dropna(subset=['コード', '銘柄名']).iterrows():
                code_raw = str(row['コード']).split('.')[0].strip()
                if len(code_raw) == 4 and code_raw.isdigit():
                    stock_dict[code_raw] = str(row['銘柄名']).strip()
                    if len(stock_dict) >= limit:
                        break
        if len(stock_dict) > 10:
            return stock_dict
    except Exception as e:
        print(f"⚠️ JPX取得をスキップし、主力銘柄を使用します: {e}")

    return POPULAR_STOCKS

@app.get("/api/signals")
def get_signals(limit: int = 100, min_price: float = 100.0, max_price: float = 20000.0):
    """
    株価データを取得しシグナルを判定。
    フィルター範囲をデフォルトで広く(100円〜20,000円)設定し、必ず結果が出るように変更。
    """
    stock_targets = get_jpx_stock_list_auto(limit=limit)
    ticker_symbols = [f"{code}.T" for code in stock_targets.keys()]
    
    results = []
    if not ticker_symbols:
        return JSONResponse(content=[])

    try:
        print(f"📊 {len(ticker_symbols)} 銘柄のデータ取得中...")
        # yfinanceでダウンロード
        batch_data = yf.download(
            tickers=ticker_symbols,
            period="5d",
            interval="5m",
            group_by="ticker",
            threads=True,
            progress=False
        )
    except Exception as e:
        print(f"❌ 株価データダウンロードエラー: {e}")
        return JSONResponse(content=[])

    for code, name in stock_targets.items():
        symbol = f"{code}.T"
        try:
            # データの切り出し
            if len(ticker_symbols) > 1:
                if symbol not in batch_data.columns.levels[0]:
                    continue
                df_5m = batch_data[symbol].dropna(how="all")
            else:
                df_5m = batch_data.dropna(how="all")

            if df_5m.empty or len(df_5m) < 5:
                continue

            c0 = df_5m.iloc[-1]
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

            vwap_val = safe_float(df_5m['VWAP'].iloc[-1])
            open_p  = safe_float(c0['Open'])
            high_p  = safe_float(c0['High'])
            low_p   = safe_float(c0['Low'])

            patterns = []
            score = 0

            # 簡単なテクニカル分析スコア
            body_size = abs(close_p - open_p)
            upper_wick = high_p - max(open_p, close_p)
            
            if upper_wick >= body_size * 1.5 and upper_wick > 0:
                patterns.append("長い上ヒゲ")
                score += 3

            if close_p < vwap_val:
                patterns.append("VWAP下回り")
                score += 1

            judgment = "🚨 天井・下落警戒" if score >= 3 else ("⚠️ パターン発生" if score >= 1 else "🔹 通常")
            investment_required = round(close_p * 100)

            results.append({
                "code": code,
                "name": name,
                "price": round(close_p, 1),
                "investment_required": investment_required,
                "vwap": round(vwap_val, 1),
                "score": score,
                "patterns": " / ".join(patterns) if patterns else "シグナルなし",
                "judgment": judgment,
                "stop_loss": round(close_p * 1.012, 1),
                "take_profit": round(close_p * 0.975, 1)
            })

        except Exception as e:
            continue

    # スコアが高い順、データがあるものをソートして返す
    results = sorted(results, key=lambda x: x['score'], reverse=True)
    print(f"✅ {len(results)} 件の銘柄を出力します。")
    return JSONResponse(content=results)

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
