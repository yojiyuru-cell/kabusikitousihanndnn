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

CACHED_RESULTS = []
IS_SCANNING = False

# 初期表示用・主要銘柄リスト
POPULAR_STOCKS = {
    "6857": "アドバンテスト", "8035": "東京エレクトロン", "6146": "ディスコ", 
    "9984": "ソフトバンクG", "7011": "三菱重工", "6526": "ソシオネクスト", 
    "6758": "ソニーG", "7203": "トヨタ", "8306": "三菱UFJ", "9104": "商船三井",
    "8002": "丸紅", "6367": "ダイキン", "6920": "レーザーテック", "7735": "スクリン",
    "4528": "小野薬品", "2413": "エムスリー", "4385": "メルカリ", "5253": "カバー",
    "9166": "GENDA", "5595": "QPS研究所", "7267": "ホンダ", "8316": "三井住友"
}

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

def safe_float(val, default=0.0):
    try:
        f = float(val)
        return default if math.isnan(f) or math.isinf(f) else f
    except Exception:
        return default

def process_df(df, code, name):
    """データフレームからシグナルとスコアを抽出"""
    if df is None or df.empty or len(df) < 3:
        return None

    # 列名の整理（MultiIndex対策）
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)

    # 必須カラムチェック
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        return None

    # 直近3本のデータ
    c0 = df.iloc[-1]
    c1 = df.iloc[-2]
    c2 = df.iloc[-3]

    close_p = safe_float(c0['Close'])
    
    # 【変更箇所】1株あたり1万円を超える銘柄、および0円以下を除外
    if close_p <= 0 or close_p > 10000:
        return None

    # VWAP計算
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    pv = tp * df['Volume']
    cum_pv = pv.cumsum()
    cum_vol = df['Volume'].cumsum()
    vwap_series = cum_pv / (cum_vol + 1e-5)

    open_p   = safe_float(c0['Open'])
    high_p   = safe_float(c0['High'])
    low_p    = safe_float(c0['Low'])
    vwap_val = safe_float(vwap_series.iloc[-1])
    prev_vwap = safe_float(vwap_series.iloc[-2])

    patterns = []
    score = 0

    # シグナル判定
    body_size = abs(close_p - open_p)
    upper_wick = high_p - max(open_p, close_p)
    if upper_wick >= max(body_size * 1.5, 3.0):
        patterns.append("長い上ヒゲ（天井打）")
        score += 3

    c1_close = safe_float(c1['Close'])
    c1_open = safe_float(c1['Open'])
    if (c1_close > c1_open) and (close_p < open_p) and (open_p >= c1_close) and (close_p <= c1_open):
        patterns.append("陰線包み足（強反落）")
        score += 3

    if (c1_close >= prev_vwap) and (close_p < vwap_val):
        patterns.append("VWAP下抜け（デッドクロス）")
        score += 3
    elif close_p < vwap_val:
        patterns.append("VWAP下回り")
        score += 1

    c2_close = safe_float(c2['Close'])
    c2_open = safe_float(c2['Open'])
    if (c2_close < c2_open) and (c1_close < c1_open) and (close_p < open_p):
        patterns.append("3本連続陰線")
        score += 2

    if score == 0:
        patterns.append("VWAP下回り" if close_p < vwap_val else "監視対象")
        score = 1

    judgment = "🚨 絶好の空売り好機" if score >= 4 else ("⚠️ 空売り検討（監視）" if score >= 2 else "🔹 VWAP下回り")

    return {
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
    }

def run_scan():
    global CACHED_RESULTS, IS_SCANNING
    if IS_SCANNING:
        return
    
    IS_SCANNING = True
    print("🔄 スキャン開始...")
    
    targets = POPULAR_STOCKS
    results = []

    # 10銘柄ずつ安全に分散取得
    codes = list(targets.keys())
    chunk_size = 10
    
    for i in range(0, len(codes), chunk_size):
        chunk_codes = codes[i:i + chunk_size]
        tickers = [f"{c}.T" for c in chunk_codes]

        try:
            # 5日分・5分足を取得
            data = yf.download(tickers, period="5d", interval="5m", group_by="ticker", threads=True, progress=False)
            
            for code in chunk_codes:
                symbol = f"{code}.T"
                try:
                    if len(chunk_codes) == 1:
                        df_single = data
                    else:
                        df_single = data[symbol] if symbol in data else None
                    
                    if df_single is not None:
                        item = process_df(df_single.dropna(how="all"), code, targets[code])
                        if item:
                            results.append(item)
                except Exception as ex:
                    continue
        except Exception as e:
            print(f"Error fetching chunk: {e}")

        # 段階的にキャッシュへ反映（画面表示を早めるため）
        if results:
            CACHED_RESULTS = sorted(results, key=lambda x: x['score'], reverse=True)

    IS_SCANNING = False
    print(f"✅ スキャン完了: Total {len(CACHED_RESULTS)} 銘柄")

@app.on_event("startup")
def startup_event():
    # 起動時にバックグラウンドでスキャン実行
    threading.Thread(target=run_scan, daemon=True).start()

@app.get("/api/signals")
def get_signals():
    # データが空の場合は再度スキャンを発火
    if not CACHED_RESULTS and not IS_SCANNING:
        threading.Thread(target=run_scan, daemon=True).start()
    return JSONResponse(content=CACHED_RESULTS)

@app.get("/api/chart/{code}")
def get_chart_data(code: str):
    try:
        ticker = yf.Ticker(f"{code}.T")
        df = ticker.history(period="5d", interval="5m")
        if df.empty:
            df = ticker.history(period="1mo", interval="1d")

        if df.empty:
            return JSONResponse(content={"candles": [], "vwap": []})

        tp = (df['High'] + df['Low'] + df['Close']) / 3
        pv = tp * df['Volume']
        cum_pv = pv.cumsum()
        cum_vol = df['Volume'].cumsum()
        df['VWAP'] = cum_pv / (cum_vol + 1e-5)

        candles, vwap_list = [], []
        for idx, row in df.tail(80).iterrows():
            ts = int(idx.timestamp())
            candles.append({
                "time": ts,
                "open": round(safe_float(row['Open']), 1),
                "high": round(safe_float(row['High']), 1),
                "low": round(safe_float(row['Low']), 1),
                "close": round(safe_float(row['Close']), 1)
            })
            vwap_list.append({
                "time": ts,
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
