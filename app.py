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

# JPX400主要構成銘柄リスト
JPX400_STOCKS = {
    # 自動車・輸送機器
    "7203": "トヨタ自動車", "7267": "ホンダ", "7201": "日産自動車", "7270": "SUBARU",
    "7202": "いすゞ自動車", "7269": "スズキ", "7259": "アイシン", "7205": "日野自動車",
    
    # 銀行・金融・証券・保険
    "8306": "三菱UFJ", "8316": "三井住友FG", "8411": "みずほFG", "8308": "りそなHD",
    "8309": "三井住友トラスト", "8604": "野村HD", "8601": "大和証券G", "8591": "オリックス",
    "8766": "東京海上", "8630": "SOMPO", "8725": "MS&AD", "7182": "ゆうちょ銀行",
    
    # 商社・卸売
    "8058": "三菱商事", "8001": "伊藤忠", "8031": "三井物産", "8002": "丸紅",
    "8015": "豊田通商", "8053": "住友商事", "8037": "カメイ",
    
    # 電気機器・半導体・電子部品
    "6758": "ソニーG", "6501": "日立製作所", "6503": "三菱電機", "6702": "富士通",
    "6752": "パナソニックHD", "6981": "村田製作所", "6762": "TDK", "6902": "デンソー",
    "6954": "ファナック", "6857": "アドバンテスト", "8035": "東京エレクトロン",
    "6146": "ディスコ", "6526": "ソシオネクスト", "6367": "ダイキン", "6920": "レーザーテック",
    "7735": "スクリン", "7751": "キヤノン", "7731": "ニコン", "6273": "SMC",
    "6701": "NEC", "6861": "キーエンス", "6971": "京セラ",
    
    # 機械・重工業・プラント
    "7011": "三菱重工", "7012": "川崎重工", "7013": "IHI", "6301": "小松製作所",
    "6302": "住友重機械", "6326": "クボタ", "6305": "日立建機",
    
    # 情報通信・IT・サービス
    "9432": "NTT", "9434": "ソフトバンク", "9433": "KDDI", "9984": "ソフトバンクG",
    "4755": "楽天グループ", "4689": "LINEヤフー", "4751": "サイバーエージェント",
    "4307": "野村総合研究所", "6098": "リクルートHD", "2413": "エムスリー", "4385": "メルカリ",
    "9735": "セコム", "9719": "SCSK", "3626": "TIS",
    
    # ゲーム・エンタメ・メディア
    "7974": "任天堂", "9684": "スクウェア・エニックス", "3659": "ネクソン",
    
    # 医薬品・化学・素材
    "4063": "信越化学", "4502": "武田薬品", "4519": "中外製薬", "4568": "第一三共",
    "4503": "アステラス製薬", "4523": "エーザイ", "4528": "小野薬品", "4901": "富士フイルム",
    "4911": "資生堂", "3407": "旭化成", "3405": "クラレ", "4188": "三菱ケミカル",
    
    # 運輸・物流・インフラ
    "9104": "商船三井", "9101": "日本郵船", "9107": "川崎汽船",
    "9020": "JR東日本", "9021": "JR西日本", "9022": "JR東海", "9201": "日本航空", "9202": "ANA",
    "9501": "東京電力HD", "9502": "中部電力", "9503": "関西電力",
    
    # 鉄鋼・金属・エネルギー
    "5401": "日本製鉄", "5411": "JFE", "5713": "住友金属鉱山", "1605": "INPEX", "5020": "ENEOS",
    
    # 不動産・建設
    "8801": "三井不動産", "8802": "三菱地所", "8830": "住友不動産", "1925": "大和ハウス", "1928": "積水ハウス",
    
    # 食品・消費財・小売
    "2802": "味の素", "2503": "キリンHD", "2502": "アサヒグループ", "3382": "セブン&アイ",
    "3092": "ZOZO", "8267": "イオン", "2897": "日清食品HD"
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
    """
    1時間足（75分足の代替）をベースに：
    1. 移動平均線（SMA20）が下向き（下降トレンド）
    2. 下落後 → 一時反発（戻り） → 再度下落を開始（戻り売り局面）の銘柄を検出
    """
    if df is None or df.empty or len(df) < 25:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)

    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        return None

    df = df.loc[~df.index.duplicated(keep='first')].sort_index()

    # 移動平均線の算出
    df['SMA5'] = df['Close'].rolling(window=5).mean()
    df['SMA20'] = df['Close'].rolling(window=20).mean()

    # VWAP計算
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    pv = tp * df['Volume']
    cum_pv = pv.cumsum()
    cum_vol = df['Volume'].cumsum()
    vwap_series = cum_pv / (cum_vol + 1e-5)

    c0 = df.iloc[-1]   # 最新の足
    c1 = df.iloc[-2]   # 1本前の足
    c2 = df.iloc[-3]   # 2本前の足
    c3 = df.iloc[-4]   # 3本前の足
    c4 = df.iloc[-5]   # 4本前の足

    close_p  = safe_float(c0['Close'])
    open_p   = safe_float(c0['Open'])
    high_p   = safe_float(c0['High'])
    low_p    = safe_float(c0['Low'])
    vwap_val = safe_float(vwap_series.iloc[-1])

    sma20_now  = safe_float(c0['SMA20'])
    sma20_prev = safe_float(c5['SMA20'] if len(df) >= 6 else c0['SMA20'])

    # 価格制限（10,000円以下）
    if close_p <= 0 or close_p > 10000:
        return None

    patterns = []
    score = 0

    # -------------------------------------------------------------
    # 条件1：上位足（1時間足/75分足相当）が下向きトレンドか
    # -------------------------------------------------------------
    is_downtrend = False
    if sma20_now < sma20_prev or close_p < sma20_now:
        is_downtrend = True
        patterns.append("60-75分足が下向きトレンド")
        score += 2

    # 下降トレンドでなければ対象外
    if not is_downtrend:
        return None

    # -------------------------------------------------------------
    # 条件2：「下落 → 反発上昇 → 再下落」の波形（戻り売りパターン）を判定
    # -------------------------------------------------------------
    # 直近3〜5本前の安値・高値の推移をチェック
    c_recent_max_high = max(safe_float(c1['High']), safe_float(c2['High']), safe_float(c3['High']))
    c_past_min_low    = min(safe_float(c3['Low']), safe_float(c4['Low']))

    # 一時的に上昇（リバウンド）した履歴があるか
    had_rebound = (c_recent_max_high > c_past_min_low * 1.005)

    # 現在の足が反落（陰線、または直近高値から押されている）か
    is_turning_down = (close_p < open_p) or (close_p < safe_float(c1['Close']))

    if had_rebound and is_turning_down:
        patterns.append("戻り完了からの再下落シグナル")
        score += 3

    # 移動平均線（SMA20）付近まで戻してからの反落（グランビルの戻り売り）
    if abs(c_recent_max_high - sma20_now) / sma20_now < 0.01 and close_p < open_p:
        patterns.append("移動平均線で頭打ち・戻り売り傾向")
        score += 2

    # 上げ止まりの長い上ヒゲが出ているか
    body_size = abs(close_p - open_p)
    upper_wick = high_p - max(open_p, close_p)
    if upper_wick >= max(body_size * 1.2, 2.0):
        patterns.append("戻り高値での上値抵抗（上ヒゲ）")
        score += 1

    # パターンが検知できなかった場合は除外
    if len(patterns) <= 1:
        return None

    # 判定メッセージの設定
    if score >= 5:
        judgment = "🎯 空売りチャンスかも？（戻り売り高精度）"
    elif score >= 3:
        judgment = "👀 戻り目からの再下落形成中"
    else:
        judgment = "🔹 反落兆候あり"

    return {
        "code": code,
        "name": name,
        "price": round(close_p, 1),
        "investment_required": round(close_p * 100),
        "vwap": round(vwap_val, 1),
        "score": score,
        "patterns": " / ".join(patterns),
        "judgment": judgment,
        "stop_loss": round(max(c_recent_max_high, close_p * 1.015), 1),
        "take_profit": round(close_p * 0.96, 1)
    }

def run_scan():
    global CACHED_RESULTS, IS_SCANNING
    if IS_SCANNING:
        return
    
    IS_SCANNING = True
    print("🔄 JPX400銘柄スキャン開始（60-75分足 下向・戻り売り判定）...")
    
    targets = JPX400_STOCKS
    results = []

    codes = list(targets.keys())
    chunk_size = 15
    
    for i in range(0, len(codes), chunk_size):
        chunk_codes = codes[i:i + chunk_size]
        tickers = [f"{c}.T" for c in chunk_codes]

        try:
            # 75分足の代わりに「60分足（1h）」を1ヶ月分取得
            data = yf.download(tickers, period="1mo", interval="60m", group_by="ticker", threads=True, progress=False)
            
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

        if results:
            CACHED_RESULTS = sorted(results, key=lambda x: x['score'], reverse=True)
            
        time.sleep(0.5)

    IS_SCANNING = False
    print(f"✅ スキャン完了: Total {len(CACHED_RESULTS)} 銘柄（戻り売り適合済み）")

@app.on_event("startup")
def startup_event():
    threading.Thread(target=run_scan, daemon=True).start()

@app.get("/api/signals")
def get_signals():
    if not CACHED_RESULTS and not IS_SCANNING:
        threading.Thread(target=run_scan, daemon=True).start()
    return JSONResponse(content=CACHED_RESULTS)

@app.get("/api/chart/{code}")
def get_chart_data(code: str):
    try:
        ticker = yf.Ticker(f"{code}.T")
        df = ticker.history(period="1mo", interval="60m")
        if df.empty:
            df = ticker.history(period="1mo", interval="1d")

        if df.empty:
            return JSONResponse(content={"candles": [], "vwap": []})

        df = df.loc[~df.index.duplicated(keep='first')].sort_index()

        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC').tz_convert('Asia/Tokyo')
        else:
            df.index = df.index.tz_convert('Asia/Tokyo')

        tp = (df['High'] + df['Low'] + df['Close']) / 3
        pv = tp * df['Volume']
        cum_pv = pv.cumsum()
        cum_vol = df['Volume'].cumsum()
        df['VWAP'] = cum_pv / (cum_vol + 1e-5)

        candles, vwap_list = [], []
        for idx, row in df.tail(80).iterrows():
            ts = int(idx.timestamp())
            
            open_p = safe_float(row['Open'])
            high_p = safe_float(row['High'])
            low_p = safe_float(row['Low'])
            close_p = safe_float(row['Close'])
            vwap_p = safe_float(row['VWAP'])

            if open_p > 0 and close_p > 0:
                candles.append({
                    "time": ts,
                    "open": round(open_p, 1),
                    "high": round(high_p, 1),
                    "low": round(low_p, 1),
                    "close": round(close_p, 1)
                })
                vwap_list.append({
                    "time": ts,
                    "value": round(vwap_p, 1)
                })

        return JSONResponse(content={"candles": candles, "vwap": vwap_list})
    except Exception as e:
        print(f"Chart error: {e}")
        return JSONResponse(content={"candles": [], "vwap": []})

@app.get("/")
@app.head("/")
def get_index():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>index.html が見つかりません</h1>", status_code=404)
