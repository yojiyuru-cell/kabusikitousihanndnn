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
    1. 75分/60分足が下向きトレンド
    2. 反発後に「再下落」を開始
    3. 🚨【重要】再上昇（反発継続・高値更新・底堅さ）の兆候がある銘柄は厳格に除外
    """
    if df is None or df.empty or len(df) < 10:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs(f"{code}.T", axis=1, level=1)
        except Exception:
            df.columns = df.columns.get_level_values(-1)

    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        return None

    df = df.dropna(subset=required_cols).loc[~df.index.duplicated(keep='first')].sort_index()

    if len(df) < 10:
        return None

    # テクニカル指標の計算
    df['SMA5'] = df['Close'].rolling(window=5, min_periods=1).mean()
    df['SMA20'] = df['Close'].rolling(window=20, min_periods=1).mean()

    tp = (df['High'] + df['Low'] + df['Close']) / 3
    pv = tp * df['Volume']
    cum_pv = pv.cumsum()
    cum_vol = df['Volume'].cumsum()
    vwap_series = cum_pv / (cum_vol + 1e-5)

    c0 = df.iloc[-1]
    c1 = df.iloc[-2] if len(df) >= 2 else c0
    c2 = df.iloc[-3] if len(df) >= 3 else c1
    c3 = df.iloc[-4] if len(df) >= 4 else c2

    close_p  = safe_float(c0['Close'])
    open_p   = safe_float(c0['Open'])
    high_p   = safe_float(c0['High'])
    low_p    = safe_float(c0['Low'])
    vwap_val = safe_float(vwap_series.iloc[-1])

    sma5_now   = safe_float(c0['SMA5'])
    sma5_prev  = safe_float(c1['SMA5'])
    sma20_now  = safe_float(c0['SMA20'])
    sma20_prev = safe_float(df['SMA20'].iloc[-5] if len(df) >= 5 else c0['SMA20'])

    if close_p <= 0 or close_p > 20000:
        return None

    # =============================================================
    # 🛑 排除フィルター：再上昇の兆候がある銘柄はすべて即弾く
    # =============================================================
    recent_high = max(safe_float(c1['High']), safe_float(c2['High']), safe_float(c3['High']))

    # 1. 直近の高値をブレイクして上に抜けている（上昇再開）
    if close_p >= recent_high:
        return None

    # 2. 直近の足が大陽線（買い勢力がかなり強い）
    body_size = close_p - open_p
    if body_size > (close_p * 0.008):  # 0.8%以上の陽線は警戒
        return None

    # 3. 短期MA(SMA5)が力強く上向いている（短期上昇トレンドへの転換）
    if sma5_now > sma5_prev * 1.003 and close_p > sma5_now:
        return None

    # 4. 長い下ヒゲが発生している（下値での強力な買戻し・底堅さ）
    lower_wick = min(open_p, close_p) - low_p
    if lower_wick > abs(body_size) * 1.5 and lower_wick > (close_p * 0.004):
        return None

    # =============================================================
    # 🎯 判定ロジック：条件を満たす「下落トレンド＆戻り売り」のみ抽出
    # =============================================================
    patterns = []
    score = 0

    # 中期トレンドが下向き（SMA20下向、または終値がSMA20の下）
    if sma20_now < sma20_prev or close_p < sma20_now:
        patterns.append("60-75分足が下向きトレンド")
        score += 2
    else:
        return None

    # 戻り目からの「反落（陰線・または高値からの押し戻され）」を確認
    recent_low = min(safe_float(c1['Low']), safe_float(c2['Low']), safe_float(c3['Low']))
    if recent_high > recent_low * 1.002:
        if close_p < open_p or close_p < safe_float(c1['Close']):
            patterns.append("戻り完了からの再下落")
            score += 2

    # 移動平均線（SMA20）付近での頭打ち（戻り売りの急所）
    if abs(recent_high - sma20_now) / (sma20_now + 1e-5) < 0.015:
        patterns.append("移動平均線で頭打ち（戻り売り）")
        score += 1

    # 上ヒゲ（上値の重さ・売り圧力）
    upper_wick = high_p - max(open_p, close_p)
    if upper_wick > abs(body_size) and upper_wick > 0.5:
        patterns.append("上ヒゲ（上値抵抗確認）")
        score += 1

    if score < 3:
        return None

    # 判定メッセージ
    if score >= 5:
        judgment = "🎯 空売りチャンスかも？（反落確定・上昇要素排除済）"
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
        "stop_loss": round(max(recent_high, close_p * 1.012), 1),
        "take_profit": round(close_p * 0.96, 1)
    }

def run_scan():
    global CACHED_RESULTS, IS_SCANNING
    if IS_SCANNING:
        return
    
    IS_SCANNING = True
    print("🔄 スキャン開始（再上昇銘柄を完全排除する戻り売り判定）...")
    
    targets = JPX400_STOCKS
    results = []
    codes = list(targets.keys())

    chunk_size = 10
    for i in range(0, len(codes), chunk_size):
        chunk_codes = codes[i:i + chunk_size]
        tickers = [f"{c}.T" for c in chunk_codes]

        try:
            data = yf.download(tickers, period="1mo", interval="60m", group_by="ticker", threads=True, progress=False)
            
            for code in chunk_codes:
                symbol = f"{code}.T"
                try:
                    df_single = None
                    if len(chunk_codes) == 1:
                        df_single = data
                    elif isinstance(data.columns, pd.MultiIndex) and symbol in data.columns.levels[0]:
                        df_single = data[symbol]

                    if df_single is None or df_single.dropna().empty:
                        ticker_obj = yf.Ticker(symbol)
                        df_single = ticker_obj.history(period="1mo", interval="60m")

                    if df_single is not None and not df_single.empty:
                        item = process_df(df_single, code, targets[code])
                        if item:
                            results.append(item)
                except Exception as ex:
                    continue
        except Exception as e:
            print(f"Chunk error: {e}")

        if results:
            CACHED_RESULTS = sorted(results, key=lambda x: x['score'], reverse=True)
            
        time.sleep(0.3)

    IS_SCANNING = False
    print(f"✅ スキャン完了: 厳選 {len(CACHED_RESULTS)} 銘柄抽出（再上昇懸念株を除外）")

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
