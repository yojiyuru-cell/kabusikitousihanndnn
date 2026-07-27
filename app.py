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
            df_5m = fetch_jpx_data(code, interval="5m", period="5d")
            if df_5m.empty or len(df_5m) < 15:
                continue

            # VWAPの計算
            df_5m['TP'] = (df_5m['High'] + df_5m['Low'] + df_5m['Close']) / 3
            df_5m['PV'] = df_5m['TP'] * df_5m['Volume']
            df_5m['Date'] = df_5m.index.date
            cum_pv = df_5m.groupby('Date')['PV'].cumsum()
            cum_vol = df_5m.groupby('Date')['Volume'].cumsum()
            df_5m['VWAP'] = cum_pv / cum_vol
            
            # 本日のデータのみ抽出
            latest_date = df_5m['Date'].iloc[-1]
            df_today = df_5m[df_5m['Date'] == latest_date]
            if len(df_today) < 3:
                continue

            # 1. 本日の騰落率（前日終値または始値比での急騰判定）
            open_today = df_today['Open'].iloc[0]
            curr_candle = df_today.iloc[-1]
            prev_candle = df_today.iloc[-2]

            close_p = curr_candle['Close']
            open_p = curr_candle['Open']
            high_p = curr_candle['High']
            low_p = curr_candle['Low']
            vol_p = curr_candle['Volume']
            vwap_val = safe_float(curr_candle['VWAP'])

            prev_close_p = prev_candle['Close']
            prev_vwap_val = safe_float(prev_candle['VWAP'])
            avg_vol = df_today['Volume'].tail(10).mean()

            # 条件A: 本日の高値圏・急騰（始値比+2.5%以上、または高値からの押し）
            day_gain = ((high_p - open_today) / open_today) * 100
            is_surging = day_gain >= 2.5

            # 条件B: VWAPクロス（前回はVWAP上 ➔ 今回はVWAP下割れ）
            is_vwap_break = (prev_close_p >= prev_vwap_val) and (close_p < vwap_val)

            # 条件C: 戻り売りパターン（すでにVWAP下で、高値でVWAPにタッチして反発失敗）
            is_vwap_retest = (close_p < vwap_val) and (high_p >= vwap_val * 0.998) and (close_p < open_p)

            # 条件D: 出来高伴う（急騰・急落時の出来高増加）
            is_volume_spike = vol_p > (avg_vol * 1.3)

            # 条件E: チャート形状（上ヒゲピンバー）
            body_size = abs(close_p - open_p)
            upper_wick = high_p - max(open_p, close_p)
            is_upper_wick = (upper_wick >= body_size * 1.8) and (upper_wick > 0)

            # --- 勝ち筋シグナル判定 ---
            judgment = "⚪️ 監視対象外"
            priority = 0

            if is_surging and is_vwap_break and is_volume_spike:
                judgment = "🔥 超強力：急騰後VWAPブレイク（成り行き売り）"
                priority = 3
            elif is_surging and is_vwap_retest and is_upper_wick:
                judgment = "🚨 高勝率：VWAP戻り売り失敗（上ヒゲピンバー）"
                priority = 3
            elif is_surging and (close_p < vwap_val):
                judgment = "⚡️ 空売りチャンス（急騰後のVWAP割り込み）"
                priority = 2
            elif is_surging and (close_p >= vwap_val):
                judgment = "🟡 急騰高値圏（VWAP割れ待ち）"
                priority = 1

            if priority > 0:
                # 損切り：エントリー価格 +1.2%（またはVWAP上抜け）
                # 利確：エントリー価格 -2.5%
                results.append({
                    "code": code,
                    "name": name,
                    "price": round(safe_float(close_p), 1),
                    "vwap": round(vwap_val, 1),
                    "gain": round(day_gain, 1),
                    "judgment": judgment,
                    "priority": priority,
                    "stop_loss": round(safe_float(close_p) * 1.012, 1),
                    "take_profit": round(safe_float(close_p) * 0.975, 1)
                })

        except Exception as e:
            print(f"銘柄スキップ {code}: {e}")
            continue

    # シグナルの優先度（高い順）にソートして返却
    results = sorted(results, key=lambda x: x['priority'], reverse=True)
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
