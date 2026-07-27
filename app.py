import os
import math
import requests
import io
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

# JPX（日本取引所グループ）から全上場銘柄リストを自動取得する関数
def get_jpx_stock_list_auto(limit: int = 300):
    jpx_url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }
    
    stock_dict = {}
    try:
        print("📥 JPX公式サイトから最新の全銘柄リストを自動取得中...")
        res = requests.get(jpx_url, headers=headers, timeout=10)
        res.raise_for_status()
        
        # Excelファイルの読み込み
        df = pd.read_excel(io.BytesIO(res.content))
        
        # 必要なカラムの抽出 (コード, 銘柄名, 市場・商品区分)
        if 'コード' in df.columns and '銘柄名' in df.columns:
            # 内国株式（プライム、スタンダード、グロース）のみ抽出
            df_stocks = df.dropna(subset=['コード', '銘柄名'])
            
            for _, row in df_stocks.iterrows():
                try:
                    code_raw = str(row['コード']).split('.')[0].strip()
                    # 4桁の数値コード（株式）のみ対象
                    if len(code_raw) == 4 and code_raw.isdigit():
                        market = str(row.get('市場・商品区分', ''))
                        if any(m in market for m in ['プライム', 'スタンダード', 'グロース', '市場']):
                            stock_dict[code_raw] = str(row['銘柄名']).strip()
                            if len(stock_dict) >= limit:
                                break
                except Exception:
                    continue
        print(f"✅ JPXから {len(stock_dict)} 銘柄を取得しました。")
    except Exception as e:
        print(f"⚠️ JPX自動取得エラー ({e})。予備の主要銘柄リストを使用します。")
        # フォールバック用の主要銘柄
        backup_codes = ["6857", "8035", "6146", "9984", "7011", "6526", "1570", "6758", "7203", "8306", "9104", "8002", "6367", "6920", "7735", "4528", "2413", "4385", "5253", "9166"]
        stock_dict = {code: f"銘柄 {code}" for code in backup_codes}

    return stock_dict

@app.get("/api/signals")
def get_signals(limit: int = 200, min_price: float = 1500.0, max_price: float = 8000.0):
    """
    JPX全銘柄自動取得 ＆ 株価フィルター（15万〜80万円 / 1,500円〜8,000円）
    """
    stock_targets = get_jpx_stock_list_auto(limit=limit)
    ticker_symbols = [f"{code}.T" for code in stock_targets.keys()]
    
    results = []
    if not ticker_symbols:
        return JSONResponse(content=[])

    try:
        print(f"📊 {len(ticker_symbols)} 銘柄の株価データをダウンロード中...")
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
            if len(ticker_symbols) > 1:
                if symbol not in batch_data.columns.levels[0]:
                    continue
                df_5m = batch_data[symbol].dropna(how="all")
            else:
                df_5m = batch_data.dropna(how="all")

            if df_5m.empty or len(df_5m) < 20:
                continue

            c0 = df_5m.iloc[-1]
            close_p = safe_float(c0['Close'])

            # 株価フィルター（100株＝15万円〜80万円 ＝ 株価 1,500円〜8,000円）
            if not (min_price <= close_p <= max_price):
                continue

            # VWAP計算
            df_5m['TP'] = (df_5m['High'] + df_5m['Low'] + df_5m['Close']) / 3
            df_5m['PV'] = df_5m['TP'] * df_5m['Volume']
            df_5m['Date'] = df_5m.index.date
            cum_pv = df_5m.groupby('Date')['PV'].cumsum()
            cum_vol = df_5m.groupby('Date')['Volume'].cumsum()
            df_5m['VWAP'] = cum_pv / cum_vol

            c1 = df_5m.iloc[-2]
            c2 = df_5m.iloc[-3]

            open_p  = safe_float(c0['Open'])
            high_p  = safe_float(c0['High'])
            low_p   = safe_float(c0['Low'])
            vwap_val = safe_float(c0['VWAP'])

            patterns = []
            score = 0

            # テクニカル分析パターン判定
            body_size = abs(close_p - open_p)
            upper_wick = high_p - max(open_p, close_p)
            
            if upper_wick >= body_size * 2 and upper_wick > 0:
                patterns.append("長い上ヒゲ（天井打）")
                score += 3

            if (c1['Close'] > c1['Open']) and (close_p < open_p) and (open_p >= c1['Close']) and (close_p <= c1['Open']):
                patterns.append("陰線包み足（強反落）")
                score += 3

            if (c2['Close'] < c2['Open']) and (c1['Close'] < c1['Open']) and (close_p < open_p):
                patterns.append("3本連続陰線")
                score += 2

            recent_15 = df_5m.tail(15)
            highs = recent_15['High'].nlargest(2).values
            if len(highs) == 2 and abs(highs[0] - highs[1]) / (highs[0] + 1e-5) < 0.003:
                if close_p < recent_15['Low'].mean():
                    patterns.append("ダブルトップ崩れ")
                    score += 4

            is_below_vwap = close_p < vwap_val
            if is_below_vwap:
                score += 1

            if score > 0 or is_below_vwap:
                judgment = "🚨 天井・下落確定" if score >= 4 else ("⚠️ パターン発生" if score >= 2 else "🔹 VWAP割り込み")
                investment_required = round(close_p * 100)

                results.append({
                    "code": code,
                    "name": name,
                    "price": round(close_p, 1),
                    "investment_required": investment_required,
                    "vwap": round(vwap_val, 1),
                    "score": score,
                    "patterns": " / ".join(patterns) if patterns else "VWAP下回り",
                    "judgment": judgment,
                    "stop_loss": round(close_p * 1.012, 1),
                    "take_profit": round(close_p * 0.975, 1)
                })

        except Exception as e:
            continue

    results = sorted(results, key=lambda x: x['score'], reverse=True)
    return JSONResponse(content=results)

@app.get("/api/chart/{code}")
def get_chart_data(code: str):
    try:
        formatted_ticker = f"{code}.T"
        stock = yf.Ticker(formatted_ticker)
        
        earnings_date = "未定"
        ex_dividend_date = "なし/未定"
        try:
            cal = stock.calendar
            if isinstance(cal, dict):
                if 'Earnings Date' in cal and len(cal['Earnings Date']) > 0:
                    earnings_date = pd.to_datetime(cal['Earnings Date'][0]).strftime('%Y-%m-%d')
                if 'Ex-Dividend Date' in cal and cal['Ex-Dividend Date']:
                    ex_dividend_date = pd.to_datetime(cal['Ex-Dividend Date']).strftime('%Y-%m-%d')
            elif isinstance(cal, pd.DataFrame) and not cal.empty:
                if 'Earnings Date' in cal.index:
                    earnings_date = pd.to_datetime(cal.loc['Earnings Date'].iloc[0]).strftime('%Y-%m-%d')
                if 'Ex-Dividend Date' in cal.index:
                    ex_dividend_date = pd.to_datetime(cal.loc['Ex-Dividend Date'].iloc[0]).strftime('%Y-%m-%d')
        except Exception:
            pass

        df_5m = stock.history(period="5d", interval="5m")
        if df_5m.empty:
            return JSONResponse(content={"candles": [], "vwap": [], "earnings_date": earnings_date, "ex_dividend_date": ex_dividend_date})

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

        return JSONResponse(content={
            "candles": candles,
            "vwap": vwap_list,
            "earnings_date": earnings_date,
            "ex_dividend_date": ex_dividend_date
        })
    except Exception as e:
        return JSONResponse(content={"candles": [], "vwap": [], "earnings_date": "未定", "ex_dividend_date": "なし/未定"})

@app.get("/")
def get_index():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>index.html が見つかりません</h1>", status_code=404)
