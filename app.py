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

# デイトレ・スイングで人気のある主要・中大型銘柄のコードリスト
POPULAR_CODES = [
    "6857", "8035", "6146", "9984", "7011", "6526", "1570", "6758", "7203", "8306",
    "9104", "8002", "6367", "6920", "7735", "4528", "2413", "4385", "5253", "9166",
    "5595", "1357", "1579", "7267", "8316", "8411", "9432", "9433", "7974", "6098",
    "4063", "6902", "6501", "7751", "6702", "7270", "4502", "4519", "4568", "6861"
]

def get_target_stock_list(limit: int = 300):
    url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
    stock_dict = {}
    try:
        df = pd.read_excel(url)
        df_stocks = df[['コード', '銘柄名', '市場・商品区分']].dropna()
        for _, row in df_stocks.iterrows():
            code = str(int(row['コード']))
            name = str(row['銘柄名'])
            market = str(row['市場・商品区分'])
            if any(m in market for m in ['プライム', 'スタンダード', 'グロース']):
                stock_dict[code] = name
                if len(stock_dict) >= limit:
                    break
    except Exception:
        stock_dict = {code: f"銘柄 {code}" for code in POPULAR_CODES}
    
    return stock_dict

@app.get("/api/signals")
def get_signals(limit: int = 300, min_price: float = 1500.0, max_price: float = 8000.0):
    """
    min_price: 15万円 (100株 = 1,500円)
    max_price: 80万円 (100株 = 8,000円)
    """
    stock_targets = get_target_stock_list(limit=limit)
    ticker_symbols = [f"{code}.T" for code in stock_targets.keys()]
    
    results = []

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
        print(f"データダウンロードエラー: {e}")
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

            # 投資資金フィルター（100株あたり15万〜80万円 ＝ 株価 1,500円〜8,000円）
            if not (min_price <= close_p <= max_price):
                continue

            # VWAPの計算
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

            # チャートパターン判定
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
                
                investment_required = round(close_p * 100)  # 100株購入に必要な資金

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
