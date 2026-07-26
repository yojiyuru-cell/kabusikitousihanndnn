mport datetime
import yfinance as yf
import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse,HTMLResponse

# ---------------------------------------------------------
# 1. データ取得・テクニカル分析ロジック
# ---------------------------------------------------------
def fetch_jpx_data(ticker_symbol: str, interval: str = "1d", period: str = "3mo") -> pd.DataFrame:
   """東証銘柄（例: '6857' -> '6857.T'）の株価データを取得"""
   formatted_ticker = f"{ticker_symbol}.T" if not ticker_symbol.endswith(".T") else ticker_symbol
   try:
       stock = yf.Ticker(formatted_ticker)
       df = stock.history(period=period, interval=interval)
       return df if not df.empty else pd.DataFrame()
   except Exception:
       return pd.DataFrame()

def check_daily_overbought(df_daily: pd.DataFrame, dev_threshold: float = 15.0, rsi_threshold: float = 70.0):
   """日足の過熱感（25日乖離率・RSI）を判定"""
   if df_daily.empty or len(df_daily) < 25:
       return False, 0.0, 0.0

   df = df_daily.copy()
   df['SMA25'] = df['Close'].rolling(window=25).mean()
   latest_close = df['Close'].iloc[-1]
   latest_sma25 = df['SMA25'].iloc[-1]
   dev_rate = ((latest_close - latest_sma25) / latest_sma25) * 100

   delta = df['Close'].diff()
   gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
   loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
   rs = gain / loss
   rsi = 100 - (100 / (1 + rs))
   latest_rsi = rsi.iloc[-1]

   is_overbought = (dev_rate >= dev_threshold) and (latest_rsi >= rsi_threshold)
   return is_overbought, round(dev_rate, 2), round(latest_rsi, 2)

def detect_chart_patterns(df_5m: pd.DataFrame) -> list:
   """5分足のローソク足形状パターン判定"""
   patterns = []
   if df_5m.empty or len(df_5m) < 5:
       return patterns

   latest = df_5m.iloc[-1]
   prev = df_5m.iloc[-2]

   # 上ヒゲ大陰線判定
   body = abs(latest['Close'] - latest['Open'])
   upper_shade = latest['High'] - max(latest['Open'], latest['Close'])
   if upper_shade > (body * 2) and latest['Close'] < latest['Open']:
       patterns.append("上ヒゲ大陰線（上値抵抗感）")

   # 高値切り下げ（ダブルトップ傾向）
   if prev['High'] > latest['High'] and latest['Close'] < prev['Close']:
       patterns.append("高値切り下げ（上昇推移の失速）")

   return patterns

def analyze_5m_short_signal(ticker_symbol: str, name: str):
   """5分足のVWAP下抜け判定"""
   df_5m = fetch_jpx_data(ticker_symbol, interval="5m", period="5d")
   if df_5m.empty or len(df_5m) < 2:
       return None

   df_5m['TP'] = (df_5m['High'] + df_5m['Low'] + df_5m['Close']) / 3
   df_5m['PV'] = df_5m['TP'] * df_5m['Volume']
   df_5m['Date'] = df_5m.index.date
   cum_pv = df_5m.groupby('Date')['PV'].cumsum()
   cum_vol = df_5m.groupby('Date')['Volume'].cumsum()
   df_5m['VWAP'] = cum_pv / cum_vol

   latest = df_5m.iloc[-1]
   prev = df_5m.iloc[-2]

   is_vwap_break = (prev['Close'] >= prev['VWAP']) and (latest['Close'] < latest['VWAP'])
   patterns = detect_chart_patterns(df_5m)

   latest_close = latest['Close']
   stop_loss = round(latest_close * 1.02)
   take_profit = round(latest_close * 0.95)

   return {
       "code": ticker_symbol,
       "name": name,
       "price": round(latest_close, 1),
       "vwap": round(latest['VWAP'], 1),
       "is_vwap_break": is_vwap_break,
       "shape_patterns": patterns,
       "stop_loss": stop_loss,
       "take_profit": take_profit
   }

# ---------------------------------------------------------
# 2. Web APIサーバー（FastAPI）
# ---------------------------------------------------------
app = FastAPI()

app.add_middleware(
   CORSMiddleware,
   allow_origins=["*"],
   allow_methods=["*"],
   allow_headers=["*"],
)

JAPAN_STOCKS = {
   "6857": "アドバンテスト",
   "8035": "東京エレクトロン",
   "6146": "ディスコ",
   "9984": "ソフトバンクグループ",
   "7011": "三菱重工業",
   "6526": "ソシオネクスト"
}

@app.get("/")
def get_index():
   return FileResponse("index.html")

@app.get("/api/signals")
def get_signals():
   results = []
   for code, name in JAPAN_STOCKS.items():
       df_daily = fetch_jpx_data(code, interval="1d", period="3mo")
       is_candidate, dev, rsi = check_daily_overbought(df_daily)
       if is_candidate:
           signal_data = analyze_5m_short_signal(code, name)
           if signal_data:
               signal_data["daily_dev"] = dev
               signal_data["daily_rsi"] = rsi
               results.append(signal_data)
   return results

@app.get("/api/chart/{code}")
def get_chart_data(code: str):
   df_5m = fetch_jpx_data(code, interval="5m", period="5d")
   if df_5m.empty:
       return {"candles": [], "vwap": []}

   df_5m['TP'] = (df_5m['High'] + df_5m['Low'] + df_5m['Close']) / 3
   df_5m['PV'] = df_5m['TP'] * df_5m['Volume']
   df_5m['Date'] = df_5m.index.date
   cum_pv = df_5m.groupby('Date')['PV'].cumsum()
   cum_vol = df_5m.groupby('Date')['Volume'].cumsum()
   df_5m['VWAP'] = cum_pv / cum_vol

   candles, vwap_list = [], []
   for index, row in df_5m.tail(60).iterrows():
       timestamp = int(index.timestamp())
       candles.append({
           "time": timestamp,
           "open": round(row['Open'], 1),
           "high": round(row['High'], 1),
           "low": round(row['Low'], 1),
           "close": round(row['Close'], 1)
       })
       vwap_list.append({
           "time": timestamp,
           "value": round(row['VWAP'], 1)
       })

   return {"candles": candles, "vwap": vwap_list}
