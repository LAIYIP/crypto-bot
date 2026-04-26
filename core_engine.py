"""
Bot v7.0 核心數據引擎
功能：只負責抓取 K 線數據，不做任何硬編碼分析
所有分析邏輯交由 AI（Gemini）處理
"""
import time
import requests
import pytz
from datetime import datetime

HKT = pytz.timezone("Asia/Hong_Kong")

_ENDPOINTS = [
    "https://data-api.binance.vision",
    "https://api.binance.us",
    "https://api1.binance.com",
    "https://api2.binance.com",
]

_INTERVAL_MAP = {
    "3m":  "3m",
    "15m": "15m",
    "1h":  "1h",
    "4h":  "4h",
    "1d":  "1d",
}

def fetch_klines(symbol: str, interval: str, limit: int = 100 ) -> list:
    iv = _INTERVAL_MAP.get(interval, interval)
    params = {"symbol": symbol, "interval": iv, "limit": limit}
    for base in _ENDPOINTS:
        try:
            url = f"{base}/api/v3/klines"
            r = requests.get(url, params=params, timeout=8)
            if r.status_code == 200:
                raw = r.json()
                if isinstance(raw, list) and len(raw) > 0:
                    return [
                        {
                            "time":   datetime.fromtimestamp(k[0] / 1000, tz=HKT).strftime("%Y-%m-%d %H:%M"),
                            "open":   float(k[1]),
                            "high":   float(k[2]),
                            "low":    float(k[3]),
                            "close":  float(k[4]),
                            "volume": float(k[5]),
                        }
                        for k in raw
                    ]
        except Exception:
            continue
    return []


def get_current_price(symbol: str) -> float:
    for base in _ENDPOINTS:
        try:
            r = requests.get(f"{base}/api/v3/ticker/price",
                             params={"symbol": symbol}, timeout=5)
            if r.status_code == 200:
                return float(r.json()["price"])
        except Exception:
            continue
    return 0.0


def get_24h_stats(symbol: str) -> dict:
    for base in _ENDPOINTS:
        try:
            r = requests.get(f"{base}/api/v3/ticker/24hr",
                             params={"symbol": symbol}, timeout=5)
            if r.status_code == 200:
                d = r.json()
                return {
                    "price_change_pct": float(d.get("priceChangePercent", 0)),
                    "high_24h":         float(d.get("highPrice", 0)),
                    "low_24h":          float(d.get("lowPrice", 0)),
                    "volume_24h":       float(d.get("volume", 0)),
                    "quote_volume_24h": float(d.get("quoteVolume", 0)),
                }
        except Exception:
            continue
    return {}


def fetch_market_data(symbol: str) -> dict:
    current_price = get_current_price(symbol)
    if current_price == 0:
        return {}
    return {
        "symbol":        symbol,
        "current_price": current_price,
        "stats_24h":     get_24h_stats(symbol),
        "klines_4h":     fetch_klines(symbol, "4h",  limit=100),
        "klines_1h":     fetch_klines(symbol, "1h",  limit=100),
        "klines_15m":    fetch_klines(symbol, "15m", limit=100),
        "klines_3m":     fetch_klines(symbol, "3m",  limit=60),
        "fetch_time":    datetime.now(HKT).strftime("%m-%d %H:%M HKT"),
    }
