#!/usr/bin/env python3
"""
ICT/SMC 加密貨幣交易信號機械人 v5.0
=====================================
三大功能：
  ① 自動入場訊號（保留 v4.2 邏輯，隨時觸發）
  ② 定時盤勢分析（BTC，每日 5 次：08:00 / 12:00 / 17:00 / 20:30 / 23:30 HKT）
     格式：當前偏向 + 看漲情景（SL/入場/TP 由上至下）+ 看跌情景（SL/入場/TP 由上至下）
  ③ 按需詳細報告（用戶發送幣種名稱，例如「BTC」「ETH」即時回覆）

框架：4H 大方向 → 1H 結構 → 15M OB/FVG 關鍵位 → 3M MSS 確認
數據源：data-api.binance.vision（無地區限制）
"""

import os
import asyncio
import logging
import requests
import time
import json
import pandas as pd
import numpy as np
from telegram import Bot, Update
from telegram.ext import Application, MessageHandler, CommandHandler, filters, ContextTypes
from datetime import datetime, timezone, timedelta
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── 設定 ──────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")
GEMINI_API_KEY     = os.environ.get("GEMINI_API_KEY", "")

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
SCAN_INTERVAL = 60          # 入場訊號掃描間隔（秒）
MIN_RR        = 2.0
RISK_FULL     = 50.0
RISK_HALF     = 25.0
HKT = timezone(timedelta(hours=8))

# 定時盤勢分析時間（HKT，格式 HH:MM）
SCHEDULED_TIMES = ["08:00", "12:00", "17:00", "20:30", "23:30"]

BINANCE_ENDPOINTS = [
    "https://data-api.binance.vision",
    "https://api.binance.us",
]

order_counters: dict = defaultdict(int)

# ── Binance API ────────────────────────────────────────────────────
def get_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame | None:
    for base in BINANCE_ENDPOINTS:
        try:
            r = requests.get(
                f"{base}/api/v3/klines",
                params={"symbol": symbol, "interval": interval, "limit": limit},
                timeout=15
            )
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, dict) or not data:
                    continue
                df = pd.DataFrame(data, columns=[
                    'ts','open','high','low','close','volume',
                    'cts','qv','tr','tbb','tbq','ign'])
                df = df[['ts','open','high','low','close','volume']].copy()
                df['ts'] = pd.to_datetime(df['ts'], unit='ms')
                for col in ['open','high','low','close','volume']:
                    df[col] = df[col].astype(float)
                return df
        except Exception as e:
            logger.warning(f"{base} {symbol} {interval}: {e}")
    return None

def get_daily_weekly_levels(symbol: str) -> dict:
    levels = {}
    for base in BINANCE_ENDPOINTS:
        try:
            r = requests.get(f"{base}/api/v3/klines",
                params={"symbol": symbol, "interval": "1d", "limit": 3}, timeout=15)
            if r.status_code == 200:
                d = r.json()
                if isinstance(d, list) and len(d) >= 2:
                    levels['PDH'] = float(d[-2][2])
                    levels['PDL'] = float(d[-2][3])
                    levels['DO']  = float(d[-1][1])
                    break
        except Exception as e:
            logger.warning(f"get_daily {base}: {e}")
    for base in BINANCE_ENDPOINTS:
        try:
            r2 = requests.get(f"{base}/api/v3/klines",
                params={"symbol": symbol, "interval": "1w", "limit": 3}, timeout=15)
            if r2.status_code == 200:
                w = r2.json()
                if isinstance(w, list) and len(w) >= 2:
                    levels['PWH'] = float(w[-2][2])
                    levels['PWL'] = float(w[-2][3])
                    levels['WO']  = float(w[-1][1])
                    break
        except Exception as e:
            logger.warning(f"get_weekly {base}: {e}")
    return levels

def get_ticker_24h(symbol: str) -> dict:
    for base in BINANCE_ENDPOINTS:
        try:
            r = requests.get(f"{base}/api/v3/ticker/24hr",
                params={"symbol": symbol}, timeout=10)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
    return {}

# ── 工具函數 ───────────────────────────────────────────────────────
def fmt(price: float, symbol: str = "BTC") -> str:
    if "BTC" in symbol:
        return f"{price:,.2f}"
    elif "ETH" in symbol:
        return f"{price:,.2f}"
    else:
        return f"{price:,.3f}"

def next_order_id(symbol: str, direction: str) -> str:
    d = "S" if direction == "bearish" else "L"
    key = f"{symbol[:3]}{d}"
    order_counters[key] += 1
    now = datetime.now(HKT)
    return f"#{symbol[:3]}-{now.strftime('%Y%m%d-%H%M')}-{d}{order_counters[key]:03d}"

def calc_atr(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or len(df) < period + 1:
        return 0.0
    h = df['high'].values
    l = df['low'].values
    c = df['close'].values
    tr = np.maximum(h[1:] - l[1:],
         np.maximum(abs(h[1:] - c[:-1]),
                    abs(l[1:] - c[:-1])))
    return float(np.mean(tr[-period:]))

# ── 市場結構分析 ───────────────────────────────────────────────────
def find_swing_points(df: pd.DataFrame, n: int = 5) -> tuple:
    highs, lows = [], []
    for i in range(n, len(df) - n):
        h = float(df.iloc[i]['high'])
        l = float(df.iloc[i]['low'])
        if h == float(df.iloc[i-n:i+n+1]['high'].max()):
            highs.append((i, h))
        if l == float(df.iloc[i-n:i+n+1]['low'].min()):
            lows.append((i, l))
    return highs, lows

def find_swing_dual(df: pd.DataFrame) -> tuple:
    sh3, sl3 = find_swing_points(df, n=3)
    sh8, sl8 = find_swing_points(df, n=8)
    def merge(a, b):
        combined = list(a)
        for item in b:
            if not any(abs(item[1] - x[1]) / max(x[1], 0.001) < 0.001 for x in combined):
                combined.append(item)
        return sorted(combined, key=lambda x: x[0])
    return merge(sh3, sh8), merge(sl3, sl8)

def get_market_structure(df: pd.DataFrame, lookback: int = 20) -> str:
    if df is None or len(df) < lookback:
        return "ranging"
    sub = df.iloc[-lookback:]
    highs, lows = find_swing_points(sub, n=3)
    if len(highs) < 2 or len(lows) < 2:
        return "ranging"
    rh = sorted(highs, key=lambda x: x[0])[-2:]
    rl = sorted(lows,  key=lambda x: x[0])[-2:]
    hh = rh[1][1] > rh[0][1]
    hl = rl[1][1] > rl[0][1]
    lh = rh[1][1] < rh[0][1]
    ll = rl[1][1] < rl[0][1]
    if hh and hl:
        return "bullish"
    elif lh and ll:
        return "bearish"
    return "ranging"

def get_bsl_ssl(df: pd.DataFrame) -> tuple:
    highs, lows = find_swing_dual(df)
    price = float(df.iloc[-1]['close'])
    bsl = min([h[1] for h in highs if h[1] > price], default=None)
    ssl = max([l[1] for l in lows  if l[1] < price], default=None)
    return bsl, ssl

def detect_3m_mss(df_3m: pd.DataFrame, direction: str) -> dict | None:
    if df_3m is None or len(df_3m) < 20:
        return None
    recent = df_3m.iloc[-20:].copy().reset_index(drop=True)
    highs, lows = find_swing_points(recent, n=3)
    if not highs or not lows:
        return None
    last = recent.iloc[-1]
    if direction == "bullish":
        last_swing_high = max(highs, key=lambda x: x[0])[1]
        if float(last['close']) > float(last['open']) and float(last['close']) > last_swing_high:
            return {"type": "bullish_mss", "break_price": last_swing_high,
                    "candle_close": float(last['close']), "direction": "bullish"}
    elif direction == "bearish":
        last_swing_low = max(lows, key=lambda x: x[0])[1]
        if float(last['close']) < float(last['open']) and float(last['close']) < last_swing_low:
            return {"type": "bearish_mss", "break_price": last_swing_low,
                    "candle_close": float(last['close']), "direction": "bearish"}
    return None

def find_displacement_fvg(df: pd.DataFrame, direction: str, lookback: int = 10) -> dict | None:
    if df is None or len(df) < 3:
        return None
    recent = df.iloc[-lookback:].copy().reset_index(drop=True)
    best_fvg = None
    best_size = 0
    for i in range(1, len(recent) - 1):
        k1 = recent.iloc[i-1]
        k2 = recent.iloc[i]
        k3 = recent.iloc[i+1]
        body_k2  = abs(float(k2['close']) - float(k2['open']))
        range_k2 = float(k2['high']) - float(k2['low'])
        if range_k2 == 0 or body_k2 / range_k2 < 0.5:
            continue
        if direction == "bullish":
            gap_low  = float(k1['high'])
            gap_high = float(k3['low'])
            if gap_high > gap_low and float(k2['close']) > float(k2['open']):
                size = gap_high - gap_low
                if size > best_size:
                    best_size = size
                    best_fvg = {"type": "bullish_disp_fvg",
                                "low": gap_low, "high": gap_high,
                                "mid": (gap_low + gap_high) / 2}
        elif direction == "bearish":
            gap_high = float(k1['low'])
            gap_low  = float(k3['high'])
            if gap_high > gap_low and float(k2['close']) < float(k2['open']):
                size = gap_high - gap_low
                if size > best_size:
                    best_size = size
                    best_fvg = {"type": "bearish_disp_fvg",
                                "low": gap_low, "high": gap_high,
                                "mid": (gap_low + gap_high) / 2}
    return best_fvg

# ── 關鍵區識別 ─────────────────────────────────────────────────────
def find_order_blocks(df: pd.DataFrame) -> list:
    obs = []
    if df is None or len(df) < 10:
        return obs
    lc = float(df.iloc[-1]['close'])
    for i in range(3, len(df) - 3):
        c    = df.iloc[i]
        body = abs(float(c['close']) - float(c['open']))
        rng  = float(c['high']) - float(c['low'])
        if rng == 0 or body / rng < 0.4:
            continue
        next3_high = float(df.iloc[i+1:i+4]['high'].max())
        next3_low  = float(df.iloc[i+1:i+4]['low'].min())
        if float(c['close']) < float(c['open']) and next3_high > float(c['high']):
            mid = (float(c['high']) + float(c['low'])) / 2
            obs.append({"type": "ob_bull", "direction": "bullish",
                        "high": float(c['high']), "low": float(c['low']), "mid": mid,
                        "label": "OB 需求區（看漲）", "touched": abs(lc - mid) / mid < 0.01})
        elif float(c['close']) > float(c['open']) and next3_low < float(c['low']):
            mid = (float(c['high']) + float(c['low'])) / 2
            obs.append({"type": "ob_bear", "direction": "bearish",
                        "high": float(c['high']), "low": float(c['low']), "mid": mid,
                        "label": "OB 供應區（看跌）", "touched": abs(lc - mid) / mid < 0.01})
    return obs

def find_fvg(df: pd.DataFrame) -> list:
    fvgs = []
    if df is None or len(df) < 3:
        return fvgs
    lc = float(df.iloc[-1]['close'])
    for i in range(1, len(df) - 1):
        k1 = df.iloc[i-1]
        k3 = df.iloc[i+1]
        gap_bull_lo = float(k1['high'])
        gap_bull_hi = float(k3['low'])
        if gap_bull_hi > gap_bull_lo:
            mid = (gap_bull_lo + gap_bull_hi) / 2
            fvgs.append({"type": "fvg_bull", "direction": "bullish",
                         "high": gap_bull_hi, "low": gap_bull_lo, "mid": mid,
                         "label": "FVG（看漲）", "touched": abs(lc - mid) / mid < 0.01})
        gap_bear_hi = float(k1['low'])
        gap_bear_lo = float(k3['high'])
        if gap_bear_hi > gap_bear_lo:
            mid = (gap_bear_hi + gap_bear_lo) / 2
            fvgs.append({"type": "fvg_bear", "direction": "bearish",
                         "high": gap_bear_hi, "low": gap_bear_lo, "mid": mid,
                         "label": "FVG（看跌）", "touched": abs(lc - mid) / mid < 0.01})
    return fvgs

def find_ifvg(df: pd.DataFrame, fvgs: list) -> list:
    ifvgs = []
    if df is None or len(df) < 5 or not fvgs:
        return ifvgs
    lc = float(df.iloc[-1]['close'])
    for fvg in fvgs:
        if fvg.get('touched'):
            mid = fvg['mid']
            ifvgs.append({"type": "ifvg", "direction": fvg['direction'],
                          "high": fvg['high'], "low": fvg['low'], "mid": mid,
                          "label": "IFVG（已填補反轉）", "touched": abs(lc - mid) / mid < 0.01})
    return ifvgs

def find_eqh_eql(df: pd.DataFrame, tol: float = 0.002) -> tuple:
    if df is None or len(df) < 10:
        return [], []
    highs, lows = find_swing_points(df, n=3)
    eqh = []
    for i in range(len(highs)):
        for j in range(i+1, len(highs)):
            if abs(highs[i][1] - highs[j][1]) / highs[i][1] < tol:
                eqh.append({"price": (highs[i][1] + highs[j][1]) / 2, "label": "EQH 等高"})
    eql = []
    for i in range(len(lows)):
        for j in range(i+1, len(lows)):
            if abs(lows[i][1] - lows[j][1]) / lows[i][1] < tol:
                eql.append({"price": (lows[i][1] + lows[j][1]) / 2, "label": "EQL 等低"})
    return eqh, eql

def calc_fib(swing_low: float, swing_high: float, direction: str) -> dict:
    diff = swing_high - swing_low
    if direction == "bearish":
        return {
            "0.5":   swing_high - diff * 0.5,
            "0.618": swing_high - diff * 0.618,
            "0.705": swing_high - diff * 0.705,
            "0.786": swing_high - diff * 0.786,
        }
    else:
        return {
            "0.5":   swing_low + diff * 0.5,
            "0.618": swing_low + diff * 0.618,
            "0.705": swing_low + diff * 0.705,
            "0.786": swing_low + diff * 0.786,
        }

def check_hierarchy_trap(price: float, direction: str,
                          zones_1h: list, zones_15m: list) -> tuple:
    if direction == "bullish":
        nearest_15m = min(
            [z for z in zones_15m if z.get('direction') == 'bullish' and z.get('mid', 0) < price],
            key=lambda z: abs(z.get('mid', 0) - price), default=None)
        if nearest_15m:
            untested_1h = [z for z in zones_1h
                           if z.get('direction') == 'bullish'
                           and z.get('mid', 0) < nearest_15m.get('mid', 0)
                           and not z.get('touched', False)]
            if untested_1h:
                return True, "層級陷阱：15M 支撐下方有 1H 未測試需求區"
    return False, ""

def find_sl_anchor_zone(entry_price: float, direction: str,
                         zones_15m: list, zones_4h: list,
                         highs_15m: list, lows_15m: list,
                         atr: float) -> tuple:
    if direction == "bullish":
        candidates = [z for z in zones_15m + zones_4h
                      if z.get('direction') == 'bullish'
                      and z.get('low', 0) < entry_price
                      and z.get('touched', False)]
        if candidates:
            anchor = max(candidates, key=lambda z: z.get('low', 0))
            sl = anchor['low'] * 0.999
            return sl, f"{anchor.get('label', 'OB/FVG')} 底部外"
        if lows_15m:
            sl_swing = min([l[1] for l in lows_15m if l[1] < entry_price], default=entry_price - atr * 2)
            return sl_swing * 0.999, "15M Swing Low 外"
        return entry_price - atr * 2, "ATR×2 保護"
    else:
        candidates = [z for z in zones_15m + zones_4h
                      if z.get('direction') == 'bearish'
                      and z.get('high', 0) > entry_price
                      and z.get('touched', False)]
        if candidates:
            anchor = min(candidates, key=lambda z: z.get('high', 0))
            sl = anchor['high'] * 1.001
            return sl, f"{anchor.get('label', 'OB/FVG')} 頂部外"
        if highs_15m:
            sl_swing = max([h[1] for h in highs_15m if h[1] > entry_price], default=entry_price + atr * 2)
            return sl_swing * 1.001, "15M Swing High 外"
        return entry_price + atr * 2, "ATR×2 保護"

def calc_confluence(zone: dict, all_zones: list, fib: dict | None) -> tuple:
    score = 0
    reasons = []
    zone_mid = zone.get('mid', 0)
    if zone_mid == 0:
        return 0, []
    for other in all_zones:
        if other is zone:
            continue
        other_mid = other.get('mid', 0)
        if other_mid and abs(zone_mid - other_mid) / zone_mid < 0.003:
            score += 1
            reasons.append(other.get('label', '其他關鍵區'))
    if fib:
        ote_low  = min(fib.get("0.618", 0), fib.get("0.786", 0))
        ote_high = max(fib.get("0.618", 0), fib.get("0.786", 0))
        if ote_low > 0 and ote_low <= zone_mid <= ote_high:
            score += 2
            reasons.append("FIB OTE 區（0.618-0.786）")
        elif fib.get("0.705", 0) and abs(zone_mid - fib["0.705"]) / fib["0.705"] < 0.003:
            score += 2
            reasons.append("FIB 0.705（最佳入場）")
        elif fib.get("0.5", 0) and abs(zone_mid - fib["0.5"]) / fib["0.5"] < 0.003:
            score += 1
            reasons.append("FIB 0.5 均衡點")
    if zone.get('tf') == '1h':
        score += 2
        reasons.append("1H 級別關鍵位")
    return score, reasons

# ── 主要分析函數 ───────────────────────────────────────────────────
def analyze_symbol(symbol: str) -> dict:
    result = {"symbol": symbol, "error": None}
    df_1h  = get_klines(symbol, "1h",  500)
    df_4h  = get_klines(symbol, "4h",  200)
    df_15m = get_klines(symbol, "15m", 300)
    df_3m  = get_klines(symbol, "3m",  200)
    levels = get_daily_weekly_levels(symbol)

    if df_1h is None or df_15m is None or df_3m is None:
        result["error"] = "無法取得數據"
        return result

    price = float(df_15m.iloc[-1]['close'])
    result["price"]  = price
    result["levels"] = levels

    struct_1h = get_market_structure(df_1h, lookback=30)
    result["struct_1h"] = struct_1h
    bsl, ssl = get_bsl_ssl(df_1h.iloc[-100:])
    result["bsl"] = bsl
    result["ssl"] = ssl

    highs_1h, lows_1h = find_swing_dual(df_1h.iloc[-100:])
    fib_1h = None
    if highs_1h and lows_1h:
        h1 = max(highs_1h, key=lambda x: x[0])[1]
        l1 = min(lows_1h,  key=lambda x: x[0])[1]
        fib_1h = calc_fib(l1, h1, struct_1h if struct_1h != "ranging" else "bearish")
    result["fib_1h"] = fib_1h

    obs_1h  = find_order_blocks(df_1h.iloc[-100:])
    fvgs_1h = find_fvg(df_1h.iloc[-100:])
    for z in obs_1h + fvgs_1h:
        z['tf'] = '1h'
    result["zones_1h"] = obs_1h + fvgs_1h

    zones_4h = []
    if df_4h is not None:
        obs_4h = find_order_blocks(df_4h.iloc[-60:])
        for z in obs_4h:
            z['tf'] = '4h'
        zones_4h = obs_4h
    result["zones_4h"] = zones_4h

    # 4H 結構方向（新增：用於定時分析）
    struct_4h = get_market_structure(df_4h, lookback=20) if df_4h is not None else "ranging"
    result["struct_4h"] = struct_4h

    obs_15m   = find_order_blocks(df_15m)
    fvgs_15m  = find_fvg(df_15m)
    ifvgs_15m = find_ifvg(df_15m, fvgs_15m)
    for z in obs_15m + fvgs_15m + ifvgs_15m:
        z['tf'] = '15m'
    zones_15m = obs_15m + fvgs_15m + ifvgs_15m
    result["zones_15m"] = zones_15m

    eqh_list, eql_list = find_eqh_eql(df_15m)
    result["eqh"] = eqh_list
    result["eql"] = eql_list

    highs_15m, lows_15m = find_swing_dual(df_15m.iloc[-50:])
    result["highs_15m"] = highs_15m
    result["lows_15m"]  = lows_15m

    fib_15m = None
    if highs_15m and lows_15m:
        h15 = max(highs_15m, key=lambda x: x[0])[1]
        l15 = min(lows_15m,  key=lambda x: x[0])[1]
        fib_15m = calc_fib(l15, h15, struct_1h if struct_1h != "ranging" else "bearish")
    result["fib_15m"] = fib_15m
    result["atr_15m"] = calc_atr(df_15m)

    if struct_1h == "bearish":
        mss_trend   = detect_3m_mss(df_3m, "bearish")
        mss_counter = detect_3m_mss(df_3m, "bullish")
    elif struct_1h == "bullish":
        mss_trend   = detect_3m_mss(df_3m, "bullish")
        mss_counter = detect_3m_mss(df_3m, "bearish")
    else:
        mss_trend = mss_counter = None

    result["mss_trend"]   = mss_trend
    result["mss_counter"] = mss_counter
    result["df_3m"]  = df_3m
    result["df_15m"] = df_15m
    result["df_1h"]  = df_1h
    result["df_4h"]  = df_4h
    return result

# ── 入場信號邏輯（保留 v4.2）─────────────────────────────────────
def find_entry_signal(result: dict) -> dict | None:
    symbol    = result["symbol"]
    price     = result["price"]
    struct_1h = result["struct_1h"]
    mss_trend   = result.get("mss_trend")
    mss_counter = result.get("mss_counter")
    fib_1h    = result.get("fib_1h")
    fib_15m   = result.get("fib_15m")
    atr_15m   = result.get("atr_15m", 0)
    levels    = result.get("levels", {})
    zones_1h  = result.get("zones_1h", [])
    zones_4h  = result.get("zones_4h", [])
    zones_15m = result.get("zones_15m", [])
    highs_15m = result.get("highs_15m", [])
    lows_15m  = result.get("lows_15m", [])
    eqh       = result.get("eqh", [])
    eql       = result.get("eql", [])
    df_3m     = result.get("df_3m")

    if struct_1h == "ranging":
        return None

    def build_signal(direction: str, trade_type: str, mss: dict) -> dict | None:
        is_trend    = (trade_type == "順勢")
        risk_amount = RISK_FULL if is_trend else RISK_HALF

        dir_zones = [z for z in zones_15m if z.get('direction') == direction]
        if not dir_zones:
            return None

        active_zone = None
        for z in sorted(dir_zones, key=lambda x: abs(x.get('mid', 0) - price)):
            if abs(z.get('mid', 0) - price) / price < 0.01:
                active_zone = z
                break
        if not active_zone:
            active_zone = min(dir_zones, key=lambda z: abs(z.get('mid', 0) - price))

        if is_trend:
            is_trap, _ = check_hierarchy_trap(price, direction, zones_1h, zones_15m)
            if is_trap:
                return None

        if not is_trend:
            strong_1h = [z for z in zones_1h
                         if z.get('direction') == direction
                         and abs(z.get('mid', 0) - price) / price < 0.015]
            key_levels = [v for v in [levels.get('PDH'), levels.get('PDL'),
                                       levels.get('PWH'), levels.get('PWL'),
                                       levels.get('DO'), levels.get('WO')] if v]
            nearby_level = any(abs(price - l) / price < 0.01 for l in key_levels)
            if not strong_1h and not nearby_level:
                return None

        disp_fvg = find_displacement_fvg(df_3m, direction, lookback=10)

        entry_price = price
        entry_label = "市價入場"
        if disp_fvg:
            entry_price = disp_fvg["mid"]
            entry_label = f"FVG 掛單 {fmt(disp_fvg['low'], symbol)} - {fmt(disp_fvg['high'], symbol)}"
        elif fib_1h:
            ote_705 = fib_1h.get("0.705", 0)
            if ote_705 and abs(price - ote_705) / price < 0.015:
                entry_price = ote_705
                entry_label = "FIB OTE 0.705 掛單"

        sl, sl_desc = find_sl_anchor_zone(
            entry_price, direction, zones_15m, zones_4h,
            highs_15m, lows_15m, atr_15m)
        sl_dist = abs(entry_price - sl)

        min_dist = max(price * 0.005, atr_15m * 1.0)
        if sl_dist < min_dist:
            sl = entry_price - min_dist if direction == "bullish" else entry_price + min_dist
            sl_dist = min_dist
            sl_desc += "（已擴展至最小距離）"

        tp1 = tp2 = None
        tp_label = ""
        if fib_15m:
            tp1 = fib_15m.get("0.5")
        if direction == "bearish":
            below_liq = [e["price"] for e in eql if e["price"] < price]
            tp2 = max(below_liq) if below_liq else None
            tp_label = "EQL 流動性" if tp2 else "15M FIB 0.5"
        else:
            above_liq = [e["price"] for e in eqh if e["price"] > price]
            tp2 = min(above_liq) if above_liq else None
            tp_label = "EQH 流動性" if tp2 else "15M FIB 0.5"

        if not tp1:
            tp1 = (entry_price - sl_dist * MIN_RR if direction == "bearish"
                   else entry_price + sl_dist * MIN_RR)
            tp_label = f"1:{MIN_RR:.0f} RR 目標"

        tp_dist = abs(entry_price - tp1)
        rr = tp_dist / sl_dist if sl_dist > 0 else 0
        if rr < MIN_RR:
            return None

        all_zones = zones_15m + zones_1h
        conf_score, conf_reasons = calc_confluence(active_zone, all_zones, fib_1h)
        if not is_trend and conf_score < 2:
            return None

        if is_trend:
            signal_type = "🏄 衝浪者 A（淺回調順勢）" if not disp_fvg else "🏄 衝浪者 B（深回調順勢）"
        else:
            signal_type = "🎯 狙擊手 A（激進逆勢）" if disp_fvg else "🎯 狙擊手 B（穩健逆勢）"

        return {
            "symbol": symbol, "price": price,
            "direction": direction,
            "signal_type": signal_type,
            "trade_type": trade_type,
            "risk_type": "全倉" if is_trend else "半倉",
            "risk_amount": risk_amount,
            "entry_price": entry_price,
            "entry_label": entry_label,
            "sl": sl, "sl_dist": sl_dist, "sl_desc": sl_desc,
            "tp1": tp1, "tp2": tp2,
            "tp_label": tp_label, "rr": rr,
            "zone": active_zone,
            "disp_fvg": disp_fvg,
            "fib_1h": fib_1h,
            "conf_score": conf_score,
            "conf_reasons": conf_reasons,
            "struct_1h": struct_1h,
            "mss": mss,
            "levels": levels,
            "bsl": result.get("bsl"),
            "ssl": result.get("ssl"),
        }

    if mss_trend:
        sig = build_signal(mss_trend["direction"], "順勢", mss_trend)
        if sig:
            return sig
    if mss_counter:
        sig = build_signal(mss_counter["direction"], "逆勢", mss_counter)
        if sig:
            return sig
    return None

# ── 入場訊號格式化 ─────────────────────────────────────────────────
def format_signal_message(sig: dict) -> str:
    symbol     = sig["symbol"]
    sym_short  = symbol.replace("USDT", "/USDT")
    direction  = sig["direction"]
    dir_emoji  = "🔴 做空 (Short)" if direction == "bearish" else "🟢 做多 (Long)"
    struct     = sig["struct_1h"]
    struct_emoji = "⬇️ 看跌" if struct == "bearish" else "⬆️ 看漲" if struct == "bullish" else "↔️ 橫盤"
    trade_type = sig["trade_type"]
    risk_type  = sig["risk_type"]
    risk_amount = sig["risk_amount"]
    entry      = sig["entry_price"]
    sl         = sig["sl"]
    sl_desc    = sig.get("sl_desc", "關鍵位外")
    tp1        = sig["tp1"]
    tp2        = sig.get("tp2")
    rr         = sig["rr"]
    zone       = sig["zone"]
    disp_fvg   = sig.get("disp_fvg")
    fib_1h     = sig.get("fib_1h")
    conf_score = sig.get("conf_score", 0)
    conf_reasons = sig.get("conf_reasons", [])
    levels     = sig.get("levels", {})
    bsl        = sig.get("bsl")
    ssl        = sig.get("ssl")
    signal_type = sig["signal_type"]
    entry_label = sig["entry_label"]
    tp_label   = sig["tp_label"]
    order_id   = next_order_id(symbol, direction)

    if conf_score >= 4:
        conf_grade = "A+（極強匯聚）"
    elif conf_score >= 3:
        conf_grade = "A（強匯聚）"
    elif conf_score >= 2:
        conf_grade = "B（中等匯聚）"
    else:
        conf_grade = "C（單一關鍵位）"

    msg  = f"🚨 【入場信號】{sym_short}\n"
    msg += "━━━━━━━━━━━━━━━━━━\n"
    msg += f"📋 訂單編號: {order_id}\n"
    msg += f"{signal_type} | {risk_type}（{risk_amount:.0f} USDT 風險）\n\n"
    msg += "✅ 確認條件:\n"
    msg += f"• 1H 結構: {struct_emoji}\n"
    msg += f"• 3M MSS 確認（{direction}）\n"
    msg += f"• 關鍵區: {zone.get('label', '關鍵位')}\n"
    msg += f"• 關鍵區範圍: {fmt(zone.get('low', 0), symbol)} - {fmt(zone.get('high', 0), symbol)}\n"
    if disp_fvg:
        msg += f"• 位移 FVG: {fmt(disp_fvg['low'], symbol)} - {fmt(disp_fvg['high'], symbol)}\n"
    msg += f"• Confluence: {conf_grade}\n"
    if conf_reasons:
        msg += f"  └ {' + '.join(conf_reasons[:3])}\n"
    msg += "\n"
    if fib_1h:
        ote_low  = min(fib_1h.get("0.618", 0), fib_1h.get("0.786", 0))
        ote_high = max(fib_1h.get("0.618", 0), fib_1h.get("0.786", 0))
        if ote_low > 0:
            msg += f"📐 1H FIB OTE 區: {fmt(ote_low, symbol)} - {fmt(ote_high, symbol)}\n"
            msg += f"   0.705（最佳入場）: {fmt(fib_1h.get('0.705', 0), symbol)}\n\n"
    msg += f"📈 交易方向: {dir_emoji}\n\n"
    msg += f"💵 入場方式: {entry_label}\n"
    if disp_fvg:
        msg += f"   30% 市價入場: {fmt(sig['price'], symbol)}\n"
        msg += f"   70% FVG 掛單: {fmt(disp_fvg['mid'], symbol)}\n"
    else:
        msg += f"   入場價格: {fmt(entry, symbol)}\n"
    msg += "\n"
    msg += f"🛑 止損 (SL): {fmt(sl, symbol)}\n"
    msg += f"   └ {sl_desc}\n\n"
    msg += f"🎯 止盈 TP1: {fmt(tp1, symbol)}\n"
    msg += f"   └ {tp_label}\n"
    if tp2 and tp2 != tp1:
        msg += f"🎯 止盈 TP2: {fmt(tp2, symbol)}\n"
        msg += f"   └ 延伸目標（流動性）\n"
    msg += f"📊 預計 RR: 1:{rr:.1f}\n\n"
    if bsl or ssl:
        msg += "💧 流動性參考:\n"
        if bsl:
            msg += f"   BSL（上方）: {fmt(bsl, symbol)}\n"
        if ssl:
            msg += f"   SSL（下方）: {fmt(ssl, symbol)}\n"
        msg += "\n"
    if any(levels.get(k) for k in ['DO', 'WO', 'PDH', 'PDL', 'PWH', 'PWL']):
        msg += "📌 重要水平:\n"
        if levels.get('DO'):
            msg += f"   今日開盤 (DO): {fmt(levels['DO'], symbol)}\n"
        if levels.get('WO'):
            msg += f"   本週開盤 (WO): {fmt(levels['WO'], symbol)}\n"
        if levels.get('PDH'):
            msg += f"   PDH: {fmt(levels['PDH'], symbol)}  |  PDL: {fmt(levels['PDL'], symbol)}\n"
        if levels.get('PWH'):
            msg += f"   PWH: {fmt(levels['PWH'], symbol)}  |  PWL: {fmt(levels['PWL'], symbol)}\n"
        msg += "\n"
    msg += "⚠️ 確認風險後入場"
    return msg

# ── 定時盤勢分析格式化（新功能）────────────────────────────────────
def format_directional_analysis(result: dict, session_name: str) -> str:
    """
    雙向情景分析格式
    多單：TP（上）→ 入場 → SL（下）
    空單：SL（上）→ 入場 → TP（下）
    """
    symbol    = result["symbol"]
    sym_short = symbol.replace("USDT", "/USDT")
    price     = result["price"]
    struct_1h = result.get("struct_1h", "ranging")
    struct_4h = result.get("struct_4h", "ranging")
    levels    = result.get("levels", {})
    bsl       = result.get("bsl")
    ssl       = result.get("ssl")
    zones_1h  = result.get("zones_1h", [])
    zones_15m = result.get("zones_15m", [])
    zones_4h  = result.get("zones_4h", [])
    highs_15m = result.get("highs_15m", [])
    lows_15m  = result.get("lows_15m", [])
    fib_1h    = result.get("fib_1h")
    atr_15m   = result.get("atr_15m", price * 0.005)
    now       = datetime.now(HKT)

    # ── 整體偏向判斷 ──────────────────────────────────────────────
    if struct_4h == "bullish" and struct_1h == "bullish":
        bias = "偏看漲"
        bias_emoji = "🟢"
        bias_reason = "4H + 1H 結構雙重看漲（HH/HL 形態）"
        primary_dir = "bullish"
    elif struct_4h == "bearish" and struct_1h == "bearish":
        bias = "偏看跌"
        bias_emoji = "🔴"
        bias_reason = "4H + 1H 結構雙重看跌（LH/LL 形態）"
        primary_dir = "bearish"
    elif struct_4h == "bullish" and struct_1h == "bearish":
        bias = "大級別看漲，短線回調"
        bias_emoji = "🟡"
        bias_reason = "4H 看漲但 1H 出現回調結構，等待 1H 確認轉漲"
        primary_dir = "bullish"
    elif struct_4h == "bearish" and struct_1h == "bullish":
        bias = "大級別看跌，短線反彈"
        bias_emoji = "🟡"
        bias_reason = "4H 看跌但 1H 出現反彈結構，等待 1H 確認轉跌"
        primary_dir = "bearish"
    else:
        bias = "橫盤整理"
        bias_emoji = "⚪"
        bias_reason = "4H/1H 結構不明確，等待方向選擇"
        primary_dir = "bullish"

    # ── 計算看漲情景 TP / 入場 / SL ──────────────────────────────
    # 入場：最近的看漲 OB/FVG 中點
    bull_zones = sorted(
        [z for z in zones_15m + zones_1h if z.get('direction') == 'bullish' and z.get('mid', 0) < price * 1.005],
        key=lambda z: abs(z.get('mid', 0) - price))
    bull_entry_zone = bull_zones[0] if bull_zones else None
    bull_entry = bull_entry_zone['mid'] if bull_entry_zone else price
    bull_entry_label = bull_entry_zone.get('label', '15M 需求區') if bull_entry_zone else "現價附近"

    # 入場條件：3M 看漲 MSS 確認
    bull_entry_cond = "等待 3M 實體陽線突破近期 Swing High（MSS 確認）"

    # SL：入場區底部外側
    bull_sl, bull_sl_desc = find_sl_anchor_zone(
        bull_entry, "bullish", zones_15m, zones_4h, highs_15m, lows_15m, atr_15m)
    min_dist = max(price * 0.005, atr_15m)
    if abs(bull_entry - bull_sl) < min_dist:
        bull_sl = bull_entry - min_dist

    # TP：上方 BSL / EQH / 1H OB 供應區
    bull_tp = None
    bull_tp_label = ""
    if bsl and bsl > price:
        bull_tp = bsl
        bull_tp_label = "BSL 上方流動性"
    else:
        above_resist = sorted(
            [z for z in zones_1h if z.get('direction') == 'bearish' and z.get('mid', 0) > price],
            key=lambda z: z.get('mid', 0))
        if above_resist:
            bull_tp = above_resist[0]['mid']
            bull_tp_label = above_resist[0].get('label', '1H 供應區')
    if not bull_tp:
        bull_tp = bull_entry + abs(bull_entry - bull_sl) * MIN_RR
        bull_tp_label = f"1:{MIN_RR:.0f} RR 目標"

    bull_rr = abs(bull_tp - bull_entry) / abs(bull_entry - bull_sl) if abs(bull_entry - bull_sl) > 0 else 0

    # ── 計算看跌情景 TP / 入場 / SL ──────────────────────────────
    bear_zones = sorted(
        [z for z in zones_15m + zones_1h if z.get('direction') == 'bearish' and z.get('mid', 0) > price * 0.995],
        key=lambda z: abs(z.get('mid', 0) - price))
    bear_entry_zone = bear_zones[0] if bear_zones else None
    bear_entry = bear_entry_zone['mid'] if bear_entry_zone else price
    bear_entry_label = bear_entry_zone.get('label', '15M 供應區') if bear_entry_zone else "現價附近"

    bear_entry_cond = "等待 3M 實體陰線跌破近期 Swing Low（MSS 確認）"

    bear_sl, bear_sl_desc = find_sl_anchor_zone(
        bear_entry, "bearish", zones_15m, zones_4h, highs_15m, lows_15m, atr_15m)
    if abs(bear_entry - bear_sl) < min_dist:
        bear_sl = bear_entry + min_dist

    bear_tp = None
    bear_tp_label = ""
    if ssl and ssl < price:
        bear_tp = ssl
        bear_tp_label = "SSL 下方流動性"
    else:
        below_support = sorted(
            [z for z in zones_1h if z.get('direction') == 'bullish' and z.get('mid', 0) < price],
            key=lambda z: z.get('mid', 0), reverse=True)
        if below_support:
            bear_tp = below_support[0]['mid']
            bear_tp_label = below_support[0].get('label', '1H 需求區')
    if not bear_tp:
        bear_tp = bear_entry - abs(bear_entry - bear_sl) * MIN_RR
        bear_tp_label = f"1:{MIN_RR:.0f} RR 目標"

    bear_rr = abs(bear_entry - bear_tp) / abs(bear_sl - bear_entry) if abs(bear_sl - bear_entry) > 0 else 0

    # ── 組裝訊息 ──────────────────────────────────────────────────
    msg  = f"📊 {sym_short} {session_name} [{now.strftime('%m-%d %H:%M')} HKT]\n"
    msg += "━━━━━━━━━━━━━━━━━━\n"
    msg += f"{bias_emoji} 當前整體偏向：【{bias}】\n"
    msg += f"主要原因：{bias_reason}\n"

    # 加入關鍵水平參考
    key_refs = []
    if levels.get('PDH'):
        key_refs.append(f"PDH {fmt(levels['PDH'], symbol)}")
    if levels.get('PDL'):
        key_refs.append(f"PDL {fmt(levels['PDL'], symbol)}")
    if levels.get('DO'):
        key_refs.append(f"DO {fmt(levels['DO'], symbol)}")
    if bsl:
        key_refs.append(f"BSL {fmt(bsl, symbol)}")
    if ssl:
        key_refs.append(f"SSL {fmt(ssl, symbol)}")
    if key_refs:
        msg += f"關鍵水平：{' | '.join(key_refs[:4])}\n"

    msg += "\n"

    # ── 看漲情景（TP 在上，SL 在下）──────────────────────────────
    msg += "🟢 看漲情景（主路線）：\n" if primary_dir == "bullish" else "🟢 看漲情景（備用路線）：\n"
    msg += f"   🎯 TP：{fmt(bull_tp, symbol)}（{bull_tp_label}）\n"
    msg += f"   📍 入場：{fmt(bull_entry, symbol)}（{bull_entry_label}）\n"
    msg += f"   🛑 SL：{fmt(bull_sl, symbol)}（{bull_sl_desc}）\n"
    msg += f"   入場條件：{bull_entry_cond}\n"
    if bull_rr >= 1.5:
        msg += f"   RR：1:{bull_rr:.1f}\n"

    msg += "\n"

    # ── 看跌情景（SL 在上，TP 在下）──────────────────────────────
    msg += "🔴 看跌情景（主路線）：\n" if primary_dir == "bearish" else "🔴 看跌情景（備用路線）：\n"
    msg += f"   🛑 SL：{fmt(bear_sl, symbol)}（{bear_sl_desc}）\n"
    msg += f"   📍 入場：{fmt(bear_entry, symbol)}（{bear_entry_label}）\n"
    msg += f"   🎯 TP：{fmt(bear_tp, symbol)}（{bear_tp_label}）\n"
    msg += f"   入場條件：{bear_entry_cond}\n"
    if bear_rr >= 1.5:
        msg += f"   RR：1:{bear_rr:.1f}\n"

    msg += "\n"

    # 逆勢備注
    if primary_dir != "ranging":
        counter_dir = "做空" if primary_dir == "bullish" else "做多"
        msg += f"⚠️ 逆勢{counter_dir}屬備用路線，建議減半倉位（風險 $25）\n"

    return msg.rstrip()

# ── 每小時詳細報告（按需查詢）─────────────────────────────────────
def format_on_demand_report(results: list) -> str:
    now = datetime.now(HKT)
    msg  = f"📋 市場詳細報告 [{now.strftime('%m-%d %H:%M')} HKT]\n"
    msg += "━━━━━━━━━━━━━━━━━━\n\n"

    for result in results:
        try:
            if result.get("error"):
                sym = result.get('symbol', '?').replace('USDT', '/USDT')
                msg += f"📌 {sym}  ⚠️ 數據錯誤: {result['error']}\n\n"
                continue

            symbol    = result["symbol"]
            sym_short = symbol.replace("USDT", "/USDT")
            price     = result["price"]
            struct_1h = result.get("struct_1h", "ranging")
            struct_4h = result.get("struct_4h", "ranging")
            struct_emoji = "⬇️ 看跌" if struct_1h == "bearish" else "⬆️ 看漲" if struct_1h == "bullish" else "↔️ 橫盤"
            levels    = result.get("levels", {})
            bsl       = result.get("bsl")
            ssl       = result.get("ssl")
            fib_1h    = result.get("fib_1h")
            zones_1h  = result.get("zones_1h", [])
            zones_15m = result.get("zones_15m", [])

            msg += f"📌 {sym_short}  💲{fmt(price, symbol)}  |  4H {struct_4h}  |  1H {struct_emoji}\n"

            candidates = []
            def add_level(p, lbl, strength, priority):
                if p and p > 0:
                    candidates.append((float(p), lbl, strength, priority))

            add_level(levels.get('PWH'), 'PWH 前週高',  'strong', 1)
            add_level(levels.get('PWL'), 'PWL 前週低',  'strong', 1)
            add_level(levels.get('PDH'), 'PDH 前日高',  'strong', 2)
            add_level(levels.get('PDL'), 'PDL 前日低',  'strong', 2)
            add_level(levels.get('WO'),  'WO 週開盤',   'strong', 3)
            add_level(levels.get('DO'),  'DO 日開盤',   'strong', 3)
            add_level(bsl, 'BSL 上方流動性', 'strong', 4)
            add_level(ssl, 'SSL 下方流動性', 'strong', 4)

            def is_dup(p):
                return any(abs(p - c[0]) / max(c[0], 0.001) < 0.003 for c in candidates)

            for z in zones_1h:
                zm = z.get('mid', 0)
                if zm and not is_dup(zm):
                    candidates.append((zm, z.get('label', '1H 關鍵區'), 'strong', 5))

            for z in zones_15m:
                zm = z.get('mid', 0)
                if zm and not is_dup(zm):
                    candidates.append((zm, z.get('label', '15M 關鍵區'), 'weak', 6))

            above = sorted([(p, lbl, st, pr) for p, lbl, st, pr in candidates if p > price], key=lambda x: x[0])
            below = sorted([(p, lbl, st, pr) for p, lbl, st, pr in candidates if p < price], key=lambda x: x[0], reverse=True)

            def fib_tag(p: float) -> str:
                if not fib_1h:
                    return ""
                tags = []
                for k in ["0.5", "0.618", "0.705", "0.786"]:
                    fv = fib_1h.get(k, 0)
                    if fv and abs(p - fv) / max(fv, 0.001) < 0.005:
                        tags.append(k)
                return f" [+{', '.join(tags)}]" if tags else ""

            outer_above = above[-1] if above else None
            outer_below = below[-1] if below else None
            near_above  = above[:2]
            near_below  = below[:2]

            def is_same(a, b):
                return a is not None and b is not None and abs(a[0] - b[0]) / max(a[0], 0.001) < 0.003

            show_outer_above = outer_above and not any(is_same(outer_above, x) for x in near_above)
            show_outer_below = outer_below and not any(is_same(outer_below, x) for x in near_below)

            if show_outer_above:
                p, lbl, st, _ = outer_above
                emoji = '🔴' if st == 'strong' else '🟠'
                msg += f"   {emoji} {fmt(p, symbol)}  {lbl}{fib_tag(p)}\n"
            for p, lbl, st, _ in reversed(near_above):
                emoji = '🔴' if st == 'strong' else '🟠'
                msg += f"   {emoji} {fmt(p, symbol)}  {lbl}{fib_tag(p)}\n"

            msg += f"   ──── 💲{fmt(price, symbol)} 現價 ────\n"

            for p, lbl, st, _ in near_below:
                emoji = '🟢' if st == 'strong' else '🔵'
                msg += f"   {emoji} {fmt(p, symbol)}  {lbl}{fib_tag(p)}\n"
            if show_outer_below:
                p, lbl, st, _ = outer_below
                emoji = '🟢' if st == 'strong' else '🔵'
                msg += f"   {emoji} {fmt(p, symbol)}  {lbl}{fib_tag(p)}\n"

            msg += "\n"

        except Exception as e:
            sym = result.get('symbol', '?').replace('USDT', '/USDT')
            msg += f"📌 {sym}  ⚠️ 快報生成錯誤: {e}\n\n"
            logger.error(f"format_on_demand_report {sym}: {e}", exc_info=True)

    return msg.rstrip()

# ── Telegram 發送 ──────────────────────────────────────────────────
async def send_msg(bot: Bot, text: str, chat_id: str = None):
    cid = chat_id or TELEGRAM_CHAT_ID
    try:
        # Telegram 單條訊息上限 4096 字符
        if len(text) <= 4096:
            await bot.send_message(chat_id=cid, text=text, parse_mode=None)
        else:
            for i in range(0, len(text), 4000):
                await bot.send_message(chat_id=cid, text=text[i:i+4000], parse_mode=None)
                await asyncio.sleep(0.3)
    except Exception as e:
        logger.error(f"Telegram 發送失敗: {e}")

# ── 定時盤勢分析推送 ───────────────────────────────────────────────
def get_session_name(hkt_time: datetime) -> str:
    hm = hkt_time.strftime("%H:%M")
    mapping = {
        "08:00": "早盤分析",
        "12:00": "午盤分析",
        "17:00": "歐盤前分析",
        "20:30": "美盤開盤分析",
        "23:30": "深夜分析",
    }
    return mapping.get(hm, "盤勢分析")

async def run_scheduled_analysis(bot: Bot):
    """發送 BTC 定時雙向情景分析"""
    try:
        result = analyze_symbol("BTCUSDT")
        if result.get("error"):
            await send_msg(bot, f"⚠️ BTC 定時分析失敗：{result['error']}")
            return
        now = datetime.now(HKT)
        session = get_session_name(now)
        msg = format_directional_analysis(result, session)
        await send_msg(bot, msg)
        logger.info(f"已發送定時分析：{session}")
    except Exception as e:
        logger.error(f"定時分析錯誤: {e}")
        await send_msg(bot, f"⚠️ 定時分析生成失敗：{e}")

# ── 指令處理：/start ───────────────────────────────────────────────
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "✅ ICT/SMC 交易信號機械人 v5.0 已啟動\n\n"
        "📊 功能：\n"
        "① 自動入場訊號（BTC/ETH/SOL，即時觸發）\n"
        "② BTC 定時盤勢分析（08:00 / 12:00 / 17:00 / 20:30 / 23:30 HKT）\n"
        "③ 按需詳細報告（直接輸入幣種名稱）\n\n"
        "💬 查詢指令：\n"
        "輸入 BTC → BTC/USDT 詳細報告\n"
        "輸入 ETH → ETH/USDT 詳細報告\n"
        "輸入 SOL → SOL/USDT 詳細報告\n"
        "輸入 BTC分析 → BTC 即時雙向情景分析\n\n"
        "🎯 框架：4H 大方向 → 1H 結構 → 15M OB/FVG → 3M MSS 確認"
    )
    await update.message.reply_text(msg)

# ── 訊息處理：幣種查詢 ─────────────────────────────────────────────
SYMBOL_MAP = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "BNB": "BNBUSDT",
    "XRP": "XRPUSDT",
    "DOGE": "DOGEUSDT",
    "ADA": "ADAUSDT",
    "AVAX": "AVAXUSDT",
    "DOT": "DOTUSDT",
    "LINK": "LINKUSDT",
}

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    text = update.message.text.strip().upper()
    chat_id = str(update.message.chat_id)

    # 雙向情景分析（例如「BTC分析」「BTC 分析」）
    for sym_key, sym_full in SYMBOL_MAP.items():
        if text in (f"{sym_key}分析", f"{sym_key} 分析", f"{sym_key}ANALYSIS"):
            await update.message.reply_text(f"⏳ 正在分析 {sym_key}/USDT 雙向情景，請稍候...")
            result = analyze_symbol(sym_full)
            if result.get("error"):
                await update.message.reply_text(f"⚠️ 分析失敗：{result['error']}")
                return
            now = datetime.now(HKT)
            msg = format_directional_analysis(result, f"即時分析 {now.strftime('%H:%M')}")
            await send_msg(context.bot, msg, chat_id=chat_id)
            return

    # 詳細報告（例如「BTC」「ETH」）
    if text in SYMBOL_MAP:
        sym_full = SYMBOL_MAP[text]
        await update.message.reply_text(f"⏳ 正在取得 {text}/USDT 詳細報告，請稍候...")
        result = analyze_symbol(sym_full)
        if result.get("error"):
            await update.message.reply_text(f"⚠️ 分析失敗：{result['error']}")
            return
        msg = format_on_demand_report([result])
        await send_msg(context.bot, msg, chat_id=chat_id)
        return

    # 未識別指令
    if len(text) <= 10 and text.isalpha():
        await update.message.reply_text(
            f"❓ 未識別幣種「{text}」\n"
            "支援：BTC / ETH / SOL / BNB / XRP / DOGE / ADA / AVAX / DOT / LINK\n"
            "查詢格式：直接輸入幣種名稱，例如 BTC\n"
            "雙向分析：輸入「BTC分析」"
        )

# ── 主掃描循環（入場訊號）─────────────────────────────────────────
async def signal_scan_loop(bot: Bot):
    last_signals: dict = {}
    while True:
        try:
            now_ts = time.time()
            for symbol in SYMBOLS:
                try:
                    result = analyze_symbol(symbol)
                    if result.get("error"):
                        logger.warning(f"{symbol} 分析錯誤: {result['error']}")
                        continue
                    signal = find_entry_signal(result)
                    if signal:
                        sig_key  = f"{symbol}_{signal['direction']}"
                        last_time = last_signals.get(sig_key, 0)
                        if now_ts - last_time > 600:
                            msg = format_signal_message(signal)
                            await send_msg(bot, msg)
                            last_signals[sig_key] = now_ts
                            logger.info(f"發送入場訊號: {symbol} {signal['direction']} {signal['trade_type']}")
                except Exception as e:
                    logger.error(f"{symbol} 掃描錯誤: {e}")
        except Exception as e:
            logger.error(f"掃描循環錯誤: {e}")
        await asyncio.sleep(SCAN_INTERVAL)

# ── 定時分析排程器 ─────────────────────────────────────────────────
async def scheduled_analysis_loop(bot: Bot):
    """每分鐘檢查一次，到達指定時間就發送分析"""
    last_sent: dict = {}
    while True:
        try:
            now = datetime.now(HKT)
            hm  = now.strftime("%H:%M")
            if hm in SCHEDULED_TIMES:
                # 每個時間點只發一次（用日期+時間作 key）
                key = now.strftime("%Y-%m-%d") + hm
                if key not in last_sent:
                    await run_scheduled_analysis(bot)
                    last_sent[key] = True
                    # 清理舊 key（只保留今日）
                    today = now.strftime("%Y-%m-%d")
                    last_sent = {k: v for k, v in last_sent.items() if k.startswith(today)}
        except Exception as e:
            logger.error(f"定時分析排程錯誤: {e}")
        await asyncio.sleep(60)  # 每分鐘檢查一次

# ── 主程式 ─────────────────────────────────────────────────────────
async def main():
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("缺少 TELEGRAM_BOT_TOKEN 或 TELEGRAM_CHAT_ID 環境變數")
        return

    # 建立 Application（支援 MessageHandler）
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    bot = app.bot

    await send_msg(bot,
        "✅ ICT/SMC 交易信號機械人 v5.0 已啟動\n\n"
        "① 自動入場訊號：BTC / ETH / SOL（即時掃描）\n"
        "② 定時盤勢分析：BTC 每日 08:00 / 12:00 / 17:00 / 20:30 / 23:30 HKT\n"
        "③ 按需查詢：輸入 BTC / ETH / SOL 等幣種名稱\n"
        "④ 雙向情景：輸入「BTC分析」取得即時雙向情景\n\n"
        "🌐 數據：data-api.binance.vision"
    )

    logger.info("機械人 v5.0 已啟動")

    # 同時運行三個異步任務
    await asyncio.gather(
        app.run_polling(drop_pending_updates=True),
        signal_scan_loop(bot),
        scheduled_analysis_loop(bot),
    )

if __name__ == "__main__":
    asyncio.run(main())
