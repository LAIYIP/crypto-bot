#!/usr/bin/env python3
"""
ICT/SMC 加密貨幣交易信號機械人 v4.0
====================================
框架（按用戶 PDF）：
  1H 定方向（BSL/SSL 流動性判斷）
  → 15M 識別關鍵位（OB/FVG/IFVG/SNR/EQH/EQL/OTE/PDH/PDL/PWH/PWL/日開/週開）
  → 層級陷阱檢查（15M 阻力上方有無 1H 未測試關鍵位？）
  → 3M MSS 確認（實體收線突破結構）
  → Displacement + FIB OTE 入場（FVG/OB 在 0.618-0.786 區域）
  → 順勢全倉（50 USDT）/ 逆勢半倉（25 USDT）
  → RR ≥ 1:2 才發信號
  → SL 放在 15M Swing High/Low 外

數據量：1H 500根（21天）/ 15M 300根（75小時）/ 3M 200根（10小時）
數據源：Binance 公開 REST API（無需認證）
"""

import os
import asyncio
import logging
import requests
import time
import pandas as pd
import numpy as np
from telegram import Bot
from telegram.ext import Application
from datetime import datetime, timezone, timedelta
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── 設定 ──────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
SCAN_INTERVAL = 60
HOURLY_REPORT_INTERVAL = 3600
MIN_RR = 2.0
RISK_FULL = 50.0
RISK_HALF = 25.0
MIN_ZONE_PCT = 0.002       # 關鍵區最小寬度 0.2%
HKT = timezone(timedelta(hours=8))
BINANCE_BASE = "https://api.binance.com"
BINANCE_ALT  = "https://api1.binance.com"

# 訂單計數器（每個幣種+方向獨立，重啟重置）
order_counters: dict = defaultdict(int)

# ── Binance API ───────────────────────────────────────
def get_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame | None:
    for base in [BINANCE_BASE, BINANCE_ALT]:
        try:
            r = requests.get(
                f"{base}/api/v3/klines",
                params={"symbol": symbol, "interval": interval, "limit": limit},
                timeout=10
            )
            if r.status_code == 200:
                df = pd.DataFrame(r.json(), columns=[
                    'ts','open','high','low','close','volume',
                    'cts','qv','tr','tbb','tbq','ign'])
                df = df[['ts','open','high','low','close','volume']].copy()
                df['ts'] = pd.to_datetime(df['ts'], unit='ms')
                for col in ['open','high','low','close','volume']:
                    df[col] = df[col].astype(float)
                return df
        except Exception as e:
            logger.warning(f"Binance {base} {symbol} {interval}: {e}")
    return None

def get_daily_weekly_levels(symbol: str) -> dict:
    """取得前日/前週高低點、今日/本週開盤"""
    levels = {}
    try:
        r = requests.get(f"{BINANCE_BASE}/api/v3/klines",
            params={"symbol": symbol, "interval": "1d", "limit": 3}, timeout=10)
        if r.status_code == 200:
            d = r.json()
            if len(d) >= 2:
                levels['PDH'] = float(d[-2][2])
                levels['PDL'] = float(d[-2][3])
                levels['DO']  = float(d[-1][1])   # 今日開盤
        r2 = requests.get(f"{BINANCE_BASE}/api/v3/klines",
            params={"symbol": symbol, "interval": "1w", "limit": 3}, timeout=10)
        if r2.status_code == 200:
            w = r2.json()
            if len(w) >= 2:
                levels['PWH'] = float(w[-2][2])
                levels['PWL'] = float(w[-2][3])
                levels['WO']  = float(w[-1][1])   # 本週開盤
    except Exception as e:
        logger.warning(f"get_daily_weekly_levels {symbol}: {e}")
    return levels

# ── 工具函數 ──────────────────────────────────────────
def fmt(price: float, symbol: str) -> str:
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

# ── 市場結構分析 ──────────────────────────────────────
def find_swing_points(df: pd.DataFrame, n: int = 5) -> tuple:
    """識別擺動高低點"""
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
    """雙層擺動點（n=3 短期 + n=8 中期）合併去重"""
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
    """判斷市場結構：bullish / bearish / ranging"""
    if df is None or len(df) < lookback:
        return "ranging"
    sub = df.iloc[-lookback:]
    highs, lows = find_swing_points(sub, n=3)
    if len(highs) < 2 or len(lows) < 2:
        return "ranging"
    rh = sorted(highs, key=lambda x: x[0])[-2:]
    rl = sorted(lows, key=lambda x: x[0])[-2:]
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
    """識別 BSL（上方流動性）和 SSL（下方流動性）"""
    highs, lows = find_swing_dual(df)
    price = float(df.iloc[-1]['close'])
    bsl = min([h[1] for h in highs if h[1] > price], default=None)
    ssl = max([l[1] for l in lows if l[1] < price], default=None)
    return bsl, ssl

def detect_3m_mss(df_3m: pd.DataFrame, direction: str) -> dict | None:
    """
    偵測 3M MSS（Market Structure Shift）
    direction: 'bullish' = 找看漲 MSS / 'bearish' = 找看跌 MSS
    條件：實體收線突破（不只是影線）
    """
    if df_3m is None or len(df_3m) < 20:
        return None
    recent = df_3m.iloc[-20:].copy().reset_index(drop=True)
    highs, lows = find_swing_points(recent, n=3)
    if not highs or not lows:
        return None
    last = recent.iloc[-1]
    if direction == "bullish":
        if not highs:
            return None
        last_swing_high = max(highs, key=lambda x: x[0])[1]
        # 實體收線突破（收盤價 > 前擺動高點）
        if float(last['close']) > float(last['open']) and float(last['close']) > last_swing_high:
            return {"type": "bullish_mss", "break_price": last_swing_high,
                    "candle_close": float(last['close']), "direction": "bullish"}
    elif direction == "bearish":
        if not lows:
            return None
        last_swing_low = max(lows, key=lambda x: x[0])[1]
        if float(last['close']) < float(last['open']) and float(last['close']) < last_swing_low:
            return {"type": "bearish_mss", "break_price": last_swing_low,
                    "candle_close": float(last['close']), "direction": "bearish"}
    return None

def find_displacement_fvg(df: pd.DataFrame, direction: str, lookback: int = 10) -> dict | None:
    """
    在最近 lookback 根蠟燭中尋找 Displacement（大蠟燭位移）形成的 FVG
    這是 MSS 後的精確入場位
    """
    if df is None or len(df) < 3:
        return None
    recent = df.iloc[-lookback:].copy().reset_index(drop=True)
    best_fvg = None
    best_size = 0
    for i in range(1, len(recent) - 1):
        k1 = recent.iloc[i-1]
        k2 = recent.iloc[i]
        k3 = recent.iloc[i+1]
        body_k2 = abs(float(k2['close']) - float(k2['open']))
        range_k2 = float(k2['high']) - float(k2['low'])
        # 確認 k2 是大蠟燭（位移）：實體 > 50% 蠟燭範圍
        if range_k2 == 0 or body_k2 / range_k2 < 0.5:
            continue
        if direction == "bullish":
            # 看漲 FVG：k1 High < k3 Low
            gap_low = float(k1['high'])
            gap_high = float(k3['low'])
            if gap_high > gap_low and float(k2['close']) > float(k2['open']):
                size = gap_high - gap_low
                if size > best_size:
                    best_size = size
                    best_fvg = {"type": "bullish_disp_fvg",
                                "low": gap_low, "high": gap_high,
                                "mid": (gap_low + gap_high) / 2,
                                "label": "位移 FVG（看漲）"}
        elif direction == "bearish":
            # 看跌 FVG：k1 Low > k3 High
            gap_high = float(k1['low'])
            gap_low = float(k3['high'])
            if gap_high > gap_low and float(k2['close']) < float(k2['open']):
                size = gap_high - gap_low
                if size > best_size:
                    best_size = size
                    best_fvg = {"type": "bearish_disp_fvg",
                                "low": gap_low, "high": gap_high,
                                "mid": (gap_low + gap_high) / 2,
                                "label": "位移 FVG（看跌）"}
    return best_fvg

# ── 關鍵區識別 ────────────────────────────────────────
def find_order_blocks(df: pd.DataFrame) -> list:
    """
    識別 OB（Order Block）
    看跌 OB：上升趨勢中最後一根陽燭，之後有大陰燭突破前低
    看漲 OB：下跌趨勢中最後一根陰燭，之後有大陽燭突破前高
    """
    obs = []
    if df is None or len(df) < 10:
        return obs
    lc = float(df.iloc[-1]['close'])
    for i in range(3, len(df) - 3):
        c = df.iloc[i]
        body = abs(float(c['close']) - float(c['open']))
        if body == 0:
            continue
        width = float(c['high']) - float(c['low'])
        if width / lc < MIN_ZONE_PCT:
            continue
        # 看跌 OB：陽燭，之後有大陰燭
        if float(c['close']) > float(c['open']):
            prev_low = float(df.iloc[max(0,i-5):i]['low'].min()) if i > 0 else float('inf')
            after = df.iloc[i+1:min(i+4, len(df))]
            if len(after) >= 1 and float(after['low'].min()) < prev_low:
                obs.append({'type': 'OB', 'direction': 'bearish',
                            'high': float(c['high']), 'low': float(c['low']),
                            'mid': (float(c['high']) + float(c['low'])) / 2,
                            'label': '看跌 OB（供應區）', 'strength': 'strong'})
        # 看漲 OB：陰燭，之後有大陽燭
        elif float(c['close']) < float(c['open']):
            prev_high = float(df.iloc[max(0,i-5):i]['high'].max()) if i > 0 else 0
            after = df.iloc[i+1:min(i+4, len(df))]
            if len(after) >= 1 and float(after['high'].max()) > prev_high:
                obs.append({'type': 'OB', 'direction': 'bullish',
                            'high': float(c['high']), 'low': float(c['low']),
                            'mid': (float(c['high']) + float(c['low'])) / 2,
                            'label': '看漲 OB（需求區）', 'strength': 'strong'})
    return obs

def find_fvg(df: pd.DataFrame) -> list:
    """識別 FVG（Fair Value Gap）"""
    fvgs = []
    if df is None or len(df) < 3:
        return fvgs
    lc = float(df.iloc[-1]['close'])
    for i in range(1, len(df) - 1):
        k1, k3 = df.iloc[i-1], df.iloc[i+1]
        # 看漲 FVG
        gap_bull = float(k3['low']) - float(k1['high'])
        if gap_bull > 0 and gap_bull / lc >= MIN_ZONE_PCT:
            fvgs.append({'type': 'FVG', 'direction': 'bullish',
                         'high': float(k3['low']), 'low': float(k1['high']),
                         'mid': (float(k3['low']) + float(k1['high'])) / 2,
                         'label': '看漲 FVG（需求缺口）', 'strength': 'medium', 'bar_idx': i})
        # 看跌 FVG
        gap_bear = float(k1['low']) - float(k3['high'])
        if gap_bear > 0 and gap_bear / lc >= MIN_ZONE_PCT:
            fvgs.append({'type': 'FVG', 'direction': 'bearish',
                         'high': float(k1['low']), 'low': float(k3['high']),
                         'mid': (float(k1['low']) + float(k3['high'])) / 2,
                         'label': '看跌 FVG（供應缺口）', 'strength': 'medium', 'bar_idx': i})
    return fvgs

def find_ifvg(df: pd.DataFrame, fvg_list: list) -> list:
    """
    識別 IFVG（Inverse FVG）
    FVG 形成後，價格進入該區域並反轉 = IFVG
    """
    ifvgs = []
    for fz in fvg_list:
        bi = fz.get('bar_idx', 0)
        if bi + 2 >= len(df):
            continue
        subsequent = df.iloc[bi+2:]
        for j in range(len(subsequent)):
            row = subsequent.iloc[j]
            entered = float(row['low']) <= fz['high'] and float(row['high']) >= fz['low']
            if entered:
                # 進入後反轉（與 FVG 方向相反的蠟燭）
                if fz['direction'] == 'bullish' and float(row['close']) < float(row['open']):
                    ifvgs.append({'type': 'IFVG', 'direction': 'bearish',
                                  'high': fz['high'], 'low': fz['low'], 'mid': fz['mid'],
                                  'label': '看跌 IFVG（反轉 FVG）', 'strength': 'medium'})
                    break
                elif fz['direction'] == 'bearish' and float(row['close']) > float(row['open']):
                    ifvgs.append({'type': 'IFVG', 'direction': 'bullish',
                                  'high': fz['high'], 'low': fz['low'], 'mid': fz['mid'],
                                  'label': '看漲 IFVG（反轉 FVG）', 'strength': 'medium'})
                    break
    return ifvgs

def find_eqh_eql(df: pd.DataFrame, tolerance: float = 0.001) -> tuple:
    """識別 EQH（Equal Highs）和 EQL（Equal Lows）流動性池"""
    eqh_list, eql_list = [], []
    highs, lows = find_swing_dual(df)
    for i in range(len(highs)):
        for j in range(i+1, len(highs)):
            if abs(highs[i][1] - highs[j][1]) / highs[j][1] <= tolerance:
                price = (highs[i][1] + highs[j][1]) / 2
                eqh_list.append({'price': price, 'label': 'EQH（上方流動性）'})
    for i in range(len(lows)):
        for j in range(i+1, len(lows)):
            if abs(lows[i][1] - lows[j][1]) / lows[j][1] <= tolerance:
                price = (lows[i][1] + lows[j][1]) / 2
                eql_list.append({'price': price, 'label': 'EQL（下方流動性）'})
    return eqh_list, eql_list

def calc_fib(swing_low: float, swing_high: float, direction: str) -> dict:
    """
    計算 FIB 回撤位
    做空：從高點下跌後，回調到 OTE（0.618-0.786）是做空位
    做多：從低點上升後，回調到 OTE（0.618-0.786）是做多位
    """
    diff = swing_high - swing_low
    if direction == "bearish":
        return {
            "0.0":    swing_high,
            "0.5":    swing_high - diff * 0.5,
            "0.618":  swing_high - diff * 0.618,
            "0.705":  swing_high - diff * 0.705,
            "0.786":  swing_high - diff * 0.786,
            "1.0":    swing_low,
            "swing_high": swing_high, "swing_low": swing_low
        }
    else:
        return {
            "0.0":    swing_low,
            "0.5":    swing_low + diff * 0.5,
            "0.618":  swing_low + diff * 0.618,
            "0.705":  swing_low + diff * 0.705,
            "0.786":  swing_low + diff * 0.786,
            "1.0":    swing_high,
            "swing_high": swing_high, "swing_low": swing_low
        }

# ── 層級陷阱檢查 ──────────────────────────────────────
def check_hierarchy_trap(price: float, direction: str,
                          zones_1h: list, zones_15m: list) -> tuple:
    """
    層級陷阱：15M 關鍵位上方/下方有 1H 未測試關鍵位
    做空時：15M 阻力上方還有 1H 阻力 → 15M 阻力是誘餌
    做多時：15M 支撐下方還有 1H 支撐 → 15M 支撐是誘餌
    返回：(is_trap: bool, reason: str)
    """
    if direction == "bearish":
        res_15m = [z for z in zones_15m if z.get('direction') == 'bearish'
                   and z.get('mid', 0) > price]
        if not res_15m:
            return False, ""
        nearest_15m = min(res_15m, key=lambda z: z.get('mid', float('inf')))
        # 1H 在 15M 阻力上方有未測試關鍵位？
        untested_1h = [z for z in zones_1h
                       if z.get('direction') == 'bearish'
                       and z.get('mid', 0) > nearest_15m.get('mid', 0)]
        if untested_1h:
            return True, f"層級陷阱：15M 阻力上方有 1H 未測試供應區"
    elif direction == "bullish":
        sup_15m = [z for z in zones_15m if z.get('direction') == 'bullish'
                   and z.get('mid', 0) < price]
        if not sup_15m:
            return False, ""
        nearest_15m = max(sup_15m, key=lambda z: z.get('mid', 0))
        untested_1h = [z for z in zones_1h
                       if z.get('direction') == 'bullish'
                       and z.get('mid', 0) < nearest_15m.get('mid', 0)]
        if untested_1h:
            return True, f"層級陷阱：15M 支撐下方有 1H 未測試需求區"
    return False, ""

# ── Confluence 評分 ───────────────────────────────────
def calc_confluence(zone: dict, all_zones: list, fib: dict | None) -> tuple:
    """
    計算關鍵區的 Confluence（匯聚）分數
    越多工具重疊，分數越高
    """
    score = 0
    reasons = []
    zone_mid = zone.get('mid', 0)
    if zone_mid == 0:
        return 0, []

    # 1. 與其他關鍵區重疊（0.3% 內）
    for other in all_zones:
        if other is zone:
            continue
        other_mid = other.get('mid', 0)
        if other_mid and abs(zone_mid - other_mid) / zone_mid < 0.003:
            score += 1
            reasons.append(other.get('label', '其他關鍵區'))

    # 2. 與 FIB OTE 區重疊
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

    # 3. 1H 級別關鍵區加分
    if zone.get('tf') == '1h':
        score += 2
        reasons.append("1H 級別關鍵位")

    return score, reasons

# ── 主要分析函數 ──────────────────────────────────────
def analyze_symbol(symbol: str) -> dict:
    """完整分析一個幣種，返回分析結果"""
    result = {"symbol": symbol, "error": None}

    # 取得數據
    df_1h  = get_klines(symbol, "1h",  500)
    df_15m = get_klines(symbol, "15m", 300)
    df_3m  = get_klines(symbol, "3m",  200)
    levels = get_daily_weekly_levels(symbol)

    if df_1h is None or df_15m is None or df_3m is None:
        result["error"] = "無法取得數據"
        return result

    price = float(df_15m.iloc[-1]['close'])
    result["price"] = price
    result["levels"] = levels

    # ── 1H 方向 + BSL/SSL ──
    struct_1h = get_market_structure(df_1h, lookback=30)
    result["struct_1h"] = struct_1h
    bsl, ssl = get_bsl_ssl(df_1h.iloc[-100:])
    result["bsl"] = bsl
    result["ssl"] = ssl

    # ── 1H FIB（用最近擺動高低點）──
    highs_1h, lows_1h = find_swing_dual(df_1h.iloc[-100:])
    fib_1h = None
    if highs_1h and lows_1h:
        h1 = max(highs_1h, key=lambda x: x[0])[1]
        l1 = min(lows_1h, key=lambda x: x[0])[1]
        fib_1h = calc_fib(l1, h1, struct_1h if struct_1h != "ranging" else "bearish")
    result["fib_1h"] = fib_1h

    # ── 1H 關鍵區（層級陷阱用）──
    obs_1h  = find_order_blocks(df_1h.iloc[-100:])
    fvgs_1h = find_fvg(df_1h.iloc[-100:])
    for z in obs_1h + fvgs_1h:
        z['tf'] = '1h'
    zones_1h = obs_1h + fvgs_1h
    result["zones_1h"] = zones_1h

    # ── 15M 關鍵區 ──
    obs_15m  = find_order_blocks(df_15m)
    fvgs_15m = find_fvg(df_15m)
    ifvgs_15m = find_ifvg(df_15m, fvgs_15m)
    for z in obs_15m + fvgs_15m + ifvgs_15m:
        z['tf'] = '15m'
    zones_15m = obs_15m + fvgs_15m + ifvgs_15m
    result["zones_15m"] = zones_15m

    # EQH/EQL
    eqh_list, eql_list = find_eqh_eql(df_15m)
    result["eqh"] = eqh_list
    result["eql"] = eql_list

    # 15M 擺動點（SL 計算用）
    highs_15m, lows_15m = find_swing_dual(df_15m.iloc[-50:])
    result["highs_15m"] = highs_15m
    result["lows_15m"] = lows_15m

    # 15M FIB（TP 計算用）
    fib_15m = None
    if highs_15m and lows_15m:
        h15 = max(highs_15m, key=lambda x: x[0])[1]
        l15 = min(lows_15m, key=lambda x: x[0])[1]
        fib_15m = calc_fib(l15, h15, struct_1h if struct_1h != "ranging" else "bearish")
    result["fib_15m"] = fib_15m

    # ATR
    result["atr_15m"] = calc_atr(df_15m)

    # ── 3M MSS 偵測 ──
    if struct_1h == "bearish":
        mss_trend   = detect_3m_mss(df_3m, "bearish")   # 順勢做空
        mss_counter = detect_3m_mss(df_3m, "bullish")   # 逆勢做多
    elif struct_1h == "bullish":
        mss_trend   = detect_3m_mss(df_3m, "bullish")   # 順勢做多
        mss_counter = detect_3m_mss(df_3m, "bearish")   # 逆勢做空
    else:
        mss_trend = mss_counter = None

    result["mss_trend"]   = mss_trend
    result["mss_counter"] = mss_counter
    result["df_3m"]  = df_3m
    result["df_15m"] = df_15m

    return result

# ── 入場信號邏輯 ──────────────────────────────────────
def find_entry_signal(result: dict) -> dict | None:
    """根據分析結果尋找入場信號，返回信號字典或 None"""
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
    zones_15m = result.get("zones_15m", [])
    highs_15m = result.get("highs_15m", [])
    lows_15m  = result.get("lows_15m", [])
    eqh       = result.get("eqh", [])
    eql       = result.get("eql", [])
    df_3m     = result.get("df_3m")
    df_15m    = result.get("df_15m")

    if struct_1h == "ranging":
        return None

    def build_signal(direction: str, trade_type: str, mss: dict) -> dict | None:
        """建立信號字典"""
        is_trend = (trade_type == "順勢")
        risk_type = "全倉" if is_trend else "半倉"
        risk_amount = RISK_FULL if is_trend else RISK_HALF

        # 找最近的 15M 關鍵區
        dir_zones = [z for z in zones_15m if z.get('direction') == direction]
        if not dir_zones:
            return None

        # 找價格附近的關鍵區（±1%）
        active_zone = None
        for z in sorted(dir_zones, key=lambda x: abs(x.get('mid', 0) - price)):
            if abs(z.get('mid', 0) - price) / price < 0.01:
                active_zone = z
                break
        if not active_zone:
            active_zone = min(dir_zones, key=lambda z: abs(z.get('mid', 0) - price))

        # 層級陷阱檢查（順勢才檢查）
        if is_trend:
            is_trap, trap_reason = check_hierarchy_trap(price, direction, zones_1h, zones_15m)
            if is_trap:
                return None

        # 逆勢需要 1H 強關鍵位支撐
        if not is_trend:
            strong_1h = [z for z in zones_1h
                         if z.get('direction') == direction
                         and abs(z.get('mid', 0) - price) / price < 0.015]
            key_levels = [v for v in [levels.get('PDH'), levels.get('PDL'),
                                       levels.get('PWH'), levels.get('PWL'),
                                       levels.get('DO'), levels.get('WO')] if v]
            nearby_level = any(abs(price - l) / price < 0.01 for l in key_levels)
            if not strong_1h and not nearby_level:
                return None  # 逆勢沒有 1H 強關鍵位，不入場

        # 尋找 Displacement FVG（MSS 後精確入場位）
        disp_fvg = find_displacement_fvg(df_3m, direction, lookback=10)

        # 計算入場位
        entry_price = price
        entry_label = "市價入場"
        if disp_fvg:
            entry_price = disp_fvg["mid"]
            entry_label = f"FVG 掛單 {fmt(disp_fvg['low'], symbol)} - {fmt(disp_fvg['high'], symbol)}"
        elif fib_1h:
            ote_705 = fib_1h.get("0.705", 0)
            if ote_705 and abs(price - ote_705) / price < 0.015:
                entry_price = ote_705
                entry_label = f"FIB OTE 0.705 掛單"

        # SL（15M Swing High/Low 外）
        min_sl_dist = price * 0.005  # 最少 0.5%
        if direction == "bearish":
            above_highs = [h[1] for h in highs_15m if h[1] > price]
            sl = min(above_highs) * 1.001 if above_highs else price * 1.005
        else:
            below_lows = [l[1] for l in lows_15m if l[1] < price]
            sl = max(below_lows) * 0.999 if below_lows else price * 0.995

        sl_dist = abs(entry_price - sl)
        # 確保 SL 距離足夠（最少 0.5% 或 1×ATR）
        min_dist = max(min_sl_dist, atr_15m * 1.0)
        if sl_dist < min_dist:
            sl = (entry_price + min_dist) if direction == "bearish" else (entry_price - min_dist)
            sl_dist = min_dist

        # TP1（15M FIB 0.5）
        tp1 = tp2 = None
        tp_label = ""
        if fib_15m:
            tp1 = fib_15m.get("0.5")
        # TP2（流動性池）
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

        # RR 過濾
        if rr < MIN_RR:
            return None

        # Confluence 評分
        all_zones = zones_15m + zones_1h
        conf_score, conf_reasons = calc_confluence(active_zone, all_zones, fib_1h)

        # 逆勢需要 Confluence ≥ 2
        if not is_trend and conf_score < 2:
            return None

        # 信號類型
        if is_trend:
            signal_type = "🏄 衝浪者 A（淺回調順勢）" if not disp_fvg else "🏄 衝浪者 B（深回調順勢）"
        else:
            signal_type = "🎯 狙擊手 A（激進逆勢）" if disp_fvg else "🎯 狙擊手 B（穩健逆勢）"

        return {
            "symbol": symbol, "price": price,
            "direction": direction,
            "signal_type": signal_type,
            "trade_type": trade_type,
            "risk_type": risk_type,
            "risk_amount": risk_amount,
            "entry_price": entry_price,
            "entry_label": entry_label,
            "sl": sl, "sl_dist": sl_dist,
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

    # 順勢信號優先
    if mss_trend:
        direction = mss_trend["direction"]
        sig = build_signal(direction, "順勢", mss_trend)
        if sig:
            return sig

    # 逆勢信號
    if mss_counter:
        direction = mss_counter["direction"]
        sig = build_signal(direction, "逆勢", mss_counter)
        if sig:
            return sig

    return None

# ── 訊息格式化 ────────────────────────────────────────
def format_signal_message(sig: dict) -> str:
    symbol     = sig["symbol"]
    sym_short  = symbol.replace("USDT", "/USDT")
    direction  = sig["direction"]
    dir_emoji  = "🔴 做空 (Short)" if direction == "bearish" else "🟢 做多 (Long)"
    dir_label  = "S" if direction == "bearish" else "L"
    struct     = sig["struct_1h"]
    struct_emoji = "⬇️ 看跌" if struct == "bearish" else "⬆️ 看漲" if struct == "bullish" else "↔️ 橫盤"
    trade_type = sig["trade_type"]
    risk_type  = sig["risk_type"]
    risk_amount = sig["risk_amount"]
    entry      = sig["entry_price"]
    sl         = sig["sl"]
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

    # Confluence 評級
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

    # FIB OTE 資訊
    if fib_1h:
        ote_low  = min(fib_1h.get("0.618", 0), fib_1h.get("0.786", 0))
        ote_high = max(fib_1h.get("0.618", 0), fib_1h.get("0.786", 0))
        if ote_low > 0:
            msg += f"📐 1H FIB OTE 區: {fmt(ote_low, symbol)} - {fmt(ote_high, symbol)}\n"
            msg += f"   0.705（最佳入場）: {fmt(fib_1h.get('0.705', 0), symbol)}\n\n"

    msg += f"📈 交易方向: {dir_emoji}\n\n"

    # 入場方式
    msg += f"💵 入場方式: {entry_label}\n"
    if disp_fvg:
        msg += f"   30% 市價入場: {fmt(sig['price'], symbol)}\n"
        msg += f"   70% FVG 掛單: {fmt(disp_fvg['mid'], symbol)}\n"
    else:
        msg += f"   入場價格: {fmt(entry, symbol)}\n"
    msg += "\n"

    msg += f"🛑 止損 (SL): {fmt(sl, symbol)}\n"
    msg += f"   └ 15M Swing {'High' if direction == 'bearish' else 'Low'} 外\n\n"

    msg += f"🎯 止盈 TP1: {fmt(tp1, symbol)}\n"
    msg += f"   └ {tp_label}\n"
    if tp2 and tp2 != tp1:
        msg += f"🎯 止盈 TP2: {fmt(tp2, symbol)}\n"
        msg += f"   └ 延伸目標（流動性）\n"
    msg += f"📊 預計 RR: 1:{rr:.1f}\n\n"

    # 流動性參考
    if bsl or ssl:
        msg += "💧 流動性參考:\n"
        if bsl:
            msg += f"   BSL（上方）: {fmt(bsl, symbol)}\n"
        if ssl:
            msg += f"   SSL（下方）: {fmt(ssl, symbol)}\n"
        msg += "\n"

    # 重要水平
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

def format_hourly_report(results: list) -> str:
    """格式化每小時市場快報"""
    now = datetime.now(HKT)
    msg  = f"🕐 每小時市場快報 [{now.strftime('%H:%M')} HKT]\n"
    msg += "━━━━━━━━━━━━━━━━━━\n\n"

    for result in results:
        if result.get("error"):
            continue
        symbol     = result["symbol"]
        sym_short  = symbol.replace("USDT", "/USDT")
        price      = result["price"]
        struct_1h  = result.get("struct_1h", "ranging")
        struct_emoji = "⬇️ 看跌" if struct_1h == "bearish" else "⬆️ 看漲" if struct_1h == "bullish" else "↔️ 橫盤"
        levels     = result.get("levels", {})
        bsl        = result.get("bsl")
        ssl        = result.get("ssl")
        fib_1h     = result.get("fib_1h")
        zones_15m  = result.get("zones_15m", [])

        msg += f"📌 {sym_short}  💲{fmt(price, symbol)}\n"
        msg += f"   1H {struct_emoji}\n"

        # BSL/SSL
        if bsl:
            msg += f"   🔼 BSL（上方流動性）: {fmt(bsl, symbol)}\n"
        if ssl:
            msg += f"   🔽 SSL（下方流動性）: {fmt(ssl, symbol)}\n"

        # FIB OTE
        if fib_1h:
            ote_low  = min(fib_1h.get("0.618", 0), fib_1h.get("0.786", 0))
            ote_high = max(fib_1h.get("0.618", 0), fib_1h.get("0.786", 0))
            if ote_low > 0:
                msg += f"   📐 OTE 區: {fmt(ote_low, symbol)} - {fmt(ote_high, symbol)}\n"

        # 重要水平
        if levels.get('DO'):
            msg += f"   📅 今日開盤: {fmt(levels['DO'], symbol)}\n"
        if levels.get('WO'):
            msg += f"   📅 本週開盤: {fmt(levels['WO'], symbol)}\n"
        if levels.get('PDH'):
            msg += f"   🔴 PDH: {fmt(levels['PDH'], symbol)}  |  PDL: {fmt(levels['PDL'], symbol)}\n"
        if levels.get('PWH'):
            msg += f"   🟣 PWH: {fmt(levels['PWH'], symbol)}  |  PWL: {fmt(levels['PWL'], symbol)}\n"

        # 最近 15M 阻力/支撐
        bear_zones = [z for z in zones_15m if z.get('direction') == 'bearish' and z.get('mid', 0) > price]
        bull_zones = [z for z in zones_15m if z.get('direction') == 'bullish' and z.get('mid', 0) < price]
        if bear_zones:
            nearest = min(bear_zones, key=lambda z: z.get('mid', float('inf')))
            msg += f"   🔴 阻力: {fmt(nearest.get('low', 0), symbol)} - {fmt(nearest.get('high', 0), symbol)}  [{nearest.get('label', '')}]\n"
        if bull_zones:
            nearest = max(bull_zones, key=lambda z: z.get('mid', 0))
            msg += f"   🟢 支撐: {fmt(nearest.get('low', 0), symbol)} - {fmt(nearest.get('high', 0), symbol)}  [{nearest.get('label', '')}]\n"

        msg += "\n"

    return msg.rstrip()

# ── Telegram 發送 ─────────────────────────────────────
async def send_msg(bot: Bot, text: str):
    try:
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text, parse_mode=None)
    except Exception as e:
        logger.error(f"Telegram 發送失敗: {e}")

# ── 主掃描循環 ────────────────────────────────────────
async def main():
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("缺少 TELEGRAM_BOT_TOKEN 或 TELEGRAM_CHAT_ID 環境變數")
        return

    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    last_hourly = 0
    last_signals: dict = {}  # 防止重複發送（同一幣種+方向 10 分鐘內不重複）

    # 啟動訊息
    await send_msg(bot,
        "✅ ICT/SMC 交易信號機械人 v4.0 已啟動\n\n"
        "📊 監控: BTC / ETH / SOL\n"
        "🎯 框架: 1H 定方向 → 15M 關鍵位 → 3M MSS → FIB OTE 入場\n"
        "🆕 新增: BSL/SSL / 層級陷阱 / Displacement FVG / 日開(DO)/週開(WO)\n"
        "📐 SL: 15M Swing High/Low 外（最少 1×ATR）\n"
        "💰 順勢全倉（50 USDT）/ 逆勢半倉（25 USDT）\n"
        "📊 RR ≥ 1:2 才發信號\n"
        "🌐 數據: Binance 公開 API（1H 500根 / 15M 300根 / 3M 200根）"
    )

    logger.info("機械人 v4.0 已啟動，開始掃描...")

    while True:
        try:
            now_ts = time.time()
            results = []

            for symbol in SYMBOLS:
                try:
                    result = analyze_symbol(symbol)
                    results.append(result)

                    if result.get("error"):
                        logger.warning(f"{symbol} 分析錯誤: {result['error']}")
                        continue

                    signal = find_entry_signal(result)
                    if signal:
                        sig_key = f"{symbol}_{signal['direction']}"
                        last_time = last_signals.get(sig_key, 0)
                        if now_ts - last_time > 600:  # 10 分鐘防重複
                            msg = format_signal_message(signal)
                            await send_msg(bot, msg)
                            last_signals[sig_key] = now_ts
                            logger.info(f"發送信號: {symbol} {signal['direction']} {signal['trade_type']}")

                except Exception as e:
                    logger.error(f"{symbol} 掃描錯誤: {e}")

            # 每小時快報
            if now_ts - last_hourly >= HOURLY_REPORT_INTERVAL:
                if results:
                    hourly_msg = format_hourly_report(results)
                    await send_msg(bot, hourly_msg)
                    last_hourly = now_ts
                    logger.info("已發送每小時快報")

        except Exception as e:
            logger.error(f"主循環錯誤: {e}")

        await asyncio.sleep(SCAN_INTERVAL)

if __name__ == "__main__":
    asyncio.run(main())
