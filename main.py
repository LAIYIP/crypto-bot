#!/usr/bin/env python3
"""
ICT/SMC 加密貨幣交易信號機械人 v4.3
====================================
框架（按用戶 PDF）：
  1H 定方向（BSL/SSL 流動性判斷）
  → 15M 識別關鍵位（OB/FVG/IFVG/SNR/EQH/EQL/OTE/PDH/PDL/PWH/PWL/日開/週開）
  → 層級陷阱檢查（15M 阻力上方有無 1H 未測試關鍵位？）
  → 3M MSS 確認（實體收線突破結構）
  → Displacement + FIB OTE 入場（FVG/OB 在 0.618-0.786 區域）
  → 順勢全倉（50 USDT）/ 逆勢半倉（25 USDT）
  → RR ≥ 1:2 才發信號
  → SL 放在「關鍵區接觸後反轉確認」的關鍵位外側

數據量：1H 500根（21天）/ 4H 200根（33天）/ 15M 300根（75小時）/ 3M 200根（10小時）
數據源：Binance data-api.binance.vision（無地區限制）+ api.binance.us 備用

v4.2 修復：
  ① SL 邏輯：先找「最近被接觸過的關鍵區」，SL 放在該區外側（非單純 Swing High/Low）
     - 做多：找價格下方最近一個曾被觸及的 OB/FVG/SNR，SL 放在該區 Low 下方 0.1%
     - 做空：找價格上方最近一個曾被觸及的 OB/FVG/SNR，SL 放在該區 High 上方 0.1%
     - 若找不到，退回 15M Swing High/Low 外
     - 加入 4H OB 輔助識別（4H OB 作為更大的 SL 錨點）
  ② 每小時快報重構：最多 6 個關鍵位
     - 最外圍 2 個（1H 最遠的上方 + 下方，定方向用）
     - 最近 2 個上方阻力 + 最近 2 個下方支撐
     - ICT/SMC 優先級：PWH/PWL/PDH/PDL/WO/DO > BSL/SSL > 1H OB/FVG > 15M OB/FVG
     - FIB 只作輔助：若最近關鍵位附近（±0.5%）有 FIB 重疊，在後方加標 [+FIB 0.618] 等
"""

import os
import asyncio
import logging
import requests
import time
import pandas as pd
import numpy as np
from telegram import Bot
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

BINANCE_ENDPOINTS = [
    "https://data-api.binance.vision",
    "https://api.binance.us",
]

order_counters: dict = defaultdict(int)

# ── Binance API ───────────────────────────────────────
def get_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame | None:
    """從 Binance 取得 K 線數據，自動嘗試多個端點"""
    for base in BINANCE_ENDPOINTS:
        try:
            r = requests.get(
                f"{base}/api/v3/klines",
                params={"symbol": symbol, "interval": interval, "limit": limit},
                timeout=15
            )
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, dict):
                    logger.warning(f"{base} {symbol} {interval}: API 錯誤 {data.get('msg', data)}")
                    continue
                if not data:
                    continue
                df = pd.DataFrame(data, columns=[
                    'ts','open','high','low','close','volume',
                    'cts','qv','tr','tbb','tbq','ign'])
                df = df[['ts','open','high','low','close','volume']].copy()
                df['ts'] = pd.to_datetime(df['ts'], unit='ms')
                for col in ['open','high','low','close','volume']:
                    df[col] = df[col].astype(float)
                return df
            else:
                logger.warning(f"{base} {symbol} {interval}: HTTP {r.status_code}")
        except Exception as e:
            logger.warning(f"{base} {symbol} {interval}: {e}")
    logger.error(f"所有端點均失敗: {symbol} {interval}")
    return None

def get_daily_weekly_levels(symbol: str) -> dict:
    """取得前日/前週高低點、今日/本週開盤"""
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
            logger.warning(f"get_daily {base} {symbol}: {e}")

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
            logger.warning(f"get_weekly {base} {symbol}: {e}")
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
    highs, lows = find_swing_dual(df)
    price = float(df.iloc[-1]['close'])
    bsl = min([h[1] for h in highs if h[1] > price], default=None)
    ssl = max([l[1] for l in lows if l[1] < price], default=None)
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
        if not highs:
            return None
        last_swing_high = max(highs, key=lambda x: x[0])[1]
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
        if range_k2 == 0 or body_k2 / range_k2 < 0.5:
            continue
        if direction == "bullish":
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
        if float(c['close']) > float(c['open']):
            prev_low = float(df.iloc[max(0,i-5):i]['low'].min()) if i > 0 else float('inf')
            after = df.iloc[i+1:min(i+4, len(df))]
            if len(after) >= 1 and float(after['low'].min()) < prev_low:
                obs.append({'type': 'OB', 'direction': 'bearish',
                            'high': float(c['high']), 'low': float(c['low']),
                            'mid': (float(c['high']) + float(c['low'])) / 2,
                            'label': 'OB−（供應區）', 'strength': 'strong'})
        elif float(c['close']) < float(c['open']):
            prev_high = float(df.iloc[max(0,i-5):i]['high'].max()) if i > 0 else 0
            after = df.iloc[i+1:min(i+4, len(df))]
            if len(after) >= 1 and float(after['high'].max()) > prev_high:
                obs.append({'type': 'OB', 'direction': 'bullish',
                            'high': float(c['high']), 'low': float(c['low']),
                            'mid': (float(c['high']) + float(c['low'])) / 2,
                            'label': 'OB+（需求區）', 'strength': 'strong'})
    return obs

def find_fvg(df: pd.DataFrame) -> list:
    fvgs = []
    if df is None or len(df) < 3:
        return fvgs
    lc = float(df.iloc[-1]['close'])
    for i in range(1, len(df) - 1):
        k1, k3 = df.iloc[i-1], df.iloc[i+1]
        gap_bull = float(k3['low']) - float(k1['high'])
        if gap_bull > 0 and gap_bull / lc >= MIN_ZONE_PCT:
            fvgs.append({'type': 'FVG', 'direction': 'bullish',
                         'high': float(k3['low']), 'low': float(k1['high']),
                         'mid': (float(k3['low']) + float(k1['high'])) / 2,
                         'label': 'FVG+（需求缺口）', 'strength': 'medium', 'bar_idx': i})
        gap_bear = float(k1['low']) - float(k3['high'])
        if gap_bear > 0 and gap_bear / lc >= MIN_ZONE_PCT:
            fvgs.append({'type': 'FVG', 'direction': 'bearish',
                         'high': float(k1['low']), 'low': float(k3['high']),
                         'mid': (float(k1['low']) + float(k3['high'])) / 2,
                         'label': 'FVG−（供應缺口）', 'strength': 'medium', 'bar_idx': i})
    return fvgs

def find_ifvg(df: pd.DataFrame, fvg_list: list) -> list:
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
                if fz['direction'] == 'bullish' and float(row['close']) < float(row['open']):
                    ifvgs.append({'type': 'IFVG', 'direction': 'bearish',
                                  'high': fz['high'], 'low': fz['low'], 'mid': fz['mid'],
                                  'label': 'IFVG−（反轉供應）', 'strength': 'medium'})
                    break
                elif fz['direction'] == 'bearish' and float(row['close']) > float(row['open']):
                    ifvgs.append({'type': 'IFVG', 'direction': 'bullish',
                                  'high': fz['high'], 'low': fz['low'], 'mid': fz['mid'],
                                  'label': 'IFVG+（反轉需求）', 'strength': 'medium'})
                    break
    return ifvgs

def find_eqh_eql(df: pd.DataFrame, tolerance: float = 0.001) -> tuple:
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

# ── SL 邏輯（v4.3 按 PDF 框架）────────────────────────
def find_sl_anchor_zone(price: float, direction: str,
                         zones_15m: list, zones_4h: list,
                         highs_15m: list, lows_15m: list,
                         atr: float) -> tuple:
    """
    v4.3 SL 邏輯（按 PDF 框架）：
    SL = 入場關鍵區的結構性止損，放在「15M 最近 Swing High/Low 外側」
    緩衝 = ATR × 0.5（避免被掃）
    最小距離 = max(ATR × 1.0, 0.5%)

    優先級：
      1) 入場 OB/FVG 的 Low/High 外側 + ATR×0.5
      2) 入場區下方（做多）/ 上方（做空）最近 15M Swing Low/High + ATR×0.5
      3) 退回：ATR 最小距離
    """
    atr_buf  = atr * 0.5
    min_dist = max(atr * 1.0, price * 0.005)

    if direction == "bullish":
        # 1) 入場 OB/FVG 的 Low 外側
        entry_zones = sorted(
            [z for z in zones_15m
             if z.get('direction') == 'bullish'
             and z.get('low', 0) < price],
            key=lambda z: z.get('low', 0), reverse=True
        )
        if entry_zones:
            anchor_low = entry_zones[0]['low']
            sl = anchor_low - atr_buf
            if (price - sl) >= min_dist:
                lbl = entry_zones[0].get('label', 'OB')
                return sl, f"15M {lbl} Low 外 (ATR×0.5 緩衝)"

        # 2) 入場區下方最近 15M Swing Low 外側
        below_lows = sorted(
            [l[1] for l in lows_15m if l[1] < price],
            reverse=True
        )
        if below_lows:
            sl = below_lows[0] - atr_buf
            if (price - sl) >= min_dist:
                return sl, f"15M Swing Low {fmt(below_lows[0], '')} 外 (ATR×0.5 緩衝)"

        # 3) 退回：ATR 最小距離
        return price - min_dist, "ATR 最小止損"

    else:  # bearish
        # 1) 入場 OB/FVG 的 High 外側
        entry_zones = sorted(
            [z for z in zones_15m
             if z.get('direction') == 'bearish'
             and z.get('high', 0) > price],
            key=lambda z: z.get('high', 0)
        )
        if entry_zones:
            anchor_high = entry_zones[0]['high']
            sl = anchor_high + atr_buf
            if (sl - price) >= min_dist:
                lbl = entry_zones[0].get('label', 'OB')
                return sl, f"15M {lbl} High 外 (ATR×0.5 緩衝)"

        # 2) 入場區上方最近 15M Swing High 外側
        above_highs = sorted(
            [h[1] for h in highs_15m if h[1] > price]
        )
        if above_highs:
            sl = above_highs[0] + atr_buf
            if (sl - price) >= min_dist:
                return sl, f"15M Swing High {fmt(above_highs[0], '')} 外 (ATR×0.5 緩衝)"

        # 3) 退回：ATR 最小距離
        return price + min_dist, "ATR 最小止損"

# ── 層級陷阱檢查 ──────────────────────────────────────
def check_hierarchy_trap(price: float, direction: str,
                          zones_1h: list, zones_15m: list) -> tuple:
    if direction == "bearish":
        res_15m = [z for z in zones_15m if z.get('direction') == 'bearish'
                   and z.get('mid', 0) > price]
        if not res_15m:
            return False, ""
        nearest_15m = min(res_15m, key=lambda z: z.get('mid', float('inf')))
        untested_1h = [z for z in zones_1h
                       if z.get('direction') == 'bearish'
                       and z.get('mid', 0) > nearest_15m.get('mid', 0)]
        if untested_1h:
            return True, "層級陷阱：15M 阻力上方有 1H 未測試供應區"
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
            return True, "層級陷阱：15M 支撐下方有 1H 未測試需求區"
    return False, ""

# ── Confluence 評分 ───────────────────────────────────
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

# ── 主要分析函數 ──────────────────────────────────────
def analyze_symbol(symbol: str) -> dict:
    """完整分析一個幣種，返回分析結果"""
    result = {"symbol": symbol, "error": None}

    df_1h  = get_klines(symbol, "1h",  500)
    df_4h  = get_klines(symbol, "4h",  200)   # v4.2 新增：4H OB 輔助 SL
    df_15m = get_klines(symbol, "15m", 300)
    df_3m  = get_klines(symbol, "3m",  200)
    levels = get_daily_weekly_levels(symbol)

    if df_1h is None or df_15m is None or df_3m is None:
        result["error"] = "無法取得數據"
        return result

    price = float(df_15m.iloc[-1]['close'])
    result["price"] = price
    result["levels"] = levels

    # 1H 方向 + BSL/SSL
    struct_1h = get_market_structure(df_1h, lookback=30)
    result["struct_1h"] = struct_1h
    bsl, ssl = get_bsl_ssl(df_1h.iloc[-100:])
    result["bsl"] = bsl
    result["ssl"] = ssl

    # 1H FIB + Swing High/Low（用於快報最外圍位 + SL 錨點）
    highs_1h, lows_1h = find_swing_dual(df_1h.iloc[-100:])
    result["highs_1h"] = highs_1h
    result["lows_1h"]  = lows_1h
    fib_1h = None
    if highs_1h and lows_1h:
        h1 = max(highs_1h, key=lambda x: x[0])[1]
        l1 = min(lows_1h, key=lambda x: x[0])[1]
        fib_1h = calc_fib(l1, h1, struct_1h if struct_1h != "ranging" else "bearish")
    result["fib_1h"] = fib_1h

    # 1H 關鍵區
    obs_1h  = find_order_blocks(df_1h.iloc[-100:])
    fvgs_1h = find_fvg(df_1h.iloc[-100:])
    for z in obs_1h + fvgs_1h:
        z['tf'] = '1h'
    zones_1h = obs_1h + fvgs_1h
    result["zones_1h"] = zones_1h

    # 4H OB（v4.2 新增：SL 錨點用）
    zones_4h = []
    if df_4h is not None:
        obs_4h = find_order_blocks(df_4h.iloc[-60:])
        for z in obs_4h:
            z['tf'] = '4h'
        zones_4h = obs_4h
    result["zones_4h"] = zones_4h

    # 15M 關鍵區
    obs_15m  = find_order_blocks(df_15m)
    fvgs_15m = find_fvg(df_15m)
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
    result["lows_15m"] = lows_15m

    fib_15m = None
    if highs_15m and lows_15m:
        h15 = max(highs_15m, key=lambda x: x[0])[1]
        l15 = min(lows_15m, key=lambda x: x[0])[1]
        fib_15m = calc_fib(l15, h15, struct_1h if struct_1h != "ranging" else "bearish")
    result["fib_15m"] = fib_15m

    result["atr_15m"] = calc_atr(df_15m)

    # 3M MSS
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

    return result

# ── 入場信號邏輯 ──────────────────────────────────────
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
        is_trend = (trade_type == "順勢")
        risk_type = "全倉" if is_trend else "半倉"
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

        # ── v4.2 新 SL 邏輯：關鍵區接觸確認 + 4H OB 錨點 ──
        sl, sl_desc = find_sl_anchor_zone(
            entry_price, direction,
            zones_15m, zones_4h,
            highs_15m, lows_15m,
            atr_15m
        )
        sl_dist = abs(entry_price - sl)

        # 確保 SL 距離足夠（最少 1×ATR 或 0.5%）
        min_dist = max(price * 0.005, atr_15m * 1.0)
        if sl_dist < min_dist:
            if direction == "bearish":
                sl = entry_price + min_dist
            else:
                sl = entry_price - min_dist
            sl_dist = min_dist
            sl_desc += "（已擴展至最小距離）"

        # TP
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

        # ── 動態 RR（最少 1:2）────────────────────────────
        # 優先：上方/下方最近的關鍵位作 TP1
        # 若 RR < 1:2，嘗試下一個更遠的關鍵位
        # 最終退回：SL 距離 × MIN_RR 作 TP1
        if not tp1:
            # 嘗試用關鍵位作 TP（找 RR >= 1:2 的最近位）
            all_levels = []
            for z in zones_1h + zones_15m:
                zm = z.get('mid', 0)
                if zm:
                    all_levels.append(zm)
            for e in eqh:
                ep = e.get('price', 0) if isinstance(e, dict) else e
                if ep:
                    all_levels.append(ep)
            for e in eql:
                ep = e.get('price', 0) if isinstance(e, dict) else e
                if ep:
                    all_levels.append(ep)

            if direction == "bearish":
                tp_candidates = sorted([l for l in all_levels if l < entry_price])
                for tc in tp_candidates:
                    if abs(entry_price - tc) / sl_dist >= MIN_RR:
                        tp1 = tc
                        tp_label = "關鍵位 TP（動態 RR）"
                        break
            else:
                tp_candidates = sorted([l for l in all_levels if l > entry_price], reverse=True)
                for tc in tp_candidates:
                    if abs(entry_price - tc) / sl_dist >= MIN_RR:
                        tp1 = tc
                        tp_label = "關鍵位 TP（動態 RR）"
                        break

            if not tp1:
                tp1 = (entry_price - sl_dist * MIN_RR if direction == "bearish"
                       else entry_price + sl_dist * MIN_RR)
                tp_label = f"1:{MIN_RR:.0f} RR 最低目標"

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
            "risk_type": risk_type,
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
        direction = mss_trend["direction"]
        sig = build_signal(direction, "順勢", mss_trend)
        if sig:
            return sig

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

# ── 每小時快報（v4.3 重構）────────────────────────────
def format_hourly_report(results: list) -> str:
    """
    每小時市場快報 v4.3
    ─────────────────────────────────────────────────────
    只顯示 BTC（減少訊息長度，ETH/SOL 繼續發入場信號）

    6 個關鍵位架構：
      ① 最外圍上方 = 1H 最近 Swing High（方向轉折點）
      ② 中間 2 個阻力 = 15M 最近供應位（由近至遠）
      ③ 中間 2 個支撐 = 15M 最近需求位（由近至遠）
      ④ 最外圍下方 = 1H 最近 Swing Low（方向轉折點）

    重疊標記（後綴）：
      若某個位置與 4H OB / 1H OB / 4H EQH/EQL / FIB 重疊
      → 在後面加 [+4H OB] / [+1H OB] / [+4H EQH] / [+0.618] 等

    層級陷阱警告：
      若 15M 阻力上方有未測試的 1H 關鍵位 → ⚠️ 層級陷阱

    顏色：
      🔴 強阻力（1H Swing High / PWH/PDH/WO/DO/BSL）
      🟠 弱阻力（15M OB/FVG/IFVG）
      🟢 強支撐（1H Swing Low / PWL/PDL/WO/DO/SSL）
      🔵 弱支撐（15M OB/FVG/IFVG）
    """
    now = datetime.now(HKT)
    msg  = f"🕐 每小時市場快報 [{now.strftime('%m-%d %H:%M')} HKT]\n"
    msg += "━━━━━━━━━━━━━━━━━━\n\n"

    # v4.3: 只顯示 BTC
    btc_result = next((r for r in results if r.get('symbol') == 'BTCUSDT'), None)
    if not btc_result:
        btc_result = results[0] if results else None
    display_results = [btc_result] if btc_result else []

    for result in display_results:
        try:
            if result.get("error"):
                sym = result.get('symbol', '?').replace('USDT', '/USDT')
                msg += f"📌 {sym}  ⚠️ 數據錯誤: {result['error']}\n\n"
                continue

            symbol    = result["symbol"]
            sym_short = symbol.replace("USDT", "/USDT")
            price     = result["price"]
            struct_1h = result.get("struct_1h", "ranging")
            struct_emoji = "⬇️ 看跌" if struct_1h == "bearish" else "⬆️ 看漲" if struct_1h == "bullish" else "↔️ 橫盤"
            levels    = result.get("levels", {})
            bsl       = result.get("bsl")
            ssl       = result.get("ssl")
            fib_1h    = result.get("fib_1h")
            zones_1h  = result.get("zones_1h", [])
            zones_4h  = result.get("zones_4h", [])
            zones_15m = result.get("zones_15m", [])
            highs_1h  = result.get("highs_1h", [])
            lows_1h   = result.get("lows_1h", [])
            eqh_list  = result.get("eqh", [])
            eql_list  = result.get("eql", [])

            # ── 層級陷阱檢查 ──────────────────────────────
            is_trap, trap_msg = check_hierarchy_trap(price, struct_1h, zones_1h, zones_15m)

            # ── 標題行 ────────────────────────────────────
            trap_warn = "  ⚠️ 層級陷阱" if is_trap else ""
            msg += f"📌 {sym_short}  💲{fmt(price, symbol)}  |  1H {struct_emoji}{trap_warn}\n"

            # ── 1H Swing High/Low（最外圍定方向位）──────────
            # 找最近的 1H Swing High（上方）和 Swing Low（下方）
            sh_above = sorted([h[1] for h in highs_1h if h[1] > price])
            sl_below = sorted([l[1] for l in lows_1h  if l[1] < price], reverse=True)
            outer_above_price = sh_above[0]  if sh_above else None
            outer_below_price = sl_below[0]  if sl_below else None

            # ── 15M 候選關鍵位池（中間四個）────────────────
            # 每個候選：(price, label, tf)
            candidates_15m = []

            def is_dup_15m(p):
                return any(abs(p - c[0]) / max(c[0], 0.001) < 0.003 for c in candidates_15m)

            # 週/日關鍵位加入 15M 候選（它們是最重要的近期位）
            for key, lbl in [('PWH','PWH 前週高'),('PDH','PDH 前日高'),
                              ('WO','WO 週開盤'),('DO','DO 日開盤'),
                              ('BSL','BSL 上方流動性')]:
                v = bsl if key == 'BSL' else levels.get(key)
                if v and v > price and not is_dup_15m(v):
                    candidates_15m.append((float(v), lbl, 'strong'))

            for key, lbl in [('PWL','PWL 前週低'),('PDL','PDL 前日低'),
                              ('WO','WO 週開盤'),('DO','DO 日開盤'),
                              ('SSL','SSL 下方流動性')]:
                v = ssl if key == 'SSL' else levels.get(key)
                if v and v < price and not is_dup_15m(v):
                    candidates_15m.append((float(v), lbl, 'strong'))

            # 15M OB/FVG/IFVG（方向過濾：上方只加供應位，下方只加需求位）
            for z in zones_15m:
                zm = z.get('mid', 0)
                if not zm or is_dup_15m(zm):
                    continue
                zdir = z.get('direction', '')
                lbl  = z.get('label', '15M 關鍵區')
                # 上方位：只接受 bearish（供應）
                if zm > price and zdir == 'bearish':
                    candidates_15m.append((zm, lbl, 'weak'))
                # 下方位：只接受 bullish（需求）
                elif zm < price and zdir == 'bullish':
                    candidates_15m.append((zm, lbl, 'weak'))

            above_15m = sorted([(p,l,s) for p,l,s in candidates_15m if p > price], key=lambda x: x[0])
            below_15m = sorted([(p,l,s) for p,l,s in candidates_15m if p < price], key=lambda x: x[0], reverse=True)

            near_above = above_15m[:2]
            near_below = below_15m[:2]

            # ── 重疊標記函數 ──────────────────────────────
            # 容差收緊至 ±0.3%，前綴改為空格分隔（無 + 號避免混淆）
            def overlap_tag(p: float) -> str:
                tags = []
                tol = 0.003  # ±0.3% 容差
                # FIB 重疊（只取最接近的一個）
                if fib_1h:
                    best_fib, best_dist = None, tol
                    for k in ["0.5", "0.618", "0.705", "0.786"]:
                        fv = fib_1h.get(k, 0)
                        if fv:
                            d = abs(p - fv) / max(fv, 0.001)
                            if d < best_dist:
                                best_dist, best_fib = d, k
                    if best_fib:
                        tags.append(best_fib)
                # 4H OB 重疊（只取最接近的一個）
                best_4h, best_dist = None, tol
                for z in zones_4h:
                    zm = z.get('mid', 0)
                    if zm:
                        d = abs(p - zm) / max(zm, 0.001)
                        if d < best_dist:
                            best_dist, best_4h = d, z
                if best_4h:
                    lbl = best_4h.get('label', '4H OB')
                    # 簡化標籤：OB+/OB− 格式
                    if 'OB+' in lbl or '需求' in lbl:
                        tags.append("4H OB+")
                    elif 'OB−' in lbl or '供應' in lbl:
                        tags.append("4H OB−")
                    else:
                        tags.append("4H OB")
                # 1H OB/FVG 重疊（只取最接近的一個）
                best_1h, best_dist = None, tol
                for z in zones_1h:
                    zm = z.get('mid', 0)
                    if zm:
                        d = abs(p - zm) / max(zm, 0.001)
                        if d < best_dist:
                            best_dist, best_1h = d, z
                if best_1h:
                    lbl = best_1h.get('label', '1H OB')
                    if 'OB+' in lbl or '需求' in lbl:
                        tags.append("1H OB+")
                    elif 'OB−' in lbl or 'FVG−' in lbl or '供應' in lbl:
                        tags.append("1H OB−")
                    elif 'FVG+' in lbl:
                        tags.append("1H FVG+")
                    else:
                        tags.append("1H OB")
                # 4H EQH/EQL 重疊
                for e in eqh_list:
                    ep = e.get('price', 0) if isinstance(e, dict) else e
                    if ep and abs(p - ep) / max(ep, 0.001) < tol:
                        tags.append("4H EQH")
                        break
                for e in eql_list:
                    ep = e.get('price', 0) if isinstance(e, dict) else e
                    if ep and abs(p - ep) / max(ep, 0.001) < tol:
                        tags.append("4H EQL")
                        break
                return f" [{', '.join(tags)}]" if tags else ""

            # ── 輸出 ──────────────────────────────────────
            # 最外圍上方（1H Swing High）
            if outer_above_price:
                ot = overlap_tag(outer_above_price)
                msg += f"   🔴 {fmt(outer_above_price, symbol)}  1H 結構高點（BOS 目標）{ot}\n"

            # 中間 2 個阻力（由遠至近，近的靠近現價）
            for p, lbl, st in reversed(near_above):
                emoji = '🔴' if st == 'strong' else '🟠'
                msg += f"   {emoji} {fmt(p, symbol)}  {lbl}{overlap_tag(p)}\n"

            # 現價分隔線
            msg += f"   ──── 💲{fmt(price, symbol)} 現價 ────\n"

            # 中間 2 個支撐（由近至遠）
            for p, lbl, st in near_below:
                emoji = '🟢' if st == 'strong' else '🔵'
                msg += f"   {emoji} {fmt(p, symbol)}  {lbl}{overlap_tag(p)}\n"

            # 最外圍下方（1H Swing Low）
            if outer_below_price:
                ot = overlap_tag(outer_below_price)
                msg += f"   🟢 {fmt(outer_below_price, symbol)}  1H 結構低點（跌破轉勢）{ot}\n"

            # 層級陷阱警告詳情
            if is_trap:
                msg += f"   ⚠️ {trap_msg}\n"

            msg += "\n"

        except Exception as e:
            sym = result.get('symbol', '?').replace('USDT', '/USDT')
            msg += f"📌 {sym}  ⚠️ 快報生成錯誤: {e}\n\n"
            logger.error(f"format_hourly_report {sym}: {e}", exc_info=True)

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
    last_signals: dict = {}

    await send_msg(bot,
        "✅ ICT/SMC 交易信號機械人 v4.3 已啟動\n\n"
        "📊 快報: BTC（每小時）  |  信號: BTC / ETH / SOL\n"
        "🎯 框架: 1H 定方向 → 15M 關鍵位 → 3M MSS → FIB OTE 入場\n"
        "🔧 v4.3①: SL = 15M OB/FVG Low/High 外側 + ATR×0.5 緩衝\n"
        "🔧 v4.3②: 快報最外圍位 = 1H 結構高/低點（BOS/轉勢點）\n"
        "🔧 v4.3③: 重疊標記（+4H OB / +1H OB / +4H EQH / +0.618）\n"
        "🔧 v4.3④: 層級陷阱警告（⚠️ 15M 阻力上方有 1H 未測試位）\n"
        "🔧 v4.3⑤: 動態 RR（最少 1:2，根據關鍵位動態計算）\n"
        "🌐 數據: data-api.binance.vision（1H 500根 / 4H 200根 / 15M 300根 / 3M 200根）"
    )

    logger.info("機械人 v4.3 已啟動，開始掃描...")

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
                        if now_ts - last_time > 600:
                            msg = format_signal_message(signal)
                            await send_msg(bot, msg)
                            last_signals[sig_key] = now_ts
                            logger.info(f"發送信號: {symbol} {signal['direction']} {signal['trade_type']}")

                except Exception as e:
                    logger.error(f"{symbol} 掃描錯誤: {e}")

            if now_ts - last_hourly >= HOURLY_REPORT_INTERVAL:
                if results:
                    try:
                        hourly_msg = format_hourly_report(results)
                        await send_msg(bot, hourly_msg)
                        last_hourly = now_ts
                        logger.info("已發送每小時快報")
                    except Exception as e:
                        logger.error(f"每小時快報發送失敗: {e}")
                        await send_msg(bot, f"⚠️ 每小時快報生成失敗: {e}")
                        last_hourly = now_ts

        except Exception as e:
            logger.error(f"主循環錯誤: {e}")

        await asyncio.sleep(SCAN_INTERVAL)

if __name__ == "__main__":
    asyncio.run(main())
