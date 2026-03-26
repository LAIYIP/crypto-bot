#!/usr/bin/env python3
"""
ICT/SMC 加密貨幣交易信號機械人 v3.0
數據源: Binance 公開 REST API（無需認證，繞過地區限制）

v3.0 核心改進:
- 修正 OB 定義：需要之後有 BOS/CHoCH 突破確認
- 修正 FVG 定義：加入最小寬度過濾（≥ 0.2%）
- 修正 IFVG 定義：進入 FVG 後反轉即算（不需完全穿越）
- 雙向偵測：順勢入場 + 逆向反轉入場（關鍵區支撐/阻力反轉）
- 假突破入場：流動性掃蕩後 CHoCH 確認
- ATR 止損改為 1.0×ATR，最少 0.5%
- 每小時快報加入趨勢方向、關鍵位置、當前價格
"""
import logging
import asyncio
import time
import requests
import pandas as pd
import numpy as np
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from datetime import datetime, timezone, timedelta
import os
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")
WATCH_SYMBOLS  = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
SCAN_INTERVAL  = 60
HKT = timezone(timedelta(hours=8))
BINANCE_BASE    = "https://api.binance.com"
BINANCE_BASE_US = "https://api.binance.us"
_active_base    = BINANCE_BASE

# 關鍵區最小寬度（相對價格百分比）
MIN_ZONE_WIDTH_PCT = 0.002   # 0.2%

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Kill Zone（僅顯示用，不限制入場）────────────────────────────────────────
KILL_ZONES = [
    ("亞洲開市",  1,  0,  3,  0),
    ("倫敦開市", 15,  0, 17,  0),
    ("紐約開市", 21, 30, 23, 30),
]

def in_kill_zone():
    now = datetime.now(HKT)
    cur = now.hour * 60 + now.minute
    for name, sh, sm, eh, em in KILL_ZONES:
        if sh*60+sm <= cur <= eh*60+em:
            return True, name
    return False, None

# ── 訂單編號 ──────────────────────────────────────────────────────────────────
order_counters = defaultdict(int)

def generate_order_id(symbol, direction):
    coin = symbol.replace("USDT", "")
    now  = datetime.now(HKT)
    date, hhmm = now.strftime("%Y%m%d"), now.strftime("%H%M")
    d = "L" if direction == "bullish" else "S"
    key = f"{coin}{date}"
    order_counters[key] += 1
    return f"#{coin}-{date}-{hhmm}-{d}{str(order_counters[key]).zfill(3)}"

# ── 狀態管理 ──────────────────────────────────────────────────────────────────
active_orders = {}
signal_states = defaultdict(lambda: {
    "state": 0,
    "last_signal_time": 0,
    "active_zone": None,
    "direction": None,
    "entry_type": None,
    "order_id": None,
})

# ── Binance API ───────────────────────────────────────────────────────────────
def get_klines(symbol, interval, limit=100):
    global _active_base
    for base in [_active_base, BINANCE_BASE_US, BINANCE_BASE]:
        try:
            r = requests.get(
                f"{base}/api/v3/klines",
                params={"symbol": symbol, "interval": interval, "limit": limit},
                timeout=10
            )
            if r.status_code == 200:
                _active_base = base
                df = pd.DataFrame(r.json(), columns=[
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

# ── ATR ───────────────────────────────────────────────────────────────────────
def calc_atr(df, period=14):
    if df is None or len(df) < period + 1:
        return None
    h = df['high'].values
    l = df['low'].values
    c = df['close'].values
    tr = np.maximum(h[1:] - l[1:],
         np.maximum(abs(h[1:] - c[:-1]),
                    abs(l[1:] - c[:-1])))
    return float(np.mean(tr[-period:]))

# ── 方向判斷（HH/HL 或 LH/LL）────────────────────────────────────────────────
def get_direction(df, lookback=10):
    if df is None or len(df) < lookback:
        return None
    sub = df.iloc[-lookback:]
    highs = sub['high'].values
    lows  = sub['low'].values
    hh = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i-1])
    lh = sum(1 for i in range(1, len(highs)) if highs[i] < highs[i-1])
    hl = sum(1 for i in range(1, len(lows))  if lows[i]  > lows[i-1])
    ll = sum(1 for i in range(1, len(lows))  if lows[i]  < lows[i-1])
    bull_score = hh + hl
    bear_score = lh + ll
    return "bullish" if bull_score > bear_score else "bearish"

# ── 擺動點識別（雙層：n=3短期 + n=8中期）────────────────────────────────────
def find_swing_points(df, n=5):
    highs, lows = [], []
    for i in range(n, len(df) - n):
        if float(df.iloc[i]['high']) == float(df.iloc[i-n:i+n+1]['high'].max()):
            highs.append((i, float(df.iloc[i]['high'])))
        if float(df.iloc[i]['low']) == float(df.iloc[i-n:i+n+1]['low'].min()):
            lows.append((i, float(df.iloc[i]['low'])))
    return highs, lows

def find_swing_points_dual(df):
    sh3, sl3 = find_swing_points(df, n=3)
    sh8, sl8 = find_swing_points(df, n=8)
    def merge(a, b):
        combined = list(a)
        for item in b:
            if not any(abs(item[1] - x[1]) / max(x[1], 0.001) < 0.001 for x in combined):
                combined.append(item)
        return sorted(combined, key=lambda x: x[0])
    return merge(sh3, sh8), merge(sl3, sl8)

# ── 前日/前週高低點 ───────────────────────────────────────────────────────────
def get_daily_weekly_levels(symbol):
    levels = {}
    try:
        r = requests.get(f"{_active_base}/api/v3/klines",
            params={"symbol": symbol, "interval": "1d", "limit": 3}, timeout=10)
        if r.status_code == 200:
            d = r.json()
            if len(d) >= 2:
                prev = d[-2]
                levels['PDH'] = float(prev[2])
                levels['PDL'] = float(prev[3])
                levels['DO']  = float(d[-1][1])
        r2 = requests.get(f"{_active_base}/api/v3/klines",
            params={"symbol": symbol, "interval": "1w", "limit": 3}, timeout=10)
        if r2.status_code == 200:
            w = r2.json()
            if len(w) >= 2:
                prev_w = w[-2]
                levels['PWH'] = float(prev_w[2])
                levels['PWL'] = float(prev_w[3])
                levels['WO']  = float(w[-1][1])
    except Exception as e:
        logger.warning(f"get_daily_weekly_levels {symbol}: {e}")
    return levels

# ── 關鍵區識別（完整 ICT/SMC，修正版）────────────────────────────────────────
def find_key_zones(df_15m, direction, df_1h=None, dw_levels=None):
    zones = []
    if df_15m is None or len(df_15m) < 30:
        return zones

    r = df_15m.iloc[-200:].copy().reset_index(drop=True)
    n = len(r)
    lc = float(r.iloc[-1]['close'])

    # ── 1. Order Block（OB）需 BOS 確認 ──────────────────────────────────────
    for i in range(2, n - 3):
        c  = r.iloc[i]
        body_c = abs(float(c['close']) - float(c['open']))
        if body_c == 0:
            continue

        # 看漲 OB：下跌中最後一根陰燭，之後急速上升突破前高
        if float(c['close']) < float(c['open']):
            prev_high = float(r.iloc[max(0,i-5):i]['high'].max()) if i > 0 else 0
            after = r.iloc[i+1:min(i+4, n)]
            if len(after) >= 2 and float(after['high'].max()) > prev_high:
                width = float(c['high']) - float(c['low'])
                if width / lc >= MIN_ZONE_WIDTH_PCT:
                    zones.append({
                        'type': 'OB', 'label': '15M 看漲 OB（需求區）',
                        'high': float(c['high']), 'low': float(c['low']),
                        'mid': float((c['high'] + c['low']) / 2),
                        'direction': 'bullish', 'strength': 'strong'
                    })

        # 看跌 OB：上升中最後一根陽燭，之後急速下跌突破前低
        if float(c['close']) > float(c['open']):
            prev_low = float(r.iloc[max(0,i-5):i]['low'].min()) if i > 0 else float('inf')
            after = r.iloc[i+1:min(i+4, n)]
            if len(after) >= 2 and float(after['low'].min()) < prev_low:
                width = float(c['high']) - float(c['low'])
                if width / lc >= MIN_ZONE_WIDTH_PCT:
                    zones.append({
                        'type': 'OB', 'label': '15M 看跌 OB（供應區）',
                        'high': float(c['high']), 'low': float(c['low']),
                        'mid': float((c['high'] + c['low']) / 2),
                        'direction': 'bearish', 'strength': 'strong'
                    })

    # ── 2. FVG（最小寬度過濾）────────────────────────────────────────────────
    fvg_zones = []
    for i in range(1, n - 1):
        k1 = r.iloc[i - 1]
        k3 = r.iloc[i + 1]

        gap_bull = float(k3['low']) - float(k1['high'])
        if gap_bull > 0 and gap_bull / lc >= MIN_ZONE_WIDTH_PCT:
            z = {
                'type': 'FVG', 'label': '15M 看漲 FVG（需求缺口）',
                'high': float(k3['low']), 'low': float(k1['high']),
                'mid': float((k3['low'] + k1['high']) / 2),
                'direction': 'bullish', 'strength': 'medium', 'bar_idx': i
            }
            zones.append(z)
            fvg_zones.append(z)

        gap_bear = float(k1['low']) - float(k3['high'])
        if gap_bear > 0 and gap_bear / lc >= MIN_ZONE_WIDTH_PCT:
            z = {
                'type': 'FVG', 'label': '15M 看跌 FVG（供應缺口）',
                'high': float(k1['low']), 'low': float(k3['high']),
                'mid': float((k1['low'] + k3['high']) / 2),
                'direction': 'bearish', 'strength': 'medium', 'bar_idx': i
            }
            zones.append(z)
            fvg_zones.append(z)

    # ── 3. IFVG（進入 FVG 後反轉即算）────────────────────────────────────────
    for fz in fvg_zones:
        bi = fz.get('bar_idx', 0)
        subsequent = r.iloc[bi + 2:] if bi + 2 < n else pd.DataFrame()
        if subsequent.empty:
            continue

        if fz['direction'] == 'bullish':
            for j in range(min(len(subsequent), 10)):
                bar = subsequent.iloc[j]
                if float(bar['low']) <= fz['high'] and float(bar['low']) >= fz['low']:
                    if j + 1 < len(subsequent):
                        next_bar = subsequent.iloc[j+1]
                        if float(next_bar['close']) < fz['mid']:
                            zones.append({
                                'type': 'IFVG', 'label': '15M 看跌 IFVG（反轉需求缺口→阻力）',
                                'high': fz['high'], 'low': fz['low'], 'mid': fz['mid'],
                                'direction': 'bearish', 'strength': 'medium'
                            })
                            break
        else:
            for j in range(min(len(subsequent), 10)):
                bar = subsequent.iloc[j]
                if float(bar['high']) >= fz['low']:
                    if j + 1 < len(subsequent):
                        next_bar = subsequent.iloc[j+1]
                        if float(next_bar['close']) > fz['mid']:
                            zones.append({
                                'type': 'IFVG', 'label': '15M 看漲 IFVG（反轉供應缺口→支撐）',
                                'high': fz['high'], 'low': fz['low'], 'mid': fz['mid'],
                                'direction': 'bullish', 'strength': 'medium'
                            })
                            break

    # ── 4. SNR（擺動高低點，雙層）──────────────────────────────────────────────
    sh, sl = find_swing_points_dual(r)
    for _, price in sh[-6:]:
        zones.append({
            'type': 'SNR', 'label': '15M 阻力 SNR',
            'high': price * 1.0015, 'low': price * 0.9985, 'mid': price,
            'direction': 'bearish', 'strength': 'medium'
        })
    for _, price in sl[-6:]:
        zones.append({
            'type': 'SNR', 'label': '15M 支撐 SNR',
            'high': price * 1.0015, 'low': price * 0.9985, 'mid': price,
            'direction': 'bullish', 'strength': 'medium'
        })

    # ── 5. Fibonacci（0.618 / 0.705 / 0.786）──────────────────────────────────
    if sh and sl:
        recent_lows  = sorted(sl, key=lambda x: x[0])
        recent_highs = sorted(sh, key=lambda x: x[0])
        if recent_lows and recent_highs:
            rl = recent_lows[0]
            rh = recent_highs[-1]
            if rl[0] < rh[0] and rh[1] > rl[1]:
                diff = rh[1] - rl[1]
                for fib, lbl in [(0.618, "FIB 0.618"), (0.705, "FIB 0.705"), (0.786, "FIB 0.786")]:
                    p2 = rh[1] - diff * fib
                    zones.append({
                        'type': 'FIB', 'label': f'15M {lbl} 回撤支撐',
                        'high': p2 * 1.001, 'low': p2 * 0.999, 'mid': p2,
                        'direction': 'bullish',
                        'strength': 'strong' if fib in [0.618, 0.705] else 'medium'
                    })
            rh2 = recent_highs[0]
            rl2 = recent_lows[-1]
            if rh2[0] < rl2[0] and rh2[1] > rl2[1]:
                diff2 = rh2[1] - rl2[1]
                for fib, lbl in [(0.618, "FIB 0.618"), (0.705, "FIB 0.705"), (0.786, "FIB 0.786")]:
                    p2 = rl2[1] + diff2 * fib
                    zones.append({
                        'type': 'FIB', 'label': f'15M {lbl} 回撤阻力',
                        'high': p2 * 1.001, 'low': p2 * 0.999, 'mid': p2,
                        'direction': 'bearish',
                        'strength': 'strong' if fib in [0.618, 0.705] else 'medium'
                    })

    # ── 6. EQH / EQL（等高/等低 = 流動性池）──────────────────────────────────
    tol = 0.002
    for i in range(len(sh)):
        for j in range(i + 1, len(sh)):
            if abs(sh[i][1] - sh[j][1]) / max(sh[i][1], 0.001) < tol:
                p2 = (sh[i][1] + sh[j][1]) / 2
                zones.append({
                    'type': 'EQH', 'label': '15M 等高點（流動性池）',
                    'high': p2 * 1.002, 'low': p2 * 0.998, 'mid': p2,
                    'direction': 'bearish', 'strength': 'strong'
                })
    for i in range(len(sl)):
        for j in range(i + 1, len(sl)):
            if abs(sl[i][1] - sl[j][1]) / max(sl[i][1], 0.001) < tol:
                p2 = (sl[i][1] + sl[j][1]) / 2
                zones.append({
                    'type': 'EQL', 'label': '15M 等低點（流動性池）',
                    'high': p2 * 1.002, 'low': p2 * 0.998, 'mid': p2,
                    'direction': 'bullish', 'strength': 'strong'
                })

    # ── 7. Breaker Block ──────────────────────────────────────────────────────
    for z in [x for x in zones if x['type'] == 'OB']:
        if z['direction'] == 'bullish' and lc < z['low'] * 0.998:
            zones.append({**z, 'type': 'Breaker',
                'label': '15M 看跌 Breaker Block', 'direction': 'bearish'})
        elif z['direction'] == 'bearish' and lc > z['high'] * 1.002:
            zones.append({**z, 'type': 'Breaker',
                'label': '15M 看漲 Breaker Block', 'direction': 'bullish'})

    # ── 8. 流動性掃蕩偵測（假突破）────────────────────────────────────────────
    recent5 = r.iloc[-5:]
    for z in [x for x in zones if x['type'] in ['EQH', 'EQL', 'SNR']]:
        swept = False
        reversed_after = False
        for i in range(len(recent5)):
            bar = recent5.iloc[i]
            if z['direction'] == 'bearish':
                if float(bar['high']) > z['high'] and float(bar['close']) < z['high']:
                    swept = True
                if swept and float(bar['close']) < z['low']:
                    reversed_after = True
            else:
                if float(bar['low']) < z['low'] and float(bar['close']) > z['low']:
                    swept = True
                if swept and float(bar['close']) > z['high']:
                    reversed_after = True
        if swept:
            sweep_dir = 'bullish' if z['direction'] == 'bearish' else 'bearish'
            label = f"假突破 {'↓ 看跌' if sweep_dir == 'bearish' else '↑ 看漲'}（{z['type']}）"
            zones.append({
                'type': 'Sweep', 'label': label,
                'high': z['high'], 'low': z['low'], 'mid': z['mid'],
                'direction': sweep_dir, 'strength': 'strong',
                'reversed': reversed_after
            })

    # ── 9. 跨時間框架 SNR（1H 映射到 15M）────────────────────────────────────
    if df_1h is not None and len(df_1h) >= 10:
        sh1h, sl1h = find_swing_points(df_1h.iloc[-50:].reset_index(drop=True), n=3)
        for _, price in sh1h[-4:]:
            zones.append({
                'type': 'HTF_SNR', 'label': '1H 阻力（跨時間框架）',
                'high': price * 1.002, 'low': price * 0.998, 'mid': price,
                'direction': 'bearish', 'strength': 'strong'
            })
        for _, price in sl1h[-4:]:
            zones.append({
                'type': 'HTF_SNR', 'label': '1H 支撐（跨時間框架）',
                'high': price * 1.002, 'low': price * 0.998, 'mid': price,
                'direction': 'bullish', 'strength': 'strong'
            })

    # ── 10. PDH/PDL/PWH/PWL/日開盤/週開盤 ────────────────────────────────────
    if dw_levels:
        mapping = [
            ('PDH', '前日高點 PDH', 'strong'),
            ('PDL', '前日低點 PDL', 'strong'),
            ('PWH', '前週高點 PWH', 'strong'),
            ('PWL', '前週低點 PWL', 'strong'),
            ('DO',  '今日開盤 DO',  'medium'),
            ('WO',  '本週開盤 WO',  'medium'),
        ]
        for key, label, strength in mapping:
            if key in dw_levels:
                price = dw_levels[key]
                actual_dir = 'bearish' if lc > price else 'bullish'
                zones.append({
                    'type': key, 'label': label,
                    'high': price * 1.001, 'low': price * 0.999, 'mid': price,
                    'direction': actual_dir, 'strength': strength
                })

    # ── 去重 ──────────────────────────────────────────────────────────────────
    deduped = []
    for z in zones:
        if not any(
            z['direction'] == d['direction'] and
            abs(z['mid'] - d['mid']) / max(d['mid'], 0.001) < 0.003
            for d in deduped
        ):
            deduped.append(z)

    deduped.sort(key=lambda z: abs(z['mid'] - lc))
    return deduped


# ── TP 區域 ───────────────────────────────────────────────────────────────────
def find_tp_zone(entry_price, direction, df_15m, df_1h=None, dw_levels=None):
    opp = 'bearish' if direction == 'bullish' else 'bullish'
    all_zones = find_key_zones(df_15m, direction, df_1h=df_1h, dw_levels=dw_levels)
    opp_zones = [z for z in all_zones if z['direction'] == opp]
    if not opp_zones:
        return None
    if direction == 'bullish':
        above = [z for z in opp_zones if z['mid'] > entry_price * 1.005]
        return min(above, key=lambda z: z['mid']) if above else None
    else:
        below = [z for z in opp_zones if z['mid'] < entry_price * 0.995]
        return max(below, key=lambda z: z['mid']) if below else None


def assess_tp_strength(tp_zone):
    if not tp_zone:
        return "未知"
    if tp_zone['type'] in ['OB', 'EQH', 'EQL', 'Breaker', 'HTF_SNR', 'PWH', 'PWL', 'PDH', 'PDL'] \
            or tp_zone.get('strength') == 'strong':
        return "強（謹慎，價格可能提前反轉）"
    return "中等"


# ── SL 計算（ATR 緩衝，最少 0.5%）────────────────────────────────────────────
def calc_sl_tp(entry, zone, direction, df_15m, tp_zone=None):
    atr = calc_atr(df_15m, 14) or (entry * 0.005)
    min_sl_dist = entry * 0.005

    if direction == 'bullish':
        sl_raw = zone['low'] - atr
        sl = min(sl_raw, entry - min_sl_dist)
        risk = entry - sl
        if tp_zone and risk > 0:
            tp = tp_zone['low'] * 0.999
            if (tp - entry) / risk < 1.5:
                tp = entry + risk * 2
        else:
            tp = entry + (risk * 2 if risk > 0 else entry * 0.01)
    else:
        sl_raw = zone['high'] + atr
        sl = max(sl_raw, entry + min_sl_dist)
        risk = sl - entry
        if tp_zone and risk > 0:
            tp = tp_zone['high'] * 1.001
            if (entry - tp) / risk < 1.5:
                tp = entry - risk * 2
        else:
            tp = entry - (risk * 2 if risk > 0 else entry * 0.01)

    return round(sl, 4), round(tp, 4)


# ── 1M 結構偵測（CHoCH / BOS）────────────────────────────────────────────────
def check_1m_structure(df_1m, direction):
    if df_1m is None or len(df_1m) < 10:
        return "none", 0
    recent = df_1m.iloc[-20:]
    cp = float(recent.iloc[-1]['close'])
    sh, sl = find_swing_points(recent, n=2)

    if direction == 'bullish':
        if not sh:
            return "none", 0
        sorted_h = sorted(sh, key=lambda x: x[0])
        last_high = sorted_h[-1][1]
        if cp > last_high:
            if len(sorted_h) > 1 and cp > max(h[1] for h in sorted_h):
                return "bos", cp
            return "choch", cp
    else:
        if not sl:
            return "none", 0
        sorted_l = sorted(sl, key=lambda x: x[0])
        last_low = sorted_l[-1][1]
        if cp < last_low:
            if len(sorted_l) > 1 and cp < min(l[1] for l in sorted_l):
                return "bos", cp
            return "choch", cp
    return "none", 0


# ── 5M 形態偵測 ───────────────────────────────────────────────────────────────
def check_5m_pattern(df_5m):
    if df_5m is None or len(df_5m) < 3:
        return "普通", None
    c, p = df_5m.iloc[-1], df_5m.iloc[-2]
    body = abs(float(c['close']) - float(c['open']))
    uw   = float(c['high']) - max(float(c['close']), float(c['open']))
    lw   = min(float(c['close']), float(c['open'])) - float(c['low'])
    pb   = abs(float(p['close']) - float(p['open']))
    if body == 0:
        return "普通", None
    if float(c['open']) > float(p['close']) and float(c['close']) < float(p['open']) and body > pb:
        return "看跌吞沒", "bearish"
    if float(c['open']) < float(p['close']) and float(c['close']) > float(p['open']) and body > pb:
        return "看漲吞沒", "bullish"
    if uw > body * 2 and lw < body * 0.5 and float(c['close']) < float(c['open']):
        return "射擊之星", "bearish"
    if lw > body * 2 and uw < body * 0.5 and float(c['close']) > float(c['open']):
        return "錘子線", "bullish"
    if uw > body * 3:
        return "Pin Bar（看跌）", "bearish"
    if lw > body * 3:
        return "Pin Bar（看漲）", "bullish"
    return "普通", None


# ── 輔助函數 ──────────────────────────────────────────────────────────────────
async def send_msg(app, chat_id, text):
    try:
        await app.bot.send_message(chat_id=chat_id, text=text, parse_mode='HTML')
    except Exception as e:
        logger.error(f"Telegram 發送失敗: {e}")

def fp(p):
    if p >= 1000:
        return f"{p:,.2f}"
    elif p >= 10:
        return f"{p:.3f}"
    else:
        return f"{p:.4f}"

def de(d):
    return "⬆️ 看漲" if d == "bullish" else "⬇️ 看跌"

def entry_type_label(et):
    if et == "trend":     return "順勢入場"
    if et == "reversal":  return "逆向反轉"
    if et == "sweep":     return "假突破反轉"
    return "入場"


# ── 留意信號發送 ──────────────────────────────────────────────────────────────
async def _send_alert(app, chat_id, dsym, dir4, dir1, az, cp, kzs, reason, entry_dir, entry_type):
    if entry_dir == 'bullish':
        guide_in  = f"{fp(az['high'])} 以上站穩 → 考慮做多"
        guide_out = f"跌破 {fp(az['low'])} → 關鍵區失守，暫不入場"
    else:
        guide_in  = f"{fp(az['low'])} 以下站穩 → 考慮做空"
        guide_out = f"升破 {fp(az['high'])} → 關鍵區失守，暫不入場"

    type_icon = {"trend": "📈 順勢", "reversal": "🔄 逆向反轉", "sweep": "⚡ 假突破反轉"}.get(entry_type, "")

    await app.bot.send_message(
        chat_id=chat_id,
        text=(
            f"⚠️ <b>【留意信號】{dsym}</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"📊 <b>4H:</b> {de(dir4)}  |  <b>1H:</b> {de(dir1)}\n"
            f"{reason}\n\n"
            f"🎯 <b>關鍵區域:</b> {az['label']}\n"
            f"📍 <b>區域範圍:</b> {fp(az['low'])} - {fp(az['high'])}\n"
            f"💲 <b>當前價格:</b> {fp(cp)}\n"
            f"🏷️ <b>入場類型:</b> {type_icon}\n\n"
            f"📌 <b>操作指引:</b>\n"
            f"• {guide_in}\n"
            f"• {guide_out}\n\n"
            f"⏰ {kzs}  |  <i>等待 1M CHoCH 確認...</i>"
        ),
        parse_mode='HTML'
    )


# ── 主掃描循環 ────────────────────────────────────────────────────────────────
async def auto_scan(app, chat_id):
    logger.info("自動掃描已啟動 v3.0")
    await send_msg(app, chat_id,
        "✅ <b>ICT 交易信號機械人 v3.0 已啟動</b>\n\n"
        "📊 監控: BTC / ETH / SOL\n"
        "🎯 策略: 4H+1H方向 → 15M關鍵區 → 1M CHoCH/BOS\n"
        "🆕 v3.0 改進:\n"
        "  • OB 需 BOS 突破確認\n"
        "  • FVG 最小寬度 0.2% 過濾\n"
        "  • IFVG 進入後反轉即算\n"
        "  • 雙向偵測（順勢 + 逆向反轉）\n"
        "  • 假突破入場\n"
        "  • SL = 關鍵區外 + 1.0×ATR（最少 0.5%）\n"
        "⏰ 每 60 秒掃描\n"
        "🌐 數據源: Binance 公開 API"
    )

    hb = 0
    while True:
        try:
            for sym in WATCH_SYMBOLS:
                try:
                    d4h  = get_klines(sym, "4h",  30)
                    d1h  = get_klines(sym, "1h",  50)
                    d15m = get_klines(sym, "15m", 200)
                    d5m  = get_klines(sym, "5m",  10)
                    d1m  = get_klines(sym, "1m",  25)
                    if any(x is None for x in [d4h, d1h, d15m, d1m]):
                        logger.warning(f"跳過 {sym}（數據獲取失敗）")
                        continue

                    dw   = get_daily_weekly_levels(sym)
                    cp   = float(d1m.iloc[-1]['close'])
                    dir4 = get_direction(d4h, 10)
                    dir1 = get_direction(d1h, 10)
                    if not dir4 or not dir1:
                        continue

                    st   = signal_states[sym]
                    now  = time.time()
                    dsym = sym.replace("USDT", "/USDT")
                    inkz, kzn = in_kill_zone()
                    kzs  = f"🔴 {kzn}" if inkz else "⚪ 非 KZ"

                    # 重置：價格離開活躍區域
                    if st["active_zone"] and st["state"] == 1:
                        z = st["active_zone"]
                        if cp < z['low'] * 0.994 or cp > z['high'] * 1.006:
                            st.update({"state": 0, "active_zone": None, "order_id": None, "entry_type": None})

                    if st["state"] == 0 and now - st["last_signal_time"] < 5400:
                        continue

                    # 取得所有關鍵區
                    all_zones = find_key_zones(d15m, dir1, df_1h=d1h, dw_levels=dw)

                    trend_zones    = [z for z in all_zones if z['direction'] == dir1]
                    opp_dir        = 'bearish' if dir1 == 'bullish' else 'bullish'
                    reversal_zones = [z for z in all_zones if z['direction'] == opp_dir]
                    sweep_zones    = [z for z in all_zones if z['type'] == 'Sweep']

                    trend_az    = next((z for z in trend_zones    if z['low'] * 0.999 <= cp <= z['high'] * 1.001), None)
                    reversal_az = next((z for z in reversal_zones if z['low'] * 0.999 <= cp <= z['high'] * 1.001), None)
                    sweep_az    = next((z for z in sweep_zones    if z['low'] * 0.999 <= cp <= z['high'] * 1.001), None)

                    # State 0：偵測入場機會
                    if st["state"] == 0:

                        # 優先級 1：假突破反轉（最高信心）
                        if sweep_az and sweep_az.get('reversed'):
                            entry_dir = sweep_az['direction']
                            reason = (f"⚡ <b>假突破反轉</b>：{sweep_az['label']}\n"
                                      f"掃蕩後已確認反轉，方向：{de(entry_dir)}")
                            await _send_alert(app, chat_id, dsym, dir4, dir1, sweep_az,
                                              cp, kzs, reason, entry_dir, "sweep")
                            st.update({"state": 1, "active_zone": sweep_az,
                                       "direction": entry_dir, "entry_type": "sweep",
                                       "last_signal_time": now})

                        # 優先級 2：順勢入場
                        elif trend_az:
                            align = ("✅ 4H 同 1H 方向一致（強信號）" if dir4 == dir1
                                     else "⚠️ 1H 逆 4H 回調（目標 4H 關鍵區）")
                            reason = (f"{align}\n"
                                      f"進入{'需求' if dir1 == 'bullish' else '供應'}區，等待 1M CHoCH 確認")
                            await _send_alert(app, chat_id, dsym, dir4, dir1, trend_az,
                                              cp, kzs, reason, dir1, "trend")
                            st.update({"state": 1, "active_zone": trend_az,
                                       "direction": dir1, "entry_type": "trend",
                                       "last_signal_time": now})

                        # 優先級 3：逆向反轉（只在強關鍵區）
                        elif reversal_az and reversal_az.get('strength') == 'strong':
                            rev_dir = reversal_az['direction']
                            reason = (f"🔄 <b>逆向反轉機會</b>（{reversal_az['type']}）\n"
                                      f"強關鍵區 {reversal_az['label']} 出現反應\n"
                                      f"1H {de(dir1)} 但關鍵區支撐/阻力強，等待 1M CHoCH")
                            await _send_alert(app, chat_id, dsym, dir4, dir1, reversal_az,
                                              cp, kzs, reason, rev_dir, "reversal")
                            st.update({"state": 1, "active_zone": reversal_az,
                                       "direction": rev_dir, "entry_type": "reversal",
                                       "last_signal_time": now})

                    # State 1：等待 1M CHoCH
                    elif st["state"] == 1:
                        az = st["active_zone"]
                        if az and (az['low'] * 0.997 <= cp <= az['high'] * 1.003):
                            stype, _ = check_1m_structure(d1m, st["direction"])
                            if stype in ["choch", "bos"]:
                                oid = generate_order_id(sym, st["direction"])
                                tpz = find_tp_zone(cp, st["direction"], d15m, df_1h=d1h, dw_levels=dw)
                                sl, tp = calc_sl_tp(cp, az, st["direction"], d15m, tpz)
                                tps  = assess_tp_strength(tpz)
                                risk = abs(cp - sl)
                                rr   = abs(tp - cp) / risk if risk > 0 else 0
                                ds   = "🟢 做多 (Long)" if st["direction"] == "bullish" else "🔴 做空 (Short)"
                                tpl  = tpz['label'] if tpz else "無明確關鍵區（使用 1:2 RR）"
                                atr_val = calc_atr(d15m, 14) or 0
                                etl  = entry_type_label(st.get("entry_type", "trend"))
                                struct_label = "1M CHoCH 反轉確認" if stype == "choch" else "1M BOS 突破確認"

                                await send_msg(app, chat_id,
                                    f"🚨 <b>【入場信號】{dsym}</b>\n"
                                    f"━━━━━━━━━━━━━━━━━━\n"
                                    f"📋 <b>訂單編號:</b> <code>{oid}</code>\n\n"
                                    f"✅ <b>確認條件:</b>\n"
                                    f"• 4H: {de(dir4)} | 1H: {de(dir1)}\n"
                                    f"• {kzs}\n"
                                    f"• {struct_label}\n"
                                    f"• 關鍵區: {az['label']}\n"
                                    f"• 入場類型: {etl}\n\n"
                                    f"📈 <b>交易方向:</b> {ds}\n\n"
                                    f"💵 <b>入場價格:</b> {fp(cp)}\n"
                                    f"🛑 <b>止損 (SL):</b> {fp(sl)}\n"
                                    f"   └ 關鍵區外 + 1.0×ATR ({fp(atr_val)})\n"
                                    f"🎯 <b>止盈 (TP):</b> {fp(tp)}\n"
                                    f"   TP 目標: {tpl}\n"
                                    f"   TP 區強度: {tps}\n"
                                    f"   預計 RR: 1:{rr:.1f}\n\n"
                                    f"⚠️ <b>確認風險後入場</b>"
                                )
                                active_orders[oid] = {
                                    "symbol": sym, "direction": st["direction"],
                                    "entry": cp, "sl": sl, "tp": tp, "tp_zone": tpz,
                                    "state": "open", "time": now, "tp_alerted": False,
                                }
                                st.update({"state": 2, "order_id": oid, "last_signal_time": now})

                    # State 2：監控持倉
                    elif st["state"] == 2:
                        oid = st["order_id"]
                        if oid and oid in active_orders:
                            o = active_orders[oid]
                            stype, _ = check_1m_structure(d1m, st["direction"])
                            if stype == "bos" and now - st["last_signal_time"] > 120:
                                await send_msg(app, chat_id,
                                    f"✅ <b>【確認信號】{dsym}</b>\n"
                                    f"━━━━━━━━━━━━━━━━━━\n"
                                    f"📋 <b>訂單編號:</b> <code>{oid}</code>\n\n"
                                    f"🔥 <b>1M BOS 突破結構確認</b>\n"
                                    f"💲 當前價格: {fp(cp)}\n\n"
                                    f"<i>趨勢已確認，可考慮加倉或持有\n"
                                    f"建議: 移動 SL 至成本價 {fp(o['entry'])} 保本</i>"
                                )
                                st.update({"state": 3, "last_signal_time": now})

                            if not o.get("tp_alerted"):
                                span = abs(o["tp"] - o["entry"])
                                if span > 0 and abs(cp - o["tp"]) / span < 0.15:
                                    pat, pdir = check_5m_pattern(d5m)
                                    opp_p = "bearish" if o["direction"] == "bullish" else "bullish"
                                    if pat != "普通" and pdir == opp_p:
                                        await send_msg(app, chat_id,
                                            f"🔔 <b>【提早 TP 警告】{dsym}</b>\n"
                                            f"━━━━━━━━━━━━━━━━━━\n"
                                            f"📋 <b>訂單編號:</b> <code>{oid}</code>\n\n"
                                            f"🕯️ TP 區域出現 <b>{pat}</b>\n"
                                            f"📍 TP 目標: {fp(o['tp'])}\n"
                                            f"💲 當前價格: {fp(cp)}\n\n"
                                            f"⚠️ <b>建議: 提前平倉 / 做套保</b>"
                                        )
                                        o["tp_alerted"] = True
                                    elif o.get("tp_zone") and o["tp_zone"].get("strength") == "strong":
                                        await send_msg(app, chat_id,
                                            f"⚡️ <b>【持倉提示】{dsym}</b>\n"
                                            f"━━━━━━━━━━━━━━━━━━\n"
                                            f"📋 <b>訂單編號:</b> <code>{oid}</code>\n\n"
                                            f"📍 接近強 TP 區域: {o['tp_zone']['label']}\n"
                                            f"💲 當前價格: {fp(cp)}\n"
                                            f"🎯 TP 目標: {fp(o['tp'])}\n\n"
                                            f"⚠️ <b>建議: 考慮移動 SL 至成本價或提前平倉</b>"
                                        )
                                        o["tp_alerted"] = True

                    # 5M 形態確認
                    if st["state"] in [1, 2]:
                        pat, pdir = check_5m_pattern(d5m)
                        if (pat != "普通" and pdir == st.get("direction") and
                                now - st["last_signal_time"] > 300):
                            oid2 = st.get("order_id", "")
                            os2 = f"\n📋 <b>訂單編號:</b> <code>{oid2}</code>" if oid2 else ""
                            await send_msg(app, chat_id,
                                f"🕯️ <b>【5M 確認形態】{dsym}</b>\n"
                                f"━━━━━━━━━━━━━━━━━━\n"
                                f"{os2}\n"
                                f"形態: <b>{pat}</b>\n"
                                f"💲 當前價格: {fp(cp)}\n"
                                f"<i>可作為額外入場確認</i>"
                            )
                            st["last_signal_time"] = now

                except Exception as e:
                    logger.error(f"掃描 {sym} 失敗: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"主循環失敗: {e}", exc_info=True)

        # 每小時市場快報
        hb += 1
        if hb >= 60:
            hb = 0
            nows = datetime.now(HKT).strftime('%m-%d %H:%M')
            inkz, kzn = in_kill_zone()
            kzs = f"🔴 {kzn}" if inkz else "⚪ 非 Kill Zone"

            lines = [
                f"🕐 <b>每小時市場快報</b> [{nows} HKT]",
                f"━━━━━━━━━━━━━━━━━━",
                f"⏰ {kzs}",
                ""
            ]
            for rsym in WATCH_SYMBOLS:
                try:
                    rd4h  = get_klines(rsym, "4h",  30)
                    rd1h  = get_klines(rsym, "1h",  50)
                    rd15m = get_klines(rsym, "15m", 200)
                    rd1m  = get_klines(rsym, "1m",   5)
                    rdw   = get_daily_weekly_levels(rsym)
                    if any(x is None for x in [rd4h, rd1h, rd15m, rd1m]):
                        lines.append(f"📌 <b>{rsym.replace('USDT','/USDT')}</b>: 數據獲取失敗")
                        continue

                    rcp   = float(rd1m.iloc[-1]['close'])
                    rdir4 = get_direction(rd4h, 10)
                    rdir1 = get_direction(rd1h, 10)
                    dsym2 = rsym.replace('USDT', '/USDT')

                    d4s = de(rdir4) if rdir4 else "N/A"
                    d1s = de(rdir1) if rdir1 else "N/A"

                    if rdir4 and rdir1:
                        trend_note = "✅ 同向" if rdir4 == rdir1 else "⚠️ 背離"
                    else:
                        trend_note = ""

                    lines.append(f"📌 <b>{dsym2}</b>  💲{fp(rcp)}")
                    lines.append(f"   4H {d4s} | 1H {d1s}  {trend_note}")

                    if rdw:
                        pdh = rdw.get('PDH'); pdl = rdw.get('PDL')
                        pwh = rdw.get('PWH'); pwl = rdw.get('PWL')
                        do  = rdw.get('DO')
                        if pdh and pdl:
                            lines.append(f"   📅 前日: 高 {fp(pdh)} / 低 {fp(pdl)}")
                        if pwh and pwl:
                            lines.append(f"   📆 前週: 高 {fp(pwh)} / 低 {fp(pwl)}")
                        if do:
                            lines.append(f"   🔵 今日開盤: {fp(do)}")

                    if rdir1:
                        rzones = find_key_zones(rd15m, rdir1, df_1h=rd1h, dw_levels=rdw)
                        above  = [z for z in rzones if z['mid'] > rcp and z['direction'] == 'bearish']
                        below  = [z for z in rzones if z['mid'] <= rcp and z['direction'] == 'bullish']
                        if below:
                            bz = max(below, key=lambda z: z['mid'])
                            lines.append(f"   🟢 最近支撐: {fp(bz['low'])}-{fp(bz['high'])} [{bz['type']}]")
                        if above:
                            az2 = min(above, key=lambda z: z['mid'])
                            lines.append(f"   🔴 最近阻力: {fp(az2['low'])}-{fp(az2['high'])} [{az2['type']}]")
                        if not below and not above:
                            lines.append(f"   ⚪ 暫無明確關鍵區")
                    lines.append("")

                except Exception as e:
                    lines.append(f"📌 <b>{rsym.replace('USDT','/USDT')}</b>: 錯誤 {e}")

            await send_msg(app, chat_id, "\n".join(lines))

        await asyncio.sleep(SCAN_INTERVAL)


# ── Telegram 指令處理 ─────────────────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 <b>ICT 交易信號機械人 v3.0</b>\n\n"
        "📊 監控: BTC / ETH / SOL\n"
        "🎯 策略: 4H+1H → 15M關鍵區 → 1M CHoCH/BOS\n"
        "🆕 v3.0: 正確 OB/FVG/IFVG 定義 + 雙向偵測 + 假突破入場\n"
        "⏰ 每 60 秒掃描",
        parse_mode='HTML'
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("機械人正在後台自動掃描市場中...")

def main():
    logger.info("正在啟動機械人 v3.0...")
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("缺少 TELEGRAM 環境變量")
        return
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    async def start_scan(context):
        await auto_scan(app, TELEGRAM_CHAT_ID)

    app.job_queue.run_once(start_scan, when=0)
    logger.info("✅ 機械人 v3.0 已啟動")
    app.run_polling()

if __name__ == '__main__':
    main()
