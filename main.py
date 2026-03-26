#!/usr/bin/env python3
"""
ICT/SMC Crypto Trading Signal Bot v2.0
Data source: Binance public REST API (no auth, bypasses geo-restriction)

New in v2.0:
- IFVG (Inverse FVG)
- PDH/PDL (Previous Day High/Low)
- PWH/PWL (Previous Week High/Low)
- Daily/Weekly Open
- Liquidity Sweep detection
- Cross-timeframe SNR (1H to 15M mapping)
- 200-bar 15M data + dual swing point detection (n=3 short + n=8 medium)
- ATR-based SL (OB outer edge + 0.5xATR14)
- Kill Zone restriction REMOVED
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

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")
WATCH_SYMBOLS  = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
SCAN_INTERVAL  = 60
HKT = timezone(timedelta(hours=8))
BINANCE_BASE    = "https://api.binance.com"
BINANCE_BASE_US = "https://api.binance.us"
_active_base    = BINANCE_BASE

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

KILL_ZONES = [
    ("Asia Open",    1,  0,  3,  0),
    ("London Open", 15,  0, 17,  0),
    ("NY Open",     21, 30, 23, 30),
]
KZ_NAMES = {"Asia Open": "亞洲開市", "London Open": "倫敦開市", "NY Open": "紐約開市"}

def in_kill_zone():
    now = datetime.now(HKT)
    cur = now.hour * 60 + now.minute
    for name, sh, sm, eh, em in KILL_ZONES:
        if sh*60+sm <= cur <= eh*60+em:
            return True, KZ_NAMES[name]
    return False, None

order_counters = defaultdict(int)

def generate_order_id(symbol, direction):
    coin = symbol.replace("USDT", "")
    now  = datetime.now(HKT)
    date, hhmm = now.strftime("%Y%m%d"), now.strftime("%H%M")
    d = "L" if direction == "bullish" else "S"
    key = f"{coin}{date}"
    order_counters[key] += 1
    return f"#{coin}-{date}-{hhmm}-{d}{str(order_counters[key]).zfill(3)}"

active_orders = {}
signal_states = defaultdict(lambda: {
    "state": 0, "last_signal_time": 0,
    "active_zone": None, "direction": None, "order_id": None,
})

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

def calc_atr(df, period=14):
    if df is None or len(df) < period + 1:
        return None
    h = df['high'].values
    l = df['low'].values
    c = df['close'].values
    tr = np.maximum(h[1:] - l[1:],
         np.maximum(abs(h[1:] - c[:-1]),
                    abs(l[1:] - c[:-1])))
    atr = float(np.mean(tr[-period:]))
    return atr

def get_direction(df, lookback=10):
    if df is None or len(df) < lookback:
        return None
    h = df.iloc[-lookback:]['high'].values
    up = sum(1 for i in range(1, len(h)) if h[i] > h[i-1])
    dn = sum(1 for i in range(1, len(h)) if h[i] < h[i-1])
    return "bullish" if up > dn else "bearish"

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
            if not any(abs(item[1] - x[1]) / x[1] < 0.001 for x in combined):
                combined.append(item)
        return sorted(combined, key=lambda x: x[0])
    return merge(sh3, sh8), merge(sl3, sl8)

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

def find_key_zones(df_15m, direction, df_1h=None, dw_levels=None):
    zones = []
    if df_15m is None or len(df_15m) < 30:
        return zones

    r = df_15m.iloc[-200:].copy().reset_index(drop=True)
    n = len(r)
    lc = float(r.iloc[-1]['close'])

    # 1. Order Blocks
    for i in range(1, n - 2):
        c  = r.iloc[i]
        p  = r.iloc[i - 1]
        n1 = r.iloc[i + 1]
        n2 = r.iloc[i + 2]
        if (c['close'] < c['open'] and
                n1['close'] > n1['open'] and
                n2['close'] > n2['open'] and
                n2['close'] > p['high']):
            zones.append({'type': 'OB', 'label': '15M 看漲 OB（需求區）',
                'high': float(c['high']), 'low': float(c['low']),
                'mid': float((c['high'] + c['low']) / 2),
                'direction': 'bullish', 'strength': 'strong'})
        if (c['close'] > c['open'] and
                n1['close'] < n1['open'] and
                n2['close'] < n2['open'] and
                n2['close'] < p['low']):
            zones.append({'type': 'OB', 'label': '15M 看跌 OB（供應區）',
                'high': float(c['high']), 'low': float(c['low']),
                'mid': float((c['high'] + c['low']) / 2),
                'direction': 'bearish', 'strength': 'strong'})

    # 2. FVG
    fvg_zones = []
    for i in range(1, n - 1):
        p  = r.iloc[i - 1]
        nk = r.iloc[i + 1]
        if float(p['high']) < float(nk['low']):
            z = {'type': 'FVG', 'label': '15M 看漲 FVG',
                 'high': float(nk['low']), 'low': float(p['high']),
                 'mid': float((nk['low'] + p['high']) / 2),
                 'direction': 'bullish', 'strength': 'medium', 'bar_idx': i}
            zones.append(z)
            fvg_zones.append(z)
        if float(p['low']) > float(nk['high']):
            z = {'type': 'FVG', 'label': '15M 看跌 FVG',
                 'high': float(p['low']), 'low': float(nk['high']),
                 'mid': float((p['low'] + nk['high']) / 2),
                 'direction': 'bearish', 'strength': 'medium', 'bar_idx': i}
            zones.append(z)
            fvg_zones.append(z)

    # 3. IFVG
    for fz in fvg_zones:
        bi = fz.get('bar_idx', 0)
        subsequent = r.iloc[bi + 2:] if bi + 2 < n else pd.DataFrame()
        if subsequent.empty:
            continue
        if fz['direction'] == 'bullish':
            if float(subsequent['close'].min()) < fz['low']:
                zones.append({'type': 'IFVG', 'label': '15M 看跌 IFVG（反轉 FVG）',
                    'high': fz['high'], 'low': fz['low'], 'mid': fz['mid'],
                    'direction': 'bearish', 'strength': 'medium'})
        else:
            if float(subsequent['close'].max()) > fz['high']:
                zones.append({'type': 'IFVG', 'label': '15M 看漲 IFVG（反轉 FVG）',
                    'high': fz['high'], 'low': fz['low'], 'mid': fz['mid'],
                    'direction': 'bullish', 'strength': 'medium'})

    # 4. SNR dual-layer
    sh, sl = find_swing_points_dual(r)
    for _, price in sh[-5:]:
        zones.append({'type': 'SNR', 'label': '15M 阻力 SNR',
            'high': price * 1.001, 'low': price * 0.999, 'mid': price,
            'direction': 'bearish', 'strength': 'medium'})
    for _, price in sl[-5:]:
        zones.append({'type': 'SNR', 'label': '15M 支撐 SNR',
            'high': price * 1.001, 'low': price * 0.999, 'mid': price,
            'direction': 'bullish', 'strength': 'medium'})

    # 5. Fibonacci
    if sh and sl:
        if direction == "bullish":
            rl = min(sl, key=lambda x: x[0])
            rh = max(sh, key=lambda x: x[0])
            if rl[0] < rh[0]:
                diff = rh[1] - rl[1]
                for fib, lbl in [(0.618, "FIB 0.618"), (0.705, "FIB 0.705"), (0.786, "FIB 0.786")]:
                    p2 = rh[1] - diff * fib
                    zones.append({'type': 'FIB', 'label': f'15M {lbl} 回撤支撐',
                        'high': p2 * 1.001, 'low': p2 * 0.999, 'mid': p2,
                        'direction': 'bullish',
                        'strength': 'strong' if fib in [0.618, 0.705] else 'medium'})
        else:
            rh = max(sh, key=lambda x: x[0])
            rl = min(sl, key=lambda x: x[0])
            if rh[0] < rl[0]:
                diff = rh[1] - rl[1]
                for fib, lbl in [(0.618, "FIB 0.618"), (0.705, "FIB 0.705"), (0.786, "FIB 0.786")]:
                    p2 = rl[1] + diff * fib
                    zones.append({'type': 'FIB', 'label': f'15M {lbl} 回撤阻力',
                        'high': p2 * 1.001, 'low': p2 * 0.999, 'mid': p2,
                        'direction': 'bearish',
                        'strength': 'strong' if fib in [0.618, 0.705] else 'medium'})

    # 6. EQH / EQL
    tol = 0.002
    for i in range(len(sh)):
        for j in range(i + 1, len(sh)):
            if abs(sh[i][1] - sh[j][1]) / sh[i][1] < tol:
                p2 = (sh[i][1] + sh[j][1]) / 2
                zones.append({'type': 'EQH', 'label': '15M Equal Highs（流動性池）',
                    'high': p2 * 1.002, 'low': p2 * 0.998, 'mid': p2,
                    'direction': 'bearish', 'strength': 'strong'})
    for i in range(len(sl)):
        for j in range(i + 1, len(sl)):
            if abs(sl[i][1] - sl[j][1]) / sl[i][1] < tol:
                p2 = (sl[i][1] + sl[j][1]) / 2
                zones.append({'type': 'EQL', 'label': '15M Equal Lows（流動性池）',
                    'high': p2 * 1.002, 'low': p2 * 0.998, 'mid': p2,
                    'direction': 'bullish', 'strength': 'strong'})

    # 7. Breaker Blocks
    for z in [x for x in zones if x['type'] == 'OB']:
        if z['direction'] == 'bullish' and lc < z['low']:
            zones.append({**z, 'type': 'Breaker',
                'label': '15M 看跌 Breaker Block', 'direction': 'bearish'})
        elif z['direction'] == 'bearish' and lc > z['high']:
            zones.append({**z, 'type': 'Breaker',
                'label': '15M 看漲 Breaker Block', 'direction': 'bullish'})

    # 8. Liquidity Sweep Detection
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
            sweep_dir = 'bearish' if z['direction'] == 'bearish' else 'bullish'
            arrow = 'down' if sweep_dir == 'bearish' else 'up'
            label = f"Liquidity Sweep {arrow} ({z['type']})"
            zones.append({'type': 'Sweep', 'label': label,
                'high': z['high'], 'low': z['low'], 'mid': z['mid'],
                'direction': sweep_dir, 'strength': 'strong',
                'reversed': reversed_after})

    # 9. Cross-TF SNR from 1H
    if df_1h is not None and len(df_1h) >= 10:
        sh1h, sl1h = find_swing_points(df_1h.iloc[-50:].reset_index(drop=True), n=3)
        for _, price in sh1h[-4:]:
            zones.append({'type': 'HTF_SNR', 'label': '1H 阻力（跨時間框架）',
                'high': price * 1.002, 'low': price * 0.998, 'mid': price,
                'direction': 'bearish', 'strength': 'strong'})
        for _, price in sl1h[-4:]:
            zones.append({'type': 'HTF_SNR', 'label': '1H 支撐（跨時間框架）',
                'high': price * 1.002, 'low': price * 0.998, 'mid': price,
                'direction': 'bullish', 'strength': 'strong'})

    # 10. PDH / PDL / PWH / PWL / Daily Open / Weekly Open
    if dw_levels:
        mapping = [
            ('PDH', '前日高點 PDH'),
            ('PDL', '前日低點 PDL'),
            ('PWH', '前週高點 PWH'),
            ('PWL', '前週低點 PWL'),
            ('DO',  '今日開盤 DO'),
            ('WO',  '本週開盤 WO'),
        ]
        for key, label in mapping:
            if key in dw_levels:
                price = dw_levels[key]
                actual_dir = 'bearish' if lc > price else 'bullish'
                strength = 'strong' if key in ['PDH', 'PDL', 'PWH', 'PWL'] else 'medium'
                zones.append({'type': key, 'label': label,
                    'high': price * 1.001, 'low': price * 0.999, 'mid': price,
                    'direction': actual_dir, 'strength': strength})

    filtered = [z for z in zones if z['direction'] == direction]
    deduped = []
    for z in filtered:
        if not any(abs(z['mid'] - d['mid']) / max(d['mid'], 0.001) < 0.003 for d in deduped):
            deduped.append(z)
    deduped.sort(key=lambda z: abs(z['mid'] - lc))
    return deduped

def find_tp_zone(price, direction, df_15m, df_1h=None, dw_levels=None):
    opp = "bearish" if direction == "bullish" else "bullish"
    zones = find_key_zones(df_15m, opp, df_1h=df_1h, dw_levels=dw_levels)
    if not zones:
        return None
    if direction == "bullish":
        above = [z for z in zones if z['mid'] > price]
        return min(above, key=lambda z: z['mid']) if above else None
    else:
        below = [z for z in zones if z['mid'] < price]
        return max(below, key=lambda z: z['mid']) if below else None

def assess_tp_strength(tp_zone):
    if not tp_zone:
        return "未知"
    if tp_zone['type'] in ['OB', 'EQH', 'EQL', 'Breaker', 'HTF_SNR', 'PWH', 'PWL', 'PDH', 'PDL'] \
            or tp_zone.get('strength') == 'strong':
        return "強（謹慎，價格可能提前反轉）"
    return "中等"

def calc_sl_tp(entry, zone, direction, df_15m, tp_zone=None):
    atr = calc_atr(df_15m, 14) or (entry * 0.003)
    buffer = 0.5 * atr
    if direction == "bullish":
        sl   = zone['low'] - buffer
        risk = entry - sl
        if tp_zone and risk > 0:
            tp = tp_zone['low'] * 0.999
            if (tp - entry) / risk < 1.5:
                tp = entry + risk * 2
        else:
            tp = entry + (risk * 2 if risk > 0 else entry * 0.02)
    else:
        sl   = zone['high'] + buffer
        risk = sl - entry
        if tp_zone and risk > 0:
            tp = tp_zone['high'] * 1.001
            if (entry - tp) / risk < 1.5:
                tp = entry - risk * 2
        else:
            tp = entry - (risk * 2 if risk > 0 else entry * 0.02)
    return round(sl, 4), round(tp, 4)

def check_1m_structure(df_1m, direction):
    if df_1m is None or len(df_1m) < 10:
        return "none", 0
    recent = df_1m.iloc[-15:]
    cp = float(recent.iloc[-1]['close'])
    sh, sl = find_swing_points(recent, n=2)
    if direction == "bullish":
        if not sh:
            return "none", 0
        sorted_h = sorted(sh, key=lambda x: x[0])
        if cp > sorted_h[-1][1]:
            if len(sorted_h) > 1 and cp > max(h[1] for h in sorted_h):
                return "bos", cp
            return "choch", cp
    else:
        if not sl:
            return "none", 0
        sorted_l = sorted(sl, key=lambda x: x[0])
        if cp < sorted_l[-1][1]:
            if len(sorted_l) > 1 and cp < min(l[1] for l in sorted_l):
                return "bos", cp
            return "choch", cp
    return "none", 0

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

async def send_msg(app, chat_id, text):
    try:
        await app.bot.send_message(chat_id=chat_id, text=text, parse_mode='HTML')
    except Exception as e:
        logger.error(f"Telegram send failed: {e}")

def fp(p):
    if p > 1000:
        return f"{p:,.2f}"
    elif p > 10:
        return f"{p:.3f}"
    else:
        return f"{p:.4f}"

def de(d):
    return "up 看漲" if d == "bullish" else "dn 看跌"

async def auto_scan(app, chat_id):
    logger.info("Auto scan started v2.0")
    await send_msg(app, chat_id,
        "OK <b>ICT Signal Bot v2.0 Started</b>\n\n"
        "Monitor: BTC / ETH / SOL\n"
        "Strategy: 4H+1H direction - 15M full key zones - 1M CHoCH/BOS\n"
        "New: IFVG / PDH/PDL / PWH/PWL / Daily+Weekly Open / Liquidity Sweep / Cross-TF SNR\n"
        "SL: OB outer edge + 0.5xATR(14) buffer\n"
        "Scan every 60s (Kill Zone restriction removed)\n"
        "Data: Binance public API"
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
                    d1m  = get_klines(sym, "1m",  20)
                    if any(x is None for x in [d4h, d1h, d15m, d1m]):
                        logger.warning(f"Skip {sym} data fetch failed")
                        continue

                    dw = get_daily_weekly_levels(sym)
                    cp   = float(d1m.iloc[-1]['close'])
                    dir4 = get_direction(d4h, 10)
                    dir1 = get_direction(d1h, 10)
                    if not dir4 or not dir1:
                        continue

                    st   = signal_states[sym]
                    now  = time.time()
                    dsym = sym.replace("USDT", "/USDT")
                    inkz, kzn = in_kill_zone()
                    kzs  = f"KZ: {kzn}" if inkz else "Non-KZ"

                    if st["active_zone"] and st["state"] == 1:
                        z = st["active_zone"]
                        if cp < z['low'] * 0.995 or cp > z['high'] * 1.005:
                            st.update({"state": 0, "active_zone": None, "order_id": None})

                    if st["state"] == 0 and now - st["last_signal_time"] < 7200:
                        continue

                    zones = find_key_zones(d15m, dir1, df_1h=d1h, dw_levels=dw)
                    az = next((z for z in zones if z['low'] * 0.999 <= cp <= z['high'] * 1.001), None)

                    if st["state"] == 0 and az:
                        if dir1 == "bullish":
                            guide_in  = f"站穩 {fp(az['high'])} 以上 - 考慮做多"
                            guide_out = f"跌破 {fp(az['low'])} - 關鍵區失守"
                        else:
                            guide_in  = f"站穩 {fp(az['low'])} 以下 - 考慮做空"
                            guide_out = f"升破 {fp(az['high'])} - 關鍵區失守"

                        align = ("4H 同 1H 方向一致（強信號）" if dir4 == dir1
                                 else "1H 逆 4H 回調（目標 4H 關鍵區）")

                        sweep_note = ""
                        sweep_zones = [z for z in zones if z['type'] == 'Sweep']
                        if sweep_zones:
                            sz = sweep_zones[0]
                            sweep_note = f"\nLiquidity Sweep: {sz['label']}"
                            if sz.get('reversed'):
                                sweep_note += "\nSweep reversed - high confidence"

                        dir4_txt = "up 看漲" if dir4 == "bullish" else "dn 看跌"
                        dir1_txt = "up 看漲" if dir1 == "bullish" else "dn 看跌"

                        await send_msg(app, chat_id,
                            f"WARNING <b>[Alert] {dsym}</b>\n"
                            f"---\n"
                            f"4H: {dir4_txt}  |  1H: {dir1_txt}\n"
                            f"{align}\n\n"
                            f"Zone: {az['label']}\n"
                            f"Range: {fp(az['low'])} - {fp(az['high'])}\n"
                            f"Price: {fp(cp)}\n"
                            f"{sweep_note}\n\n"
                            f"Guide:\n"
                            f"- {guide_in}\n"
                            f"- {guide_out}\n\n"
                            f"{kzs}  Waiting 1M CHoCH..."
                        )
                        st.update({"state": 1, "active_zone": az,
                                   "direction": dir1, "last_signal_time": now})

                    elif st["state"] == 1 and az:
                        stype, _ = check_1m_structure(d1m, st["direction"])
                        if stype == "choch":
                            oid  = generate_order_id(sym, st["direction"])
                            tpz  = find_tp_zone(cp, st["direction"], d15m, df_1h=d1h, dw_levels=dw)
                            sl, tp = calc_sl_tp(cp, st["active_zone"], st["direction"], d15m, tpz)
                            tps  = assess_tp_strength(tpz)
                            risk = abs(cp - sl)
                            rr   = abs(tp - cp) / risk if risk > 0 else 0
                            ds   = "LONG (做多)" if st["direction"] == "bullish" else "SHORT (做空)"
                            tpl  = tpz['label'] if tpz else "No clear zone (using 1:2 RR)"
                            atr_val = calc_atr(d15m, 14) or 0
                            dir4_txt = "up 看漲" if dir4 == "bullish" else "dn 看跌"
                            dir1_txt = "up 看漲" if dir1 == "bullish" else "dn 看跌"
                            await send_msg(app, chat_id,
                                f"SIGNAL <b>[Entry] {dsym}</b>\n"
                                f"---\n"
                                f"Order: <code>{oid}</code>\n\n"
                                f"Confirmed:\n"
                                f"- 4H: {dir4_txt} | 1H: {dir1_txt}\n"
                                f"- {kzs}\n"
                                f"- 1M CHoCH confirmed\n"
                                f"- Zone: {st['active_zone']['label']}\n\n"
                                f"Direction: {ds}\n\n"
                                f"Entry: {fp(cp)}\n"
                                f"SL: {fp(sl)}\n"
                                f"   OB edge + 0.5xATR ({fp(atr_val)})\n"
                                f"TP: {fp(tp)}\n"
                                f"   Target: {tpl}\n"
                                f"   TP strength: {tps}\n"
                                f"   RR: 1:{rr:.1f}\n\n"
                                f"Confirm risk before entry"
                            )
                            active_orders[oid] = {
                                "symbol": sym, "direction": st["direction"],
                                "entry": cp, "sl": sl, "tp": tp, "tp_zone": tpz,
                                "state": "open", "time": now, "tp_alerted": False,
                            }
                            st.update({"state": 2, "order_id": oid, "last_signal_time": now})

                    elif st["state"] == 2:
                        oid = st["order_id"]
                        if oid and oid in active_orders:
                            o = active_orders[oid]
                            stype, _ = check_1m_structure(d1m, st["direction"])
                            if stype == "bos":
                                await send_msg(app, chat_id,
                                    f"CONFIRM <b>[BOS Confirmed] {dsym}</b>\n"
                                    f"---\n"
                                    f"Order: <code>{oid}</code>\n\n"
                                    f"1M BOS structure confirmed\n"
                                    f"Price: {fp(cp)}\n\n"
                                    f"Trend confirmed - consider adding or holding\n"
                                    f"Tip: Move SL to breakeven {fp(o['entry'])}"
                                )
                                st.update({"state": 3, "last_signal_time": now})

                            if not o.get("tp_alerted"):
                                span = abs(o["tp"] - o["entry"])
                                if span > 0 and abs(cp - o["tp"]) / span < 0.15:
                                    pat, pdir = check_5m_pattern(d5m)
                                    opp = "bearish" if o["direction"] == "bullish" else "bullish"
                                    if pat != "普通" and pdir == opp:
                                        await send_msg(app, chat_id,
                                            f"ALERT <b>[Early TP Warning] {dsym}</b>\n"
                                            f"---\n"
                                            f"Order: <code>{oid}</code>\n\n"
                                            f"5M pattern at TP zone: <b>{pat}</b>\n"
                                            f"TP target: {fp(o['tp'])}\n"
                                            f"Price: {fp(cp)}\n\n"
                                            f"Consider early exit or hedge"
                                        )
                                        o["tp_alerted"] = True
                                    elif o.get("tp_zone") and o["tp_zone"].get("strength") == "strong":
                                        await send_msg(app, chat_id,
                                            f"HOLD <b>[Position Alert] {dsym}</b>\n"
                                            f"---\n"
                                            f"Order: <code>{oid}</code>\n\n"
                                            f"Near strong TP zone: {o['tp_zone']['label']}\n"
                                            f"Price: {fp(cp)}\n"
                                            f"TP: {fp(o['tp'])}\n\n"
                                            f"Consider moving SL to breakeven or early exit"
                                        )
                                        o["tp_alerted"] = True

                    if st["state"] in [1, 2]:
                        pat, pdir = check_5m_pattern(d5m)
                        if (pat != "普通" and pdir == st.get("direction") and
                                now - st["last_signal_time"] > 300):
                            oid = st.get("order_id", "")
                            os2 = f"\nOrder: <code>{oid}</code>" if oid else ""
                            await send_msg(app, chat_id,
                                f"CANDLE <b>[5M Pattern] {dsym}</b>\n"
                                f"---\n"
                                f"{os2}\n"
                                f"Pattern: <b>{pat}</b>\n"
                                f"Price: {fp(cp)}\n"
                                f"Use as additional entry confirmation"
                            )
                            st["last_signal_time"] = now

                except Exception as e:
                    logger.error(f"Scan {sym} failed: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"Main loop failed: {e}", exc_info=True)

        hb += 1
        if hb >= 60:
            hb = 0
            nows = datetime.now(HKT).strftime('%H:%M')
            inkz, kzn = in_kill_zone()
            kzs = f"KZ: {kzn}" if inkz else "Non-KZ"

            report_lines = [
                f"CLOCK <b>Hourly Report [{nows} HKT]</b>",
                f"---",
                f"{kzs}",
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
                        report_lines.append(f"{rsym.replace('USDT','/USDT')}: data error")
                        continue
                    rcp    = float(rd1m.iloc[-1]['close'])
                    rdir4  = get_direction(rd4h, 10)
                    rdir1  = get_direction(rd1h, 10)
                    rzones = find_key_zones(rd15m, rdir1, df_1h=rd1h, dw_levels=rdw) if rdir1 else []
                    dsym2  = rsym.replace('USDT', '/USDT')
                    d4s    = ("up 看漲" if rdir4 == "bullish" else "dn 看跌") if rdir4 else "N/A"
                    d1s    = ("up 看漲" if rdir1 == "bullish" else "dn 看跌") if rdir1 else "N/A"
                    report_lines.append(f"<b>{dsym2}</b>  {fp(rcp)}")
                    report_lines.append(f"  4H {d4s} | 1H {d1s}")
                    if rdw:
                        pdh = rdw.get('PDH'); pdl = rdw.get('PDL')
                        pwh = rdw.get('PWH'); pwl = rdw.get('PWL')
                        if pdh and pdl:
                            report_lines.append(f"  PDH {fp(pdh)} / PDL {fp(pdl)}")
                        if pwh and pwl:
                            report_lines.append(f"  PWH {fp(pwh)} / PWL {fp(pwl)}")
                    if rzones:
                        above = [z for z in rzones if z['mid'] > rcp]
                        below = [z for z in rzones if z['mid'] <= rcp]
                        if below:
                            bz = max(below, key=lambda z: z['mid'])
                            report_lines.append(f"  Support: {fp(bz['low'])}-{fp(bz['high'])} [{bz['type']}]")
                        if above:
                            az2 = min(above, key=lambda z: z['mid'])
                            report_lines.append(f"  Resist:  {fp(az2['low'])}-{fp(az2['high'])} [{az2['type']}]")
                    else:
                        report_lines.append(f"  No key zones")
                    report_lines.append("")
                except Exception as e:
                    report_lines.append(f"{rsym.replace('USDT','/USDT')}: error {e}")

            await send_msg(app, chat_id, "\n".join(report_lines))

        await asyncio.sleep(SCAN_INTERVAL)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "ICT Signal Bot v2.0\n\n"
        "Monitor: BTC / ETH / SOL\n"
        "Strategy: 4H+1H - 15M full zones - 1M CHoCH/BOS\n"
        "New: IFVG / PDH/PDL / PWH/PWL / Liquidity Sweep / Cross-TF SNR\n"
        "Scan every 60s",
        parse_mode='HTML'
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Bot is scanning in background...")

def main():
    logger.info("Starting bot v2.0...")
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("Missing TELEGRAM env vars")
        return
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    async def start_scan(context):
        await auto_scan(app, TELEGRAM_CHAT_ID)

    app.job_queue.run_once(start_scan, when=0)
    logger.info("Bot v2.0 started")
    app.run_polling()

if __name__ == '__main__':
    main()
