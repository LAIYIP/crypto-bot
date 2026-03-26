#!/usr/bin/env python3
"""
ICT/SMC Crypto Trading Signal Bot
Data source: Binance public REST API (no auth, bypasses geo-restriction)
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
WATCH_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
SCAN_INTERVAL = 60
HKT = timezone(timedelta(hours=8))
BINANCE_BASE    = "https://api.binance.com"
BINANCE_BASE_US = "https://api.binance.us"
_active_base    = BINANCE_BASE

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

KILL_ZONES = [
    ("Asia Open",   1,  0,  3,  0),
    ("London Open",15,  0, 17,  0),
    ("NY Open",    21, 30, 23, 30),
]
KZ_NAMES = {
    "Asia Open":   "亞洲開市",
    "London Open": "倫敦開市",
    "NY Open":     "紐約開市",
}

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

def find_key_zones(df_15m, direction):
    zones = []
    if df_15m is None or len(df_15m) < 30:
        return zones
    r = df_15m.iloc[-30:].copy().reset_index(drop=True)
    n = len(r)

    for i in range(1, n-2):
        c, p, n1, n2 = r.iloc[i], r.iloc[i-1], r.iloc[i+1], r.iloc[i+2]
        if (c['close'] < c['open'] and n1['close'] > n1['open'] and
                n2['close'] > n2['open'] and n2['close'] > p['high']):
            zones.append({'type':'OB','label':'15M 看漲 OB（需求區）',
                'high':float(c['high']),'low':float(c['low']),
                'mid':float((c['high']+c['low'])/2),
                'direction':'bullish','strength':'strong'})
        if (c['close'] > c['open'] and n1['close'] < n1['open'] and
                n2['close'] < n2['open'] and n2['close'] < p['low']):
            zones.append({'type':'OB','label':'15M 看跌 OB（供應區）',
                'high':float(c['high']),'low':float(c['low']),
                'mid':float((c['high']+c['low'])/2),
                'direction':'bearish','strength':'strong'})

    for i in range(1, n-1):
        p, nk = r.iloc[i-1], r.iloc[i+1]
        if float(p['high']) < float(nk['low']):
            zones.append({'type':'FVG','label':'15M 看漲 FVG',
                'high':float(nk['low']),'low':float(p['high']),
                'mid':float((nk['low']+p['high'])/2),
                'direction':'bullish','strength':'medium'})
        if float(p['low']) > float(nk['high']):
            zones.append({'type':'FVG','label':'15M 看跌 FVG',
                'high':float(p['low']),'low':float(nk['high']),
                'mid':float((p['low']+nk['high'])/2),
                'direction':'bearish','strength':'medium'})

    sh, sl = find_swing_points(r, n=3)
    for _, price in sh[-3:]:
        zones.append({'type':'SNR','label':'15M 阻力位 SNR',
            'high':price*1.001,'low':price*0.999,'mid':price,
            'direction':'bearish','strength':'medium'})
    for _, price in sl[-3:]:
        zones.append({'type':'SNR','label':'15M 支撐位 SNR',
            'high':price*1.001,'low':price*0.999,'mid':price,
            'direction':'bullish','strength':'medium'})

    if sh and sl:
        if direction == "bullish":
            rl = min(sl, key=lambda x: x[0])
            rh = max(sh, key=lambda x: x[0])
            if rl[0] < rh[0]:
                diff = rh[1] - rl[1]
                for fib, lbl in [(0.618,"FIB 0.618"),(0.705,"FIB 0.705"),(0.786,"FIB 0.786")]:
                    p2 = rh[1] - diff * fib
                    zones.append({'type':'FIB','label':f'15M {lbl} 回撤支撐',
                        'high':p2*1.001,'low':p2*0.999,'mid':p2,'direction':'bullish',
                        'strength':'strong' if fib in [0.618,0.705] else 'medium'})
        else:
            rh = max(sh, key=lambda x: x[0])
            rl = min(sl, key=lambda x: x[0])
            if rh[0] < rl[0]:
                diff = rh[1] - rl[1]
                for fib, lbl in [(0.618,"FIB 0.618"),(0.705,"FIB 0.705"),(0.786,"FIB 0.786")]:
                    p2 = rl[1] + diff * fib
                    zones.append({'type':'FIB','label':f'15M {lbl} 回撤阻力',
                        'high':p2*1.001,'low':p2*0.999,'mid':p2,'direction':'bearish',
                        'strength':'strong' if fib in [0.618,0.705] else 'medium'})

    tol = 0.002
    for i in range(len(sh)):
        for j in range(i+1, len(sh)):
            if abs(sh[i][1]-sh[j][1])/sh[i][1] < tol:
                p2 = (sh[i][1]+sh[j][1])/2
                zones.append({'type':'EQH','label':'15M Equal Highs（流動性聚集）',
                    'high':p2*1.002,'low':p2*0.998,'mid':p2,
                    'direction':'bearish','strength':'strong'})
    for i in range(len(sl)):
        for j in range(i+1, len(sl)):
            if abs(sl[i][1]-sl[j][1])/sl[i][1] < tol:
                p2 = (sl[i][1]+sl[j][1])/2
                zones.append({'type':'EQL','label':'15M Equal Lows（流動性聚集）',
                    'high':p2*1.002,'low':p2*0.998,'mid':p2,
                    'direction':'bullish','strength':'strong'})

    lc = float(r.iloc[-1]['close'])
    for z in [x for x in zones if x['type'] == 'OB']:
        if z['direction'] == 'bullish' and lc < z['low']:
            zones.append({**z,'type':'Breaker','label':'15M 看跌 Breaker Block','direction':'bearish'})
        elif z['direction'] == 'bearish' and lc > z['high']:
            zones.append({**z,'type':'Breaker','label':'15M 看漲 Breaker Block','direction':'bullish'})

    filtered = [z for z in zones if z['direction'] == direction]
    deduped = []
    for z in filtered:
        if not any(abs(z['mid']-d['mid'])/d['mid'] < 0.003 for d in deduped):
            deduped.append(z)
    return deduped

def find_tp_zone(price, direction, df_15m):
    opp = "bearish" if direction == "bullish" else "bullish"
    zones = find_key_zones(df_15m, opp)
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
    if tp_zone['type'] in ['OB','EQH','EQL','Breaker'] or tp_zone.get('strength') == 'strong':
        return "強（謹慎，價格可能提前反轉）"
    return "中等"

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
    uw = float(c['high']) - max(float(c['close']), float(c['open']))
    lw = min(float(c['close']), float(c['open'])) - float(c['low'])
    pb = abs(float(p['close']) - float(p['open']))
    if body == 0:
        return "普通", None
    if float(c['open']) > float(p['close']) and float(c['close']) < float(p['open']) and body > pb:
        return "看跌吞沒", "bearish"
    if float(c['open']) < float(p['close']) and float(c['close']) > float(p['open']) and body > pb:
        return "看漲吞沒", "bullish"
    if uw > body*2 and lw < body*0.5 and float(c['close']) < float(c['open']):
        return "射擊之星", "bearish"
    if lw > body*2 and uw < body*0.5 and float(c['close']) > float(c['open']):
        return "錘子線", "bullish"
    if uw > body*3:
        return "Pin Bar（看跌）", "bearish"
    if lw > body*3:
        return "Pin Bar（看漲）", "bullish"
    return "普通", None

def calc_sl_tp(entry, zone, direction, tp_zone=None):
    if direction == "bullish":
        sl = zone['low'] * 0.998
        risk = entry - sl
        if tp_zone and risk > 0:
            tp = tp_zone['low'] * 0.999
            if (tp - entry) / risk < 1.5:
                tp = entry + risk * 2
        else:
            tp = entry + (risk * 2 if risk > 0 else entry * 0.02)
    else:
        sl = zone['high'] * 1.002
        risk = sl - entry
        if tp_zone and risk > 0:
            tp = tp_zone['high'] * 1.001
            if (entry - tp) / risk < 1.5:
                tp = entry - risk * 2
        else:
            tp = entry - (risk * 2 if risk > 0 else entry * 0.02)
    return round(sl, 4), round(tp, 4)

async def send_msg(app, chat_id, text):
    try:
        await app.bot.send_message(chat_id=chat_id, text=text, parse_mode='HTML')
    except Exception as e:
        logger.error(f"Telegram 發送失敗: {e}")

def fp(p):
    if p > 1000:
        return f"{p:,.2f}"
    elif p > 10:
        return f"{p:.3f}"
    else:
        return f"{p:.4f}"

def de(d):
    return "⬆️ 看漲" if d == "bullish" else "⬇️ 看跌"

async def auto_scan(app, chat_id):
    logger.info("自動掃描已啟動")
    await send_msg(app, chat_id,
        "✅ <b>ICT 交易信號機械人已啟動</b>\n\n"
        "📊 監控: BTC / ETH / SOL\n"
        "🎯 策略: 4H+1H方向 → 15M關鍵區 → Kill Zone → 1M CHoCH/BOS\n"
        "⏰ 每 60 秒掃描一次\n"
        "🌐 數據源: Binance 公開 API（無需認證）"
    )

    hb = 0
    while True:
        try:
            for sym in WATCH_SYMBOLS:
                try:
                    d4h  = get_klines(sym, "4h",  30)
                    d1h  = get_klines(sym, "1h",  20)
                    d15m = get_klines(sym, "15m", 40)
                    d5m  = get_klines(sym, "5m",  10)
                    d1m  = get_klines(sym, "1m",  20)
                    if any(x is None for x in [d4h, d1h, d15m, d1m]):
                        logger.warning(f"跳過 {sym}（數據獲取失敗）")
                        continue

                    cp   = float(d1m.iloc[-1]['close'])
                    dir4 = get_direction(d4h, 10)
                    dir1 = get_direction(d1h, 10)
                    if not dir4 or not dir1:
                        continue

                    st   = signal_states[sym]
                    now  = time.time()
                    dsym = sym.replace("USDT", "/USDT")

                    if st["active_zone"] and st["state"] == 1:
                        z = st["active_zone"]
                        if cp < z['low'] * 0.995 or cp > z['high'] * 1.005:
                            st.update({"state": 0, "active_zone": None, "order_id": None})

                    if st["state"] == 0 and now - st["last_signal_time"] < 7200:
                        continue

                    zones = find_key_zones(d15m, dir1)
                    az = next((z for z in zones if z['low']*0.999 <= cp <= z['high']*1.001), None)

                    if st["state"] == 0 and az:
                        if dir1 == "bullish":
                            gh = f"{fp(az['high'])} 以上站穩"
                            gb = f"跌破 {fp(az['low'])}"
                            ga = "考慮做多"
                            gx = "關鍵區失守，暫不入場"
                        else:
                            gh = f"{fp(az['low'])} 以下站穩"
                            gb = f"升破 {fp(az['high'])}"
                            ga = "考慮做空"
                            gx = "關鍵區失守，暫不入場"
                        aln = ("✅ 4H 同 1H 方向一致（強信號）" if dir4 == dir1
                               else "⚠️ 1H 逆 4H 回調（目標 4H 關鍵區）")
                        inkz, kzn = in_kill_zone()
                        kzs = f"✅ 現在係 Kill Zone ({kzn})" if inkz else "⏳ 等待 Kill Zone 時段..."
                        await send_msg(app, chat_id,
                            f"⚠️ <b>【留意信號】{dsym}</b>\n"
                            f"━━━━━━━━━━━━━━━━━━\n"
                            f"📊 <b>4H:</b> {de(dir4)}  |  <b>1H:</b> {de(dir1)}\n"
                            f"{aln}\n\n"
                            f"🎯 <b>關鍵區域:</b> {az['label']}\n"
                            f"📍 <b>區域範圍:</b> {fp(az['low'])} - {fp(az['high'])}\n"
                            f"💲 <b>當前價格:</b> {fp(cp)}\n\n"
                            f"📌 <b>操作指引:</b>\n"
                            f"• {gh} → {ga}\n"
                            f"• {gb} → {gx}\n\n"
                            f"⏰ {kzs}\n"
                            f"<i>等待 1M CHoCH 確認反轉...</i>"
                        )
                        st.update({"state": 1, "active_zone": az,
                                   "direction": dir1, "last_signal_time": now})

                    elif st["state"] == 1 and az:
                        inkz, kzn = in_kill_zone()
                        stype, _ = check_1m_structure(d1m, st["direction"])
                        if stype == "choch" and inkz:
                            oid = generate_order_id(sym, st["direction"])
                            tpz = find_tp_zone(cp, st["direction"], d15m)
                            sl, tp = calc_sl_tp(cp, st["active_zone"], st["direction"], tpz)
                            tps  = assess_tp_strength(tpz)
                            risk = abs(cp - sl)
                            rr   = abs(tp - cp) / risk if risk > 0 else 0
                            ds   = "🟢 做多 (Long)" if st["direction"] == "bullish" else "🔴 做空 (Short)"
                            tpl  = tpz['label'] if tpz else "無明確關鍵區（使用 1:2 RR）"
                            await send_msg(app, chat_id,
                                f"🚨 <b>【入場信號】{dsym}</b>\n"
                                f"━━━━━━━━━━━━━━━━━━\n"
                                f"📋 <b>訂單編號:</b> <code>{oid}</code>\n\n"
                                f"✅ <b>確認條件:</b>\n"
                                f"• 4H: {de(dir4)} | 1H: {de(dir1)}\n"
                                f"• Kill Zone: {kzn}\n"
                                f"• 1M CHoCH 反轉確認\n"
                                f"• 關鍵區: {st['active_zone']['label']}\n\n"
                                f"📈 <b>交易方向:</b> {ds}\n\n"
                                f"💵 <b>入場價格:</b> {fp(cp)}\n"
                                f"🛑 <b>止損 (SL):</b> {fp(sl)}\n"
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
                        elif stype == "choch" and not inkz:
                            await send_msg(app, chat_id,
                                f"⚡️ <b>【結構提示】{dsym}</b>\n"
                                f"━━━━━━━━━━━━━━━━━━\n"
                                f"1M CHoCH 出現，但現在不在 Kill Zone\n"
                                f"💲 當前價格: {fp(cp)}\n"
                                f"<i>等待 Kill Zone 時段再確認入場...</i>"
                            )

                    elif st["state"] == 2:
                        oid = st["order_id"]
                        if oid and oid in active_orders:
                            o = active_orders[oid]
                            stype, _ = check_1m_structure(d1m, st["direction"])
                            if stype == "bos":
                                await send_msg(app, chat_id,
                                    f"✅ <b>【確認信號】{dsym}</b>\n"
                                    f"━━━━━━━━━━━━━━━━━━\n"
                                    f"📋 <b>訂單編號:</b> <code>{oid}</code>\n\n"
                                    f"🔥 <b>1M BOS 突破結構確認</b>\n"
                                    f"💲 當前價格: {fp(cp)}\n\n"
                                    f"<i>趨勢已確認，可考慮加倉或持有</i>"
                                )
                                st.update({"state": 3, "last_signal_time": now})
                            if not o.get("tp_alerted"):
                                span = abs(o["tp"] - o["entry"])
                                if span > 0 and abs(cp - o["tp"]) / span < 0.15:
                                    pat, pdir = check_5m_pattern(d5m)
                                    opp = "bearish" if o["direction"] == "bullish" else "bullish"
                                    if pat != "普通" and pdir == opp:
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

                    if st["state"] in [1, 2]:
                        pat, pdir = check_5m_pattern(d5m)
                        if (pat != "普通" and pdir == st.get("direction") and
                                now - st["last_signal_time"] > 300):
                            oid = st.get("order_id", "")
                            os2 = f"\n📋 <b>訂單編號:</b> <code>{oid}</code>" if oid else ""
                            await send_msg(app, chat_id,
                                f"🕯️ <b>【5M 確認信號】{dsym}</b>\n"
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

        hb += 1
        if hb >= 60:
            hb = 0
            nows = datetime.now(HKT).strftime('%H:%M')
            inkz, kzn = in_kill_zone()
            kzs = f"🔴 Kill Zone: {kzn}" if inkz else "⚪ 非 Kill Zone 時段"

            report_lines = [
                f"🕐 <b>每小時市場快報</b> [{nows} HKT]",
                f"━━━━━━━━━━━━━━━━━━",
                f"⏰ {kzs}",
                ""
            ]
            for rsym in WATCH_SYMBOLS:
                try:
                    rd4h  = get_klines(rsym, "4h",  30)
                    rd1h  = get_klines(rsym, "1h",  20)
                    rd15m = get_klines(rsym, "15m", 40)
                    rd1m  = get_klines(rsym, "1m",   5)
                    if any(x is None for x in [rd4h, rd1h, rd15m, rd1m]):
                        report_lines.append(f"📌 <b>{rsym.replace('USDT','/USDT')}</b>: 數據獲取失敗")
                        continue
                    rcp   = float(rd1m.iloc[-1]['close'])
                    rdir4 = get_direction(rd4h, 10)
                    rdir1 = get_direction(rd1h, 10)
                    rzones = find_key_zones(rd15m, rdir1) if rdir1 else []
                    dsym2 = rsym.replace('USDT', '/USDT')
                    d4s = de(rdir4) if rdir4 else "N/A"
                    d1s = de(rdir1) if rdir1 else "N/A"
                    report_lines.append(f"📌 <b>{dsym2}</b>  💲{fp(rcp)}")
                    report_lines.append(f"   4H {d4s} | 1H {d1s}")
                    if rzones:
                        above = [z for z in rzones if z['mid'] > rcp]
                        below = [z for z in rzones if z['mid'] <= rcp]
                        if below:
                            bz = max(below, key=lambda z: z['mid'])
                            report_lines.append(f"   🟢 支撐: {fp(bz['low'])} - {fp(bz['high'])}  ({bz['type']})")
                        if above:
                            az2 = min(above, key=lambda z: z['mid'])
                            report_lines.append(f"   🔴 阻力: {fp(az2['low'])} - {fp(az2['high'])}  ({az2['type']})")
                    else:
                        report_lines.append(f"   ⚪ 暫無關鍵區")
                    report_lines.append("")
                except Exception as e:
                    report_lines.append(f"📌 <b>{rsym.replace('USDT','/USDT')}</b>: 錯誤 {e}")

            await send_msg(app, chat_id, "\n".join(report_lines))

        await asyncio.sleep(SCAN_INTERVAL)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 <b>歡迎使用 ICT 交易信號機械人</b>\n\n"
        "📊 監控: BTC / ETH / SOL\n"
        "🎯 策略: 4H+1H → 15M關鍵區 → Kill Zone → 1M CHoCH/BOS\n"
        "⏰ 每 60 秒掃描一次",
        parse_mode='HTML'
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("機械人正在後台自動掃描市場中...")

def main():
    logger.info("正在啟動機械人...")
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("缺少 TELEGRAM 環境變量")
        return
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    async def start_scan(context):
        await auto_scan(app, TELEGRAM_CHAT_ID)

    app.job_queue.run_once(start_scan, when=0)
    logger.info("✅ 機械人已啟動")
    app.run_polling()

if __name__ == '__main__':
    main()
