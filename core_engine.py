"""
Bot v6.0 核心分析引擎
主框架：15M 關鍵位識別 + 重疊計分系統
方向判斷：1H（主力）+ 4H（加分）
入場確認：3M MSS
"""

import time
import requests
import numpy as np
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional

HKT = timezone(timedelta(hours=8))

# ─────────────────────────────────────────
# 資料拉取
# ─────────────────────────────────────────

# Bybit interval mapping (Binance format -> Bybit format)
_BYBIT_INTERVAL_MAP = {
    "1m": "1", "3m": "3", "5m": "5", "15m": "15", "30m": "30",
    "1h": "60", "2h": "120", "4h": "240", "6h": "360", "12h": "720",
    "1d": "D", "1w": "W",
}


def fetch_klines(symbol: str, interval: str, limit: int = 300) -> list[dict]:
    """從 Bybit 拉取 K 線數據（相容 US 地區，不受 Binance 封鎖）"""
    bybit_interval = _BYBIT_INTERVAL_MAP.get(interval, interval)
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": bybit_interval,
        "limit": limit,
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        if data.get("retCode") != 0:
            return []
        items = data["result"]["list"]
        if not items:
            return []
        # Bybit 返回最新在前（降序），需反轉為升序
        return [
            {
                "ts": int(d[0]),
                "open": float(d[1]),
                "high": float(d[2]),
                "low": float(d[3]),
                "close": float(d[4]),
                "volume": float(d[5]),
            }
            for d in reversed(items)
        ]
    except Exception:
        return []


def get_current_price(symbol: str) -> float:
    """取得現價（Bybit）"""
    try:
        r = requests.get(
            "https://api.bybit.com/v5/market/tickers",
            params={"category": "linear", "symbol": symbol},
            timeout=5,
        )
        data = r.json()
        if data.get("retCode") == 0:
            return float(data["result"]["list"][0]["lastPrice"])
        return 0.0
    except Exception:
        return 0.0


# ─────────────────────────────────────────
# ATR 計算
# ─────────────────────────────────────────

def calc_atr(klines: list[dict], period: int = 14) -> float:
    """計算 ATR（14 期）"""
    if len(klines) < period + 1:
        return 0.0
    trs = []
    for i in range(1, len(klines)):
        h = klines[i]["high"]
        l = klines[i]["low"]
        pc = klines[i - 1]["close"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return float(np.mean(trs[-period:]))


# ─────────────────────────────────────────
# 市場結構判斷
# ─────────────────────────────────────────

def get_swing_points(klines: list[dict], lookback: int = 20) -> dict:
    """找最近的 Swing High / Swing Low"""
    recent = klines[-lookback:]
    swing_high = max(k["high"] for k in recent)
    swing_low = min(k["low"] for k in recent)
    return {"swing_high": swing_high, "swing_low": swing_low}


def get_market_structure(klines: list[dict], lookback: int = 30) -> str:
    """
    判斷市場結構：bullish / bearish / ranging
    用最近 lookback 根 K 線的 Swing High / Low 判斷
    """
    if len(klines) < lookback + 5:
        return "ranging"

    recent = klines[-lookback:]
    highs = [k["high"] for k in recent]
    lows = [k["low"] for k in recent]

    # 找局部高低點（簡化版：分三段比較）
    seg = lookback // 3
    h1 = max(highs[:seg])
    h2 = max(highs[seg : seg * 2])
    h3 = max(highs[seg * 2 :])
    l1 = min(lows[:seg])
    l2 = min(lows[seg : seg * 2])
    l3 = min(lows[seg * 2 :])

    bullish = h3 > h2 > h1 and l3 > l2 > l1
    bearish = h3 < h2 < h1 and l3 < l2 < l1

    if bullish:
        return "bullish"
    elif bearish:
        return "bearish"
    else:
        return "ranging"


# ─────────────────────────────────────────
# 關鍵水平計算
# ─────────────────────────────────────────

def get_key_levels(klines_1h: list[dict], klines_4h: list[dict]) -> dict:
    """計算 PDH/PDL/DO/WO/PWH/PWL/BSL/SSL"""
    now_hkt = datetime.now(HKT)

    # 今日開盤（DO）
    today_start = now_hkt.replace(hour=0, minute=0, second=0, microsecond=0)
    today_ts = int(today_start.timestamp() * 1000)
    do = next(
        (k["open"] for k in klines_1h if k["ts"] >= today_ts),
        klines_1h[-1]["open"] if klines_1h else 0,
    )

    # 本週開盤（WO）
    week_start = today_start - timedelta(days=now_hkt.weekday())
    week_ts = int(week_start.timestamp() * 1000)
    wo = next(
        (k["open"] for k in klines_1h if k["ts"] >= week_ts),
        klines_1h[-1]["open"] if klines_1h else 0,
    )

    # 前日高低（PDH/PDL）
    yesterday_start = today_start - timedelta(days=1)
    yesterday_end = today_start
    yd_ts_start = int(yesterday_start.timestamp() * 1000)
    yd_ts_end = int(yesterday_end.timestamp() * 1000)
    yd_klines = [k for k in klines_1h if yd_ts_start <= k["ts"] < yd_ts_end]
    pdh = max((k["high"] for k in yd_klines), default=0)
    pdl = min((k["low"] for k in yd_klines), default=0)

    # 前週高低（PWH/PWL）
    prev_week_start = week_start - timedelta(weeks=1)
    pw_ts_start = int(prev_week_start.timestamp() * 1000)
    pw_ts_end = int(week_start.timestamp() * 1000)
    pw_klines = [k for k in klines_1h if pw_ts_start <= k["ts"] < pw_ts_end]
    pwh = max((k["high"] for k in pw_klines), default=0)
    pwl = min((k["low"] for k in pw_klines), default=0)

    # BSL / SSL（最近 1H Swing High/Low）
    swings_1h = get_swing_points(klines_1h, lookback=20)
    bsl = swings_1h["swing_high"]
    ssl = swings_1h["swing_low"]

    return {
        "do": do, "wo": wo,
        "pdh": pdh, "pdl": pdl,
        "pwh": pwh, "pwl": pwl,
        "bsl": bsl, "ssl": ssl,
    }


# ─────────────────────────────────────────
# OB 識別
# ─────────────────────────────────────────

@dataclass
class OrderBlock:
    direction: str        # "bullish" / "bearish"
    high: float
    low: float
    mid: float
    ts: int               # 形成時間戳
    timeframe: str        # "15m" / "1h" / "4h"
    strength: float = 1.0 # 1.0 = 正常, 0.5 = 弱化, 0.0 = 失效
    is_sweep: bool = False  # 假突破後回收


def detect_obs(klines: list[dict], timeframe: str, lookback: int = 50) -> list[OrderBlock]:
    """
    識別 Order Block
    規則：最後一根反向蠟燭前的實體
    """
    obs = []
    recent = klines[-lookback:] if len(klines) >= lookback else klines

    for i in range(2, len(recent) - 1):
        curr = recent[i]
        prev = recent[i - 1]
        nxt = recent[i + 1]

        # 看漲 OB：下跌蠟燭後緊接上漲突破
        if (prev["close"] < prev["open"] and  # 前一根係陰線
                curr["close"] > curr["open"] and  # 當前係陽線
                curr["close"] > prev["high"]):  # 突破前一根高點
            ob = OrderBlock(
                direction="bullish",
                high=prev["high"],
                low=prev["low"],
                mid=(prev["high"] + prev["low"]) / 2,
                ts=prev["ts"],
                timeframe=timeframe,
            )
            obs.append(ob)

        # 看跌 OB：上漲蠟燭後緊接下跌突破
        if (prev["close"] > prev["open"] and  # 前一根係陽線
                curr["close"] < curr["open"] and  # 當前係陰線
                curr["close"] < prev["low"]):  # 跌破前一根低點
            ob = OrderBlock(
                direction="bearish",
                high=prev["high"],
                low=prev["low"],
                mid=(prev["high"] + prev["low"]) / 2,
                ts=prev["ts"],
                timeframe=timeframe,
            )
            obs.append(ob)

    return obs


def update_ob_validity(obs: list[OrderBlock], current_price: float, klines: list[dict]) -> list[OrderBlock]:
    """
    更新 OB 有效性：
    - 穿越 50% → 弱化（strength = 0.5）
    - 完全穿越 → 失效（strength = 0.0）
    - 假突破（極速刺穿後 3M 收回）→ 有效，is_sweep = True
    """
    valid_obs = []
    for ob in obs:
        ob_range = ob.high - ob.low
        if ob_range <= 0:
            continue

        if ob.direction == "bullish":
            penetration = ob.high - current_price
            if current_price < ob.low:
                # 完全穿越 → 失效
                ob.strength = 0.0
            elif current_price < ob.mid:
                # 穿越超過 50% → 弱化
                ob.strength = 0.5
            # 假突破檢查：若最近 3 根 K 線刺穿後收回
            recent_lows = [k["low"] for k in klines[-3:]]
            recent_closes = [k["close"] for k in klines[-3:]]
            if any(l < ob.low for l in recent_lows) and all(c > ob.low for c in recent_closes):
                ob.is_sweep = True
                ob.strength = max(ob.strength, 1.0)  # 假突破視為有效

        elif ob.direction == "bearish":
            if current_price > ob.high:
                # 完全穿越 → 失效
                ob.strength = 0.0
            elif current_price > ob.mid:
                # 穿越超過 50% → 弱化
                ob.strength = 0.5
            # 假突破檢查
            recent_highs = [k["high"] for k in klines[-3:]]
            recent_closes = [k["close"] for k in klines[-3:]]
            if any(h > ob.high for h in recent_highs) and all(c < ob.high for c in recent_closes):
                ob.is_sweep = True
                ob.strength = max(ob.strength, 1.0)

        if ob.strength > 0:
            valid_obs.append(ob)

    return valid_obs


# ─────────────────────────────────────────
# FVG 識別
# ─────────────────────────────────────────

@dataclass
class FVG:
    direction: str   # "bullish" / "bearish"
    high: float
    low: float
    mid: float
    ts: int
    timeframe: str


def detect_fvgs(klines: list[dict], timeframe: str, lookback: int = 50) -> list[FVG]:
    """識別 Fair Value Gap（三根蠟燭之間的缺口）"""
    fvgs = []
    recent = klines[-lookback:] if len(klines) >= lookback else klines

    for i in range(1, len(recent) - 1):
        k1 = recent[i - 1]
        k3 = recent[i + 1]

        # 看漲 FVG：k1 高點 < k3 低點
        if k1["high"] < k3["low"]:
            fvgs.append(FVG(
                direction="bullish",
                high=k3["low"],
                low=k1["high"],
                mid=(k3["low"] + k1["high"]) / 2,
                ts=recent[i]["ts"],
                timeframe=timeframe,
            ))

        # 看跌 FVG：k1 低點 > k3 高點
        if k1["low"] > k3["high"]:
            fvgs.append(FVG(
                direction="bearish",
                high=k1["low"],
                low=k3["high"],
                mid=(k1["low"] + k3["high"]) / 2,
                ts=recent[i]["ts"],
                timeframe=timeframe,
            ))

    return fvgs


# ─────────────────────────────────────────
# FIB 計算
# ─────────────────────────────────────────

def calc_fib(klines: list[dict], direction: str, lookback: int = 50) -> dict:
    """
    計算 FIB 水平
    direction = "bullish" → 從 Swing Low 到 Swing High（做多折扣區在 0.5 以下）
    direction = "bearish" → 從 Swing High 到 Swing Low（做空溢價區在 0.5 以上）
    """
    recent = klines[-lookback:] if len(klines) >= lookback else klines
    swing_high = max(k["high"] for k in recent)
    swing_low = min(k["low"] for k in recent)
    diff = swing_high - swing_low

    if diff <= 0:
        return {}

    levels = {}
    for ratio in [0.0, 0.236, 0.382, 0.5, 0.618, 0.705, 0.786, 1.0]:
        if direction == "bullish":
            levels[str(ratio)] = swing_high - diff * ratio
        else:
            levels[str(ratio)] = swing_low + diff * ratio

    levels["swing_high"] = swing_high
    levels["swing_low"] = swing_low
    return levels


# ─────────────────────────────────────────
# EQH / EQL 識別
# ─────────────────────────────────────────

def find_eqh_eql(klines: list[dict], tolerance: float = 0.002) -> dict:
    """
    找等高點（EQH）和等低點（EQL）
    tolerance = 0.2% 內視為等高/等低
    """
    recent = klines[-50:] if len(klines) >= 50 else klines
    highs = [k["high"] for k in recent]
    lows = [k["low"] for k in recent]

    # EQH：找兩個相近的高點
    eqh = None
    for i in range(len(highs) - 1):
        for j in range(i + 1, len(highs)):
            if abs(highs[i] - highs[j]) / highs[i] < tolerance:
                eqh = (highs[i] + highs[j]) / 2
                break
        if eqh:
            break

    # EQL：找兩個相近的低點
    eql = None
    for i in range(len(lows) - 1):
        for j in range(i + 1, len(lows)):
            if abs(lows[i] - lows[j]) / lows[i] < tolerance:
                eql = (lows[i] + lows[j]) / 2
                break
        if eql:
            break

    return {"eqh": eqh, "eql": eql}


# ─────────────────────────────────────────
# 重疊計分系統（核心）
# ─────────────────────────────────────────

@dataclass
class KeyZone:
    """代表一個帶有重疊分數的關鍵區域"""
    price: float          # 代表價格（中點）
    high: float           # 區域上邊界
    low: float            # 區域下邊界
    score: float          # 重疊分數
    direction: str        # "bullish" / "bearish" / "neutral"
    labels: list[str] = field(default_factory=list)  # 組成此區域的關鍵位標籤
    is_in_discount: bool = False   # 是否在折扣區（FIB 0.5 以下）
    is_in_1h_range: bool = False   # 是否在 1H 高低點範圍內
    timeframe_primary: str = "15m"


def score_key_zones(
    current_price: float,
    direction: str,  # "bullish" / "bearish"
    obs_15m: list[OrderBlock],
    obs_1h: list[OrderBlock],
    obs_4h: list[OrderBlock],
    fvgs_15m: list[FVG],
    fvgs_1h: list[FVG],
    fib: dict,
    key_levels: dict,
    eqh_eql: dict,
    klines_15m: list[dict],
    now_ts: int,
) -> list[KeyZone]:
    """
    為所有 15M 關鍵位計算重疊分數
    主框架：15M
    加分：1H OB / 4H OB / FIB / 關鍵水平 / FVG / EQH/EQL / 時效性
    """
    zones: list[KeyZone] = []
    one_day_ms = 24 * 3600 * 1000

    # 1H Swing High/Low 範圍
    swings_1h_range = get_swing_points(klines_15m, lookback=60)  # 用 15M 最近 60 根近似 1H 範圍
    range_high = swings_1h_range["swing_high"]
    range_low = swings_1h_range["swing_low"]

    # 折扣區邊界
    discount_boundary = fib.get("0.5", current_price)

    # 建立候選位置（以 15M OB 為基礎）
    candidate_obs = [ob for ob in obs_15m if ob.strength > 0]

    # 若 15M OB 太少，補充 1H OB 的中點作為候選
    if len(candidate_obs) < 3:
        for ob in obs_1h:
            if ob.strength > 0:
                # 創建一個虛擬 15M OB
                candidate_obs.append(OrderBlock(
                    direction=ob.direction,
                    high=ob.high,
                    low=ob.low,
                    mid=ob.mid,
                    ts=ob.ts,
                    timeframe="1h",
                    strength=ob.strength,
                ))

    for ob in candidate_obs:
        if ob.direction != direction:
            continue  # 只看與方向一致的 OB

        score = 0.0
        labels = []

        # 基礎分：15M OB = 1 分，1H OB = 1.5 分，4H OB = 2 分
        if ob.timeframe == "15m":
            score += 1.0
            labels.append("15M OB")
        elif ob.timeframe == "1h":
            score += 1.5
            labels.append("1H OB")
        elif ob.timeframe == "4h":
            score += 2.0
            labels.append("4H OB")

        # 弱化扣分
        if ob.strength == 0.5:
            score *= 0.5
        elif ob.is_sweep:
            score += 0.5
            labels.append("Sweep")

        # 時效性加分：24 小時內形成
        if now_ts - ob.ts < one_day_ms:
            score += 0.5
            labels.append("新鮮")

        # 加分：1H OB 包含此位置
        for ob1h in obs_1h:
            if ob1h.direction == direction and ob1h.strength > 0:
                if ob1h.low <= ob.mid <= ob1h.high:
                    score += 1.5
                    labels.append("1H OB 包含")
                    break

        # 加分：4H OB 包含此位置
        for ob4h in obs_4h:
            if ob4h.direction == direction and ob4h.strength > 0:
                if ob4h.low <= ob.mid <= ob4h.high:
                    score += 2.0
                    labels.append("4H OB 包含")
                    break

        # 加分：FVG 包含此位置
        for fvg in fvgs_15m + fvgs_1h:
            if fvg.direction == direction:
                if fvg.low <= ob.mid <= fvg.high:
                    score += 1.0
                    labels.append(f"FVG({fvg.timeframe})")
                    break

        # 加分：FIB 水平在此 OB 範圍內
        fib_labels = []
        for ratio_str, fib_price in fib.items():
            if ratio_str in ["swing_high", "swing_low"]:
                continue
            if ob.low <= fib_price <= ob.high:
                ratio = float(ratio_str)
                score += 1.0
                fib_labels.append(f"FIB {ratio_str}")
        if fib_labels:
            labels.extend(fib_labels[:2])  # 最多顯示兩個 FIB 標籤

        # 加分：關鍵水平在此 OB 範圍內
        for level_name, level_price in key_levels.items():
            if level_price > 0 and ob.low <= level_price <= ob.high:
                score += 1.0
                labels.append(level_name.upper())

        # 加分：EQH / EQL 在此 OB 範圍內
        if direction == "bullish" and eqh_eql.get("eql"):
            if ob.low <= eqh_eql["eql"] <= ob.high:
                score += 1.0
                labels.append("EQL")
        if direction == "bearish" and eqh_eql.get("eqh"):
            if ob.low <= eqh_eql["eqh"] <= ob.high:
                score += 1.0
                labels.append("EQH")

        # 判斷是否在折扣區
        in_discount = False
        if direction == "bullish":
            in_discount = ob.mid <= discount_boundary
        else:
            in_discount = ob.mid >= discount_boundary

        # 判斷是否在 1H 高低點範圍內
        in_1h_range = range_low <= ob.mid <= range_high

        zone = KeyZone(
            price=ob.mid,
            high=ob.high,
            low=ob.low,
            score=score,
            direction=direction,
            labels=labels,
            is_in_discount=in_discount,
            is_in_1h_range=in_1h_range,
            timeframe_primary=ob.timeframe,
        )
        zones.append(zone)

    # 排序：分數高優先，其次折扣區優先
    zones.sort(key=lambda z: (z.score, z.is_in_discount), reverse=True)
    return zones


# ─────────────────────────────────────────
# 3M MSS 偵測
# ─────────────────────────────────────────

def detect_3m_mss(klines_3m: list[dict], direction: str) -> dict:
    """
    偵測 3M 市場結構轉變（MSS）
    direction = "bullish" → 找 3M 實體陽線突破最近 Swing High（N 字形）
    direction = "bearish" → 找 3M 實體陰線跌破最近 Swing Low（反 N 字形）
    返回：{"confirmed": bool, "mss_price": float, "fvg": FVG or None}
    """
    if len(klines_3m) < 10:
        return {"confirmed": False, "mss_price": 0.0, "fvg": None}

    recent = klines_3m[-20:]
    last = recent[-1]
    prev_bars = recent[:-1]

    if direction == "bullish":
        # 找最近的 Swing High（排除最後一根）
        swing_high = max(k["high"] for k in prev_bars[-10:])
        # 確認：最後一根實體陽線收盤突破 Swing High
        is_mss = (
            last["close"] > last["open"] and  # 陽線
            last["close"] > swing_high  # 突破 Swing High
        )
        if is_mss:
            # 搵 3M FVG（最後三根）
            fvg = None
            if len(recent) >= 3:
                k1 = recent[-3]
                k3 = recent[-1]
                if k1["high"] < k3["low"]:
                    fvg = FVG(
                        direction="bullish",
                        high=k3["low"],
                        low=k1["high"],
                        mid=(k3["low"] + k1["high"]) / 2,
                        ts=recent[-2]["ts"],
                        timeframe="3m",
                    )
            return {"confirmed": True, "mss_price": last["close"], "fvg": fvg}

    elif direction == "bearish":
        swing_low = min(k["low"] for k in prev_bars[-10:])
        is_mss = (
            last["close"] < last["open"] and  # 陰線
            last["close"] < swing_low  # 跌破 Swing Low
        )
        if is_mss:
            fvg = None
            if len(recent) >= 3:
                k1 = recent[-3]
                k3 = recent[-1]
                if k1["low"] > k3["high"]:
                    fvg = FVG(
                        direction="bearish",
                        high=k1["low"],
                        low=k3["high"],
                        mid=(k1["low"] + k3["high"]) / 2,
                        ts=recent[-2]["ts"],
                        timeframe="3m",
                    )
            return {"confirmed": True, "mss_price": last["close"], "fvg": fvg}

    return {"confirmed": False, "mss_price": 0.0, "fvg": None}


# ─────────────────────────────────────────
# TP 搜尋
# ─────────────────────────────────────────

def find_tp_levels(
    entry: float,
    sl: float,
    direction: str,
    obs_15m: list[OrderBlock],
    fvgs_15m: list[FVG],
    key_levels: dict,
    eqh_eql: dict,
    current_price: float,
) -> dict:
    """
    搜尋 TP1（15M 小阻力，RR ≥ 1:1）和 TP2（1H 大關鍵位）
    若 TP1 RR < 1:1，向上/下找下一個，並標注原因
    """
    sl_dist = abs(entry - sl)
    if sl_dist <= 0:
        return {"tp1": 0, "tp2": 0, "tp1_label": "", "tp2_label": "", "tp1_rr": 0, "tp2_rr": 0, "tp1_note": ""}

    min_rr = 1.0
    tp1_candidates = []
    tp2_candidates = []

    if direction == "bullish":
        # 搜尋上方阻力位
        # 15M 看跌 OB（小阻力）
        for ob in obs_15m:
            if ob.direction == "bearish" and ob.strength > 0 and ob.low > entry:
                rr = (ob.low - entry) / sl_dist
                tp1_candidates.append((ob.low, f"15M 看跌 OB", rr))

        # EQH（上方等高點）
        if eqh_eql.get("eqh") and eqh_eql["eqh"] > entry:
            rr = (eqh_eql["eqh"] - entry) / sl_dist
            tp1_candidates.append((eqh_eql["eqh"], "EQH 等高點", rr))

        # BSL / PDH / DO
        for name, price in key_levels.items():
            if price > entry:
                rr = (price - entry) / sl_dist
                tp2_candidates.append((price, name.upper(), rr))

        # FVG（看跌，上方）
        for fvg in fvgs_15m:
            if fvg.direction == "bearish" and fvg.low > entry:
                rr = (fvg.low - entry) / sl_dist
                tp1_candidates.append((fvg.low, "15M 看跌 FVG", rr))

    else:  # bearish
        # 搜尋下方支撐位
        for ob in obs_15m:
            if ob.direction == "bullish" and ob.strength > 0 and ob.high < entry:
                rr = (entry - ob.high) / sl_dist
                tp1_candidates.append((ob.high, "15M 看漲 OB", rr))

        if eqh_eql.get("eql") and eqh_eql["eql"] < entry:
            rr = (entry - eqh_eql["eql"]) / sl_dist
            tp1_candidates.append((eqh_eql["eql"], "EQL 等低點", rr))

        for name, price in key_levels.items():
            if price < entry and price > 0:
                rr = (entry - price) / sl_dist
                tp2_candidates.append((price, name.upper(), rr))

        for fvg in fvgs_15m:
            if fvg.direction == "bullish" and fvg.high < entry:
                rr = (entry - fvg.high) / sl_dist
                tp1_candidates.append((fvg.high, "15M 看漲 FVG", rr))

    # 排序：優先選 RR ≥ 1 的最近目標
    tp1_candidates.sort(key=lambda x: abs(x[0] - entry))
    tp2_candidates.sort(key=lambda x: abs(x[0] - entry))

    # 選 TP1
    tp1_note = ""
    tp1_price, tp1_label, tp1_rr = 0, "", 0
    for price, label, rr in tp1_candidates:
        if rr >= min_rr:
            tp1_price, tp1_label, tp1_rr = price, label, rr
            break

    if tp1_price == 0 and tp1_candidates:
        # 找不到 RR ≥ 1 的，取最近的並標注
        tp1_price, tp1_label, tp1_rr = tp1_candidates[0]
        tp1_note = f"⚠️ 最近阻力位 RR 僅 1:{tp1_rr:.1f}，已選用最近可用目標"

    if tp1_price == 0:
        # 完全找不到，用 RR 公式
        if direction == "bullish":
            tp1_price = entry + sl_dist * 1.5
        else:
            tp1_price = entry - sl_dist * 1.5
        tp1_label = "1:1.5 RR 目標"
        tp1_rr = 1.5
        tp1_note = "⚠️ 無明顯阻力位，使用 RR 公式計算"

    # 選 TP2（比 TP1 更遠）
    tp2_price, tp2_label, tp2_rr = 0, "", 0
    for price, label, rr in tp2_candidates:
        if direction == "bullish" and price > tp1_price and rr >= 2.0:
            tp2_price, tp2_label, tp2_rr = price, label, rr
            break
        elif direction == "bearish" and price < tp1_price and rr >= 2.0:
            tp2_price, tp2_label, tp2_rr = price, label, rr
            break

    if tp2_price == 0:
        if direction == "bullish":
            tp2_price = entry + sl_dist * 3.0
        else:
            tp2_price = entry - sl_dist * 3.0
        tp2_label = "1:3 RR 目標"
        tp2_rr = 3.0

    return {
        "tp1": tp1_price,
        "tp1_label": tp1_label,
        "tp1_rr": tp1_rr,
        "tp1_note": tp1_note,
        "tp2": tp2_price,
        "tp2_label": tp2_label,
        "tp2_rr": tp2_rr,
    }


# ─────────────────────────────────────────
# 完整分析（單一幣種）
# ─────────────────────────────────────────

def analyze_symbol(symbol: str) -> dict:
    """
    對單一幣種進行完整分析
    返回所有需要的數據供各功能使用
    """
    now_ts = int(time.time() * 1000)

    # 拉取 K 線
    klines_4h = fetch_klines(symbol, "4h", 200)
    klines_1h = fetch_klines(symbol, "1h", 500)
    klines_15m = fetch_klines(symbol, "15m", 300)
    klines_3m = fetch_klines(symbol, "3m", 200)

    if not klines_15m or not klines_1h:
        return {}

    current_price = klines_15m[-1]["close"]

    # ATR（15M 14 期）
    atr_15m = calc_atr(klines_15m, 14)

    # 市場結構
    struct_4h = get_market_structure(klines_4h, 30) if klines_4h else "ranging"
    struct_1h = get_market_structure(klines_1h, 30)

    # 關鍵水平
    key_levels = get_key_levels(klines_1h, klines_4h)

    # OB 識別
    obs_15m_raw = detect_obs(klines_15m, "15m", 80)
    obs_1h_raw = detect_obs(klines_1h, "1h", 50)
    obs_4h_raw = detect_obs(klines_4h, "4h", 30) if klines_4h else []

    # 更新 OB 有效性
    obs_15m = update_ob_validity(obs_15m_raw, current_price, klines_15m)
    obs_1h = update_ob_validity(obs_1h_raw, current_price, klines_1h)
    obs_4h = update_ob_validity(obs_4h_raw, current_price, klines_4h) if klines_4h else []

    # FVG 識別
    fvgs_15m = detect_fvgs(klines_15m, "15m", 80)
    fvgs_1h = detect_fvgs(klines_1h, "1h", 50)

    # FIB（根據 1H 方向）
    fib_direction = struct_1h if struct_1h != "ranging" else "bullish"
    fib = calc_fib(klines_1h, fib_direction, 50)

    # EQH / EQL
    eqh_eql = find_eqh_eql(klines_15m)

    # Swing points
    swings_15m = get_swing_points(klines_15m, 20)
    swings_1h = get_swing_points(klines_1h, 20)

    # 3M MSS
    mss_bull = detect_3m_mss(klines_3m, "bullish")
    mss_bear = detect_3m_mss(klines_3m, "bearish")

    return {
        "symbol": symbol,
        "current_price": current_price,
        "atr_15m": atr_15m,
        "struct_4h": struct_4h,
        "struct_1h": struct_1h,
        "key_levels": key_levels,
        "obs_15m": obs_15m,
        "obs_1h": obs_1h,
        "obs_4h": obs_4h,
        "fvgs_15m": fvgs_15m,
        "fvgs_1h": fvgs_1h,
        "fib": fib,
        "eqh_eql": eqh_eql,
        "swings_15m": swings_15m,
        "swings_1h": swings_1h,
        "mss_bull": mss_bull,
        "mss_bear": mss_bear,
        "klines_15m": klines_15m,
        "klines_1h": klines_1h,
        "klines_3m": klines_3m,
        "now_ts": now_ts,
    }
