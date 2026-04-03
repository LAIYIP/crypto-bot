"""
Bot v6.0 訊號生成模組
功能一：自動入場訊號
功能二/四：定時/即時雙向分析
功能五：掛單建議
"""

from datetime import datetime, timezone, timedelta
from core_engine import (
    analyze_symbol, score_key_zones, find_tp_levels,
    KeyZone, HKT
)

# ─────────────────────────────────────────
# 輔助函數
# ─────────────────────────────────────────

def fmt_price(p: float, symbol: str = "") -> str:
    """格式化價格"""
    if "BTC" in symbol or p > 10000:
        return f"{p:,.2f}"
    elif p > 100:
        return f"{p:,.3f}"
    else:
        return f"{p:,.4f}"


def is_low_liquidity() -> bool:
    """判斷是否低流動性時段（00:00-06:00 HKT）"""
    now_hkt = datetime.now(HKT)
    return 0 <= now_hkt.hour < 6


def get_session_label() -> str:
    """取得當前時段標籤"""
    now_hkt = datetime.now(HKT)
    h = now_hkt.hour
    if 8 <= h < 12:
        return "早盤"
    elif 12 <= h < 17:
        return "午盤"
    elif 17 <= h < 20:
        return "歐洲盤"
    elif 20 <= h < 23:
        return "美盤"
    else:
        return "深夜盤"


def get_limit_order_expiry() -> str:
    """根據當前時間計算掛單有效期"""
    now_hkt = datetime.now(HKT)
    h = now_hkt.hour

    if 0 <= h < 8:
        expiry = now_hkt.replace(hour=8, minute=0, second=0, microsecond=0)
        label = "亞洲盤開市"
    elif 8 <= h < 12:
        expiry = now_hkt.replace(hour=12, minute=0, second=0, microsecond=0)
        label = "午盤"
    elif 12 <= h < 17:
        expiry = now_hkt.replace(hour=17, minute=0, second=0, microsecond=0)
        label = "歐洲盤開市"
    elif 17 <= h < 20:
        expiry = now_hkt.replace(hour=20, minute=30, second=0, microsecond=0)
        label = "美盤開市"
    elif 20 <= h < 23:
        expiry = now_hkt.replace(hour=23, minute=30, second=0, microsecond=0)
        label = "紐約深夜"
    else:
        expiry = (now_hkt + timedelta(days=1)).replace(hour=8, minute=0, second=0, microsecond=0)
        label = "次日亞洲盤開市"

    return f"至 {expiry.strftime('%H:%M')} HKT（{label}前取消）"


def get_overall_bias(struct_4h: str, struct_1h: str) -> tuple[str, str]:
    """返回 (bias_emoji, bias_text)"""
    if struct_4h == "bullish" and struct_1h == "bullish":
        return "🟢", "偏看漲（4H + 1H 雙重確認）"
    elif struct_4h == "bearish" and struct_1h == "bearish":
        return "🔴", "偏看跌（4H + 1H 雙重確認）"
    elif struct_1h == "bullish":
        return "🟡", "1H 看漲，4H 尚未確認"
    elif struct_1h == "bearish":
        return "🟡", "1H 看跌，4H 尚未確認"
    elif struct_4h == "bullish":
        return "🟡", "4H 看漲，1H 整理中"
    elif struct_4h == "bearish":
        return "🟡", "4H 看跌，1H 整理中"
    else:
        return "⚪", "橫盤整理，等待方向"


def _zone_behavior_hint(zone: KeyZone, direction: str, eqh_eql: dict, current_price: float) -> str:
    """
    根據區間性質生成「到達後應看到的行為」提示
    這是分析報告的核心改進：告訴用戶到達區間後應觀察什麼
    """
    labels = zone.labels
    label_str = " + ".join(labels[:3]) if labels else "OB"

    # 判斷區間主要性質
    has_eql = "EQL" in labels
    has_eqh = "EQH" in labels
    has_ob = any("OB" in l for l in labels)
    has_fvg = any("FVG" in l for l in labels)
    has_fib = any("FIB" in l for l in labels)
    has_bsl = "BSL" in labels
    has_ssl = "SSL" in labels
    has_pdh = "PDH" in labels
    has_pdl = "PDL" in labels

    zone_range = f"{zone.low:,.2f} - {zone.high:,.2f}"

    if direction == "bullish":
        # 看漲情景：價格下探到支撐區
        if has_eql:
            liquidity = f"EQL 等低點（{zone.price:,.2f}）為下方流動性池"
            action = "價格觸及 EQL 後，觀察是否出現流動性獵取（假跌破後快速收回），3M 出現陽線 MSS 確認反轉入場做多"
        elif has_ssl:
            liquidity = f"SSL 下方流動性（{zone.price:,.2f}）"
            action = "價格觸及 SSL 後，觀察是否出現假跌破（Sweep），3M 收回後出現陽線突破近期 Swing High 確認入場"
        elif has_fvg and has_ob:
            liquidity = f"OB + FVG 重疊區（{zone_range}）"
            action = "價格回踩入 OB 範圍，觀察是否在 FVG 中點附近止跌，3M 出現陽線 MSS 後入場"
        elif has_ob:
            tf = "15M" if zone.timeframe_primary == "15m" else "1H" if zone.timeframe_primary == "1h" else "4H"
            liquidity = f"{tf} 看漲 OB（{zone_range}）"
            action = f"價格回踩 {tf} OB 範圍，觀察是否在 OB 中點（{zone.price:,.2f}）附近出現支撐蠟燭（長下影線或陽線包裹），3M 出現陽線 MSS 確認入場"
        elif has_fib:
            fib_labels = [l for l in labels if "FIB" in l]
            fib_str = fib_labels[0] if fib_labels else "FIB"
            liquidity = f"{fib_str} 回調位（{zone.price:,.2f}）"
            action = "價格回調至 FIB 黃金比例位，觀察是否出現支撐蠟燭，3M 結構轉漲後入場"
        elif has_pdl:
            liquidity = f"PDL 前日低（{zone.price:,.2f}）"
            action = "價格觸及 PDL，觀察是否出現假跌破後快速收回，3M 陽線 MSS 確認後入場做多"
        else:
            liquidity = f"關鍵支撐區（{zone_range}）"
            action = "價格進入支撐區，觀察是否出現止跌蠟燭（陽線包裹或長下影線），3M 陽線突破 Swing High 後入場"

    else:
        # 看跌情景：價格上升到阻力區
        if has_eqh:
            liquidity = f"EQH 等高點（{zone.price:,.2f}）為上方流動性池"
            action = "價格觸及 EQH 後，觀察是否出現流動性獵取（假突破後快速收回），3M 出現陰線 MSS 確認反轉入場做空"
        elif has_bsl:
            liquidity = f"BSL 上方流動性（{zone.price:,.2f}）"
            action = "價格觸及 BSL 後，觀察是否出現假突破（Sweep），3M 收回後出現陰線跌破近期 Swing Low 確認入場"
        elif has_fvg and has_ob:
            liquidity = f"OB + FVG 重疊區（{zone_range}）"
            action = "價格反彈入 OB 範圍，觀察是否在 FVG 中點附近遇阻，3M 出現陰線 MSS 後入場"
        elif has_ob:
            tf = "15M" if zone.timeframe_primary == "15m" else "1H" if zone.timeframe_primary == "1h" else "4H"
            liquidity = f"{tf} 看跌 OB（{zone_range}）"
            action = f"價格反彈至 {tf} OB 範圍，觀察是否在 OB 中點（{zone.price:,.2f}）附近出現阻力蠟燭（長上影線或陰線包裹），3M 出現陰線 MSS 確認入場"
        elif has_fib:
            fib_labels = [l for l in labels if "FIB" in l]
            fib_str = fib_labels[0] if fib_labels else "FIB"
            liquidity = f"{fib_str} 反彈位（{zone.price:,.2f}）"
            action = "價格反彈至 FIB 黃金比例位，觀察是否出現阻力蠟燭，3M 結構轉跌後入場"
        elif has_pdh:
            liquidity = f"PDH 前日高（{zone.price:,.2f}）"
            action = "價格觸及 PDH，觀察是否出現假突破後快速收回，3M 陰線 MSS 確認後入場做空"
        else:
            liquidity = f"關鍵阻力區（{zone_range}）"
            action = "價格進入阻力區，觀察是否出現止漲蠟燭（陰線包裹或長上影線），3M 陰線跌破 Swing Low 後入場"

    return liquidity, action


# ─────────────────────────────────────────
# 功能一：自動入場訊號生成
# ─────────────────────────────────────────

def generate_auto_signal(data: dict) -> dict | None:
    """
    根據分析數據生成自動入場訊號
    返回訊號字典，或 None（無訊號）
    """
    symbol = data["symbol"]
    current_price = data["current_price"]
    atr = data["atr_15m"]
    struct_1h = data["struct_1h"]
    struct_4h = data["struct_4h"]
    mss_bull = data["mss_bull"]
    mss_bear = data["mss_bear"]

    # 確定主方向（1H 為主）
    if struct_1h == "bullish":
        signal_dir = "bullish"
    elif struct_1h == "bearish":
        signal_dir = "bearish"
    else:
        return None  # 1H 橫盤，不發自動訊號

    # 確認 3M MSS
    mss = mss_bull if signal_dir == "bullish" else mss_bear
    if not mss["confirmed"]:
        return None

    # 計算重疊分數，找最佳入場位
    zones = score_key_zones(
        current_price=current_price,
        direction=signal_dir,
        obs_15m=data["obs_15m"],
        obs_1h=data["obs_1h"],
        obs_4h=data["obs_4h"],
        fvgs_15m=data["fvgs_15m"],
        fvgs_1h=data["fvgs_1h"],
        fib=data["fib"],
        key_levels=data["key_levels"],
        eqh_eql=data["eqh_eql"],
        klines_15m=data["klines_15m"],
        now_ts=data["now_ts"],
    )

    if not zones:
        return None

    # 選最佳入場區（分數最高）
    best_zone = zones[0]

    # 入場位：3M FVG 中點（若有），否則用 OB 中點
    fvg_3m = mss.get("fvg")
    if fvg_3m:
        entry = fvg_3m.mid
        entry_label = f"3M FVG（建議掛單回踩 {fvg_3m.low:.2f}-{fvg_3m.high:.2f}）"
    else:
        entry = best_zone.price
        entry_label = f"{best_zone.timeframe_primary.upper()} OB 中點"

    # SL：關鍵位框架外側 + ATR × 0.3 呼吸空間
    if signal_dir == "bullish":
        sl = best_zone.low - atr * 0.3
    else:
        sl = best_zone.high + atr * 0.3

    # 確認 SL 方向正確
    if signal_dir == "bullish" and sl >= entry:
        sl = entry - atr * 1.5
    if signal_dir == "bearish" and sl <= entry:
        sl = entry + atr * 1.5

    # 搜尋 TP
    tp_data = find_tp_levels(
        entry=entry,
        sl=sl,
        direction=signal_dir,
        obs_15m=data["obs_15m"],
        fvgs_15m=data["fvgs_15m"],
        key_levels=data["key_levels"],
        eqh_eql=data["eqh_eql"],
        current_price=current_price,
    )

    # 逆勢判斷
    is_counter = (struct_4h != "ranging" and struct_4h != signal_dir)
    high_prob = (struct_4h == signal_dir)

    return {
        "symbol": symbol,
        "direction": signal_dir,
        "entry": entry,
        "entry_label": entry_label,
        "sl": sl,
        "sl_label": f"{best_zone.timeframe_primary.upper()} {'OB 底部' if signal_dir == 'bullish' else 'OB 頂部'}外 + ATR×0.3",
        "tp1": tp_data["tp1"],
        "tp1_label": tp_data["tp1_label"],
        "tp1_rr": tp_data["tp1_rr"],
        "tp1_note": tp_data["tp1_note"],
        "tp2": tp_data["tp2"],
        "tp2_label": tp_data["tp2_label"],
        "tp2_rr": tp_data["tp2_rr"],
        "zone_score": best_zone.score,
        "zone_labels": best_zone.labels,
        "is_in_discount": best_zone.is_in_discount,
        "is_counter": is_counter,
        "high_prob": high_prob,
        "struct_1h": struct_1h,
        "struct_4h": struct_4h,
        "mss_price": mss["mss_price"],
        "current_price": current_price,
        "atr": atr,
    }


def format_auto_signal(sig: dict) -> str:
    """格式化自動入場訊號訊息"""
    now_str = datetime.now(HKT).strftime("%m-%d %H:%M HKT")
    sym = sig["symbol"]
    direction = sig["direction"]
    is_bull = direction == "bullish"

    dir_emoji = "🟢" if is_bull else "🔴"
    dir_text = "做多 (Long)" if is_bull else "做空 (Short)"

    # 勝率標注
    if sig["high_prob"]:
        prob_tag = "✅ 高勝率（4H + 1H 同向）"
    elif sig["is_counter"]:
        prob_tag = "⚠️ 逆勢入場，建議半倉"
    else:
        prob_tag = "🟡 順勢（1H 確認，4H 尚未同步）"

    # 低流動性標注
    low_liq = "⚠️ 低流動性時段，訊號可信度較低\n" if is_low_liquidity() else ""

    # 重疊標籤
    zone_info = " + ".join(sig["zone_labels"][:4]) if sig["zone_labels"] else "OB"
    score_str = f"{sig['zone_score']:.1f}"

    # 結構
    struct_4h_map = {"bullish": "⬆️ 看漲", "bearish": "⬇️ 看跌", "ranging": "↔️ 橫盤"}
    struct_1h_map = {"bullish": "⬆️ 看漲", "bearish": "⬇️ 看跌", "ranging": "↔️ 橫盤"}

    # 折扣區標注
    discount_tag = "（折扣區）" if sig["is_in_discount"] else ""

    tp1_note = f"\n   {sig['tp1_note']}" if sig["tp1_note"] else ""

    if is_bull:
        trade_block = (
            f"🎯 TP2：{sig['tp2']:,.2f}（{sig['tp2_label']}）  RR 1:{sig['tp2_rr']:.1f}\n"
            f"🎯 TP1：{sig['tp1']:,.2f}（{sig['tp1_label']}）  RR 1:{sig['tp1_rr']:.1f}{tp1_note}\n"
            f"📍 入場：{sig['entry']:,.2f}（{sig['entry_label']}）{discount_tag}\n"
            f"🛑 SL：{sig['sl']:,.2f}（{sig['sl_label']}）"
        )
    else:
        trade_block = (
            f"🛑 SL：{sig['sl']:,.2f}（{sig['sl_label']}）\n"
            f"📍 入場：{sig['entry']:,.2f}（{sig['entry_label']}）{discount_tag}\n"
            f"🎯 TP1：{sig['tp1']:,.2f}（{sig['tp1_label']}）  RR 1:{sig['tp1_rr']:.1f}{tp1_note}\n"
            f"🎯 TP2：{sig['tp2']:,.2f}（{sig['tp2_label']}）  RR 1:{sig['tp2_rr']:.1f}"
        )

    msg = (
        f"🚨 【入場訊號】{sym} [{now_str}]\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"{prob_tag}\n"
        f"{low_liq}"
        f"\n"
        f"📈 方向：{dir_emoji} {dir_text}\n"
        f"🔍 4H：{struct_4h_map.get(sig['struct_4h'], '?')}  |  1H：{struct_1h_map.get(sig['struct_1h'], '?')}\n"
        f"🧩 關鍵區域：{zone_info}（重疊分 {score_str}）\n"
        f"✅ 3M MSS 確認：{sig['mss_price']:,.2f}\n"
        f"\n"
        f"──────────────────\n"
        f"{trade_block}\n"
        f"──────────────────\n"
        f"💲 現價：{sig['current_price']:,.2f}"
    )
    return msg


# ─────────────────────────────────────────
# 功能二/四：雙向情景分析（重寫版）
# 格式：4H 方向 → 1H 主力 → 15M 區間（性質）→ 3M 行為提示
# ─────────────────────────────────────────

def format_directional_analysis(data: dict, session_label: str = "") -> str:
    """
    生成雙向情景分析訊息
    新格式：
    - 4H 方向（背景）
    - 1H 主力方向
    - 15M 關鍵區間（OB/FVG/EQL/EQH 性質）
    - 3M 等待行為（流動性獵取 / MSS 確認）
    """
    now_str = datetime.now(HKT).strftime("%m-%d %H:%M HKT")
    sym = data["symbol"]
    current_price = data["current_price"]
    atr = data["atr_15m"]
    struct_1h = data["struct_1h"]
    struct_4h = data["struct_4h"]
    key_levels = data["key_levels"]
    eqh_eql = data["eqh_eql"]

    if not session_label:
        session_label = get_session_label()

    bias_emoji, bias_text = get_overall_bias(struct_4h, struct_1h)

    # 關鍵水平字串
    kl = key_levels
    kl_str = (
        f"PDH {kl['pdh']:,.2f} | PDL {kl['pdl']:,.2f} | "
        f"DO {kl['do']:,.2f} | BSL {kl['bsl']:,.2f} | SSL {kl['ssl']:,.2f}"
    )

    # 結構描述
    struct_map = {"bullish": "⬆️ 看漲", "bearish": "⬇️ 看跌", "ranging": "↔️ 整理"}
    s4h = struct_map.get(struct_4h, "?")
    s1h = struct_map.get(struct_1h, "?")

    # 生成兩個方向的關鍵區
    bull_zones = score_key_zones(
        current_price=current_price, direction="bullish",
        obs_15m=data["obs_15m"], obs_1h=data["obs_1h"], obs_4h=data["obs_4h"],
        fvgs_15m=data["fvgs_15m"], fvgs_1h=data["fvgs_1h"],
        fib=data["fib"], key_levels=key_levels,
        eqh_eql=eqh_eql, klines_15m=data["klines_15m"], now_ts=data["now_ts"],
    )
    bear_zones = score_key_zones(
        current_price=current_price, direction="bearish",
        obs_15m=data["obs_15m"], obs_1h=data["obs_1h"], obs_4h=data["obs_4h"],
        fvgs_15m=data["fvgs_15m"], fvgs_1h=data["fvgs_1h"],
        fib=data["fib"], key_levels=key_levels,
        eqh_eql=eqh_eql, klines_15m=data["klines_15m"], now_ts=data["now_ts"],
    )

    def build_scenario_block(zone: KeyZone, direction: str, is_main: bool) -> str:
        """
        生成單個情景區塊
        格式：
        [方向標題]
        4H：xxx  1H：xxx
        15M 區間：xxx（性質）
        到達後觀察：xxx（流動性獵取/MSS 行為）
        入場：xxx  SL：xxx  TP1：xxx  TP2：xxx
        """
        is_bull = direction == "bullish"
        dir_emoji = "🟢" if is_bull else "🔴"
        dir_text = "看漲" if is_bull else "看跌"
        main_tag = "（主路線）" if is_main else "（備用）"

        # SL
        if is_bull:
            sl = zone.low - atr * 0.3
        else:
            sl = zone.high + atr * 0.3

        # TP
        tp_data = find_tp_levels(
            entry=zone.price, sl=sl, direction=direction,
            obs_15m=data["obs_15m"], fvgs_15m=data["fvgs_15m"],
            key_levels=key_levels, eqh_eql=eqh_eql,
            current_price=current_price,
        )

        zone_label = " + ".join(zone.labels[:3]) if zone.labels else "OB"
        zone_range = f"{zone.low:,.2f} - {zone.high:,.2f}"

        # 區間性質與行為提示
        liquidity_desc, action_desc = _zone_behavior_hint(zone, direction, eqh_eql, current_price)

        # 入場條件（根據是否有 3M MSS 已確認）
        if is_bull:
            entry_cond = "3M 陽線實體突破近期 Swing High（MSS 確認）"
        else:
            entry_cond = "3M 陰線實體跌破近期 Swing Low（MSS 確認）"

        tp1_note = f"\n      ⚠️ {tp_data['tp1_note']}" if tp_data.get("tp1_note") else ""

        if is_bull:
            trade_lines = (
                f"   🎯 TP2：{tp_data['tp2']:,.2f}（{tp_data['tp2_label']}）RR 1:{tp_data['tp2_rr']:.1f}\n"
                f"   🎯 TP1：{tp_data['tp1']:,.2f}（{tp_data['tp1_label']}）RR 1:{tp_data['tp1_rr']:.1f}{tp1_note}\n"
                f"   📍 入場：{zone.price:,.2f}（{zone_label}）\n"
                f"   🛑 SL：{sl:,.2f}（框架底部外 + ATR×0.3）"
            )
        else:
            trade_lines = (
                f"   🛑 SL：{sl:,.2f}（框架頂部外 + ATR×0.3）\n"
                f"   📍 入場：{zone.price:,.2f}（{zone_label}）\n"
                f"   🎯 TP1：{tp_data['tp1']:,.2f}（{tp_data['tp1_label']}）RR 1:{tp_data['tp1_rr']:.1f}{tp1_note}\n"
                f"   🎯 TP2：{tp_data['tp2']:,.2f}（{tp_data['tp2_label']}）RR 1:{tp_data['tp2_rr']:.1f}"
            )

        block = (
            f"{dir_emoji} {dir_text}情景{main_tag}\n"
            f"   ├ 15M 關鍵區：{liquidity_desc}\n"
            f"   ├ 到達後觀察：{action_desc}\n"
            f"   ├ 入場確認：{entry_cond}\n"
            f"   └─────────────────\n"
            f"{trade_lines}\n"
        )
        return block

    # 判斷主方向
    aligned = (
        (struct_4h == "bullish" and struct_1h == "bullish") or
        (struct_4h == "bearish" and struct_1h == "bearish")
    )
    # 盤整時：用 4H 方向作為主方向；4H 也盤整則雙向均等
    if struct_1h != "ranging":
        main_dir = struct_1h
    elif struct_4h != "ranging":
        main_dir = struct_4h
    else:
        main_dir = None  # 完全雙向

    header = (
        f"📊 {sym} {session_label}分析 [{now_str}]\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"{bias_emoji} 整體偏向：【{bias_text}】\n"
        f"🕐 4H：{s4h}  |  1H：{s1h}\n"
        f"📌 關鍵水平：{kl_str}\n"
        f"💲 現價：{current_price:,.2f}\n"
    )

    if is_low_liquidity():
        header += "⚠️ 低流動性時段，訊號可信度較低\n"

    body = "\n"

    if main_dir == "bullish" and bull_zones:
        # 主方向看漲
        body += build_scenario_block(bull_zones[0], "bullish", is_main=True)
        if bear_zones:
            bz = bear_zones[0]
            liq, act = _zone_behavior_hint(bz, "bearish", eqh_eql, current_price)
            body += (
                f"🔴 看跌備用：若價格升至 {bz.price:,.2f}（{' + '.join(bz.labels[:2]) if bz.labels else 'OB'}）\n"
                f"   觀察：{act}\n"
            )

    elif main_dir == "bearish" and bear_zones:
        # 主方向看跌
        body += build_scenario_block(bear_zones[0], "bearish", is_main=True)
        if bull_zones:
            bz = bull_zones[0]
            liq, act = _zone_behavior_hint(bz, "bullish", eqh_eql, current_price)
            body += (
                f"🟢 看漲備用：若價格回踩 {bz.price:,.2f}（{' + '.join(bz.labels[:2]) if bz.labels else 'OB'}）\n"
                f"   觀察：{act}\n"
            )

    else:
        # 雙向均等（1H + 4H 均盤整，或方向不同）
        if bull_zones:
            body += build_scenario_block(bull_zones[0], "bullish", is_main=False)
        if bear_zones:
            body += build_scenario_block(bear_zones[0], "bearish", is_main=False)
        if not main_dir:
            body += "⚠️ 4H + 1H 均橫盤，兩個方向均以半倉入場，等待結構明確後加倉\n"
        else:
            body += "⚠️ 4H + 1H 方向未同步，建議等待 1H 結構確認後再入場\n"

    return header + body


# ─────────────────────────────────────────
# 功能三：按需詳細報告
# ─────────────────────────────────────────

def format_on_demand_report(data: dict) -> str:
    """生成按需詳細報告（打幣種名觸發）"""
    now_str = datetime.now(HKT).strftime("%m-%d %H:%M HKT")
    sym = data["symbol"]
    current_price = data["current_price"]
    struct_1h = data["struct_1h"]
    struct_4h = data["struct_4h"]
    key_levels = data["key_levels"]
    eqh_eql = data["eqh_eql"]
    struct_map = {"bullish": "⬆️ 看漲", "bearish": "⬇️ 看跌", "ranging": "↔️ 橫盤"}
    bias_emoji, bias_text = get_overall_bias(struct_4h, struct_1h)

    # 整理所有關鍵位（由高至低）
    all_levels = []
    kl = key_levels
    for name, price in [
        ("PWH 前週高", kl["pwh"]),
        ("BSL 上方流動性", kl["bsl"]),
        ("PDH 前日高", kl["pdh"]),
        ("DO 今日開盤", kl["do"]),
        ("WO 本週開盤", kl["wo"]),
        ("PDL 前日低", kl["pdl"]),
        ("SSL 下方流動性", kl["ssl"]),
        ("PWL 前週低", kl["pwl"]),
    ]:
        if price > 0:
            all_levels.append((price, name))
    if eqh_eql.get("eqh"):
        all_levels.append((eqh_eql["eqh"], "EQH 等高點"))
    if eqh_eql.get("eql"):
        all_levels.append((eqh_eql["eql"], "EQL 等低點"))

    # 加入 1H OB
    for ob in data["obs_1h"][:3]:
        label = f"1H {'看漲' if ob.direction == 'bullish' else '看跌'} OB"
        all_levels.append((ob.mid, label))

    all_levels.sort(key=lambda x: x[0], reverse=True)

    # 重新排列，把現價插入正確位置
    above = [(p, n) for p, n in all_levels if p > current_price]
    below = [(p, n) for p, n in all_levels if p <= current_price]
    above.sort(key=lambda x: x[0], reverse=True)
    below.sort(key=lambda x: x[0], reverse=True)

    levels_final = ""
    for price, name in above:
        levels_final += f"   🔴 {price:,.2f}  {name}\n"
    levels_final += f"   ──── 💲{current_price:,.2f} 現價 ────\n"
    for price, name in below:
        levels_final += f"   🟢 {price:,.2f}  {name}\n"

    msg = (
        f"📋 {sym} 詳細報告 [{now_str}]\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"📌 {sym}  💲{current_price:,.2f}\n"
        f"   4H {struct_map.get(struct_4h, '?')}  |  1H {struct_map.get(struct_1h, '?')}\n"
        f"   {bias_emoji} 整體偏向：{bias_text}\n"
        f"\n"
        f"📊 關鍵位置（由高至低）：\n"
        f"{levels_final}"
    )
    if is_low_liquidity():
        msg += "\n⚠️ 低流動性時段"
    return msg


# ─────────────────────────────────────────
# 功能五：掛單建議（重寫版）
# 修復：盤整時不再返回「暫無方向」
# 新增：區間性質 + 到達後行為提示
# ─────────────────────────────────────────

def format_limit_order(data: dict) -> str:
    """
    生成掛單建議
    邏輯：
    1. 1H 有方向 → 用 1H 方向
    2. 1H 橫盤但 4H 有方向 → 用 4H 方向（標注為逆勢注意）
    3. 1H + 4H 均橫盤 → 雙向掛單（等待突破方向確認）
    """
    now_str = datetime.now(HKT).strftime("%m-%d %H:%M HKT")
    sym = data["symbol"]
    current_price = data["current_price"]
    atr = data["atr_15m"]
    struct_1h = data["struct_1h"]
    struct_4h = data["struct_4h"]
    eqh_eql = data["eqh_eql"]

    # 確定方向邏輯（修復：盤整不再直接返回無方向）
    if struct_1h == "bullish":
        directions = ["bullish"]
        dir_note = ""
    elif struct_1h == "bearish":
        directions = ["bearish"]
        dir_note = ""
    elif struct_4h == "bullish":
        directions = ["bullish"]
        dir_note = "⚠️ 1H 整理中，以 4H 看漲方向為主，建議半倉"
    elif struct_4h == "bearish":
        directions = ["bearish"]
        dir_note = "⚠️ 1H 整理中，以 4H 看跌方向為主，建議半倉"
    else:
        # 完全橫盤：雙向掛單，等待突破
        directions = ["bullish", "bearish"]
        dir_note = "⚠️ 4H + 1H 均橫盤，雙向掛單等待突破，觸發一邊後取消另一邊"

    header = (
        f"📌 {sym} 掛單建議 [{now_str}]\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"🕐 4H：{'⬆️' if struct_4h == 'bullish' else '⬇️' if struct_4h == 'bearish' else '↔️'}  "
        f"1H：{'⬆️' if struct_1h == 'bullish' else '⬇️' if struct_1h == 'bearish' else '↔️'}\n"
    )
    if dir_note:
        header += f"{dir_note}\n"

    expiry = get_limit_order_expiry()
    body = ""

    for direction in directions:
        # 找關鍵區
        zones = score_key_zones(
            current_price=current_price,
            direction=direction,
            obs_15m=data["obs_15m"],
            obs_1h=data["obs_1h"],
            obs_4h=data["obs_4h"],
            fvgs_15m=data["fvgs_15m"],
            fvgs_1h=data["fvgs_1h"],
            fib=data["fib"],
            key_levels=data["key_levels"],
            eqh_eql=eqh_eql,
            klines_15m=data["klines_15m"],
            now_ts=data["now_ts"],
        )

        # 過濾：入場位必須距現價 ≥ 0.3%（確保有回調空間）
        min_distance = current_price * 0.003
        valid_zones = []
        for z in zones:
            if direction == "bullish" and z.price < current_price - min_distance:
                valid_zones.append(z)
            elif direction == "bearish" and z.price > current_price + min_distance:
                valid_zones.append(z)

        is_bull = direction == "bullish"
        dir_emoji = "🟢" if is_bull else "🔴"
        dir_text = "做多" if is_bull else "做空"

        if not valid_zones:
            body += (
                f"\n{dir_emoji} {dir_text}：\n"
                f"   現價附近無足夠距離的關鍵位，暫無掛單位置\n"
            )
            continue

        best = valid_zones[0]
        entry = best.price
        zone_label = " + ".join(best.labels[:3]) if best.labels else "OB"

        # SL：框架外側 + ATR × 0.3
        if is_bull:
            sl = best.low - atr * 0.3
        else:
            sl = best.high + atr * 0.3

        # 低流動性加寬 SL
        liq_note = ""
        if is_low_liquidity():
            sl_extra = atr * 0.2
            if is_bull:
                sl -= sl_extra
            else:
                sl += sl_extra
            liq_note = "（含夜間加寬 ATR×0.2）"

        sl_dist = abs(entry - sl)

        # TP
        tp_data = find_tp_levels(
            entry=entry, sl=sl, direction=direction,
            obs_15m=data["obs_15m"], fvgs_15m=data["fvgs_15m"],
            key_levels=data["key_levels"], eqh_eql=eqh_eql,
            current_price=current_price,
        )

        # 區間性質與行為提示
        liquidity_desc, action_desc = _zone_behavior_hint(best, direction, eqh_eql, current_price)

        # 取消條件
        if is_bull:
            cancel_price = entry - sl_dist * 0.5
            cancel_note = f"若未回調直接跌破 {cancel_price:,.2f}，取消掛單"
            trade_block = (
                f"   🎯 TP2：{tp_data['tp2']:,.2f}（{tp_data['tp2_label']}）RR 1:{tp_data['tp2_rr']:.1f}\n"
                f"   🎯 TP1：{tp_data['tp1']:,.2f}（{tp_data['tp1_label']}）RR 1:{tp_data['tp1_rr']:.1f}\n"
                f"   📍 入場：{entry:,.2f}（{zone_label}）\n"
                f"   🛑 SL：{sl:,.2f}（底部外 + ATR×0.3{liq_note}）"
            )
        else:
            cancel_price = entry + sl_dist * 0.5
            cancel_note = f"若未反彈直接突破 {cancel_price:,.2f}，取消掛單"
            trade_block = (
                f"   🛑 SL：{sl:,.2f}（頂部外 + ATR×0.3{liq_note}）\n"
                f"   📍 入場：{entry:,.2f}（{zone_label}）\n"
                f"   🎯 TP1：{tp_data['tp1']:,.2f}（{tp_data['tp1_label']}）RR 1:{tp_data['tp1_rr']:.1f}\n"
                f"   🎯 TP2：{tp_data['tp2']:,.2f}（{tp_data['tp2_label']}）RR 1:{tp_data['tp2_rr']:.1f}"
            )

        # 勝率標注
        if struct_4h == direction and struct_1h == direction:
            prob_tag = "✅ 高勝率（4H + 1H 同向）"
        elif struct_4h == direction or struct_1h == direction:
            prob_tag = "🟡 中等勝率（單一時框確認）"
        else:
            prob_tag = "⚠️ 逆勢掛單，建議半倉"

        tp1_note_str = f"\n   ⚠️ {tp_data['tp1_note']}" if tp_data.get("tp1_note") else ""

        body += (
            f"\n{dir_emoji} {dir_text}掛單：{prob_tag}\n"
            f"   🧩 區間：{liquidity_desc}\n"
            f"   👁 到達後觀察：{action_desc}\n"
            f"   ─────────────────\n"
            f"{trade_block}{tp1_note_str}\n"
            f"   📊 RR（至 TP1）：1:{tp_data['tp1_rr']:.1f}\n"
            f"   ⚠️ {cancel_note}\n"
        )

    footer = (
        f"\n⏰ 有效期：{expiry}\n"
    )
    if is_low_liquidity():
        footer += "⚠️ 低流動性時段，SL 已自動加寬\n"

    return header + body + footer
