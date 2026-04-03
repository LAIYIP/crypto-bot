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
    """
    返回 (bias_emoji, bias_text)
    """
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
        return None  # 1H 橫盤，不發訊號

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
# 功能二/四：雙向情景分析
# ─────────────────────────────────────────

def format_directional_analysis(data: dict, session_label: str = "") -> str:
    """
    生成雙向情景分析訊息
    4H + 1H 同向 → 主方向完整 + 另一方向一行備注
    不同向 → 平衡展示兩個方向
    """
    now_str = datetime.now(HKT).strftime("%m-%d %H:%M HKT")
    sym = data["symbol"]
    current_price = data["current_price"]
    atr = data["atr_15m"]
    struct_1h = data["struct_1h"]
    struct_4h = data["struct_4h"]
    key_levels = data["key_levels"]

    if not session_label:
        session_label = get_session_label()

    bias_emoji, bias_text = get_overall_bias(struct_4h, struct_1h)

    # 關鍵水平字串
    kl = key_levels
    kl_str = (
        f"PDH {kl['pdh']:,.2f} | PDL {kl['pdl']:,.2f} | "
        f"DO {kl['do']:,.2f} | BSL {kl['bsl']:,.2f}"
    )

    # 生成看漲情景
    bull_zones = score_key_zones(
        current_price=current_price,
        direction="bullish",
        obs_15m=data["obs_15m"],
        obs_1h=data["obs_1h"],
        obs_4h=data["obs_4h"],
        fvgs_15m=data["fvgs_15m"],
        fvgs_1h=data["fvgs_1h"],
        fib=data["fib"],
        key_levels=key_levels,
        eqh_eql=data["eqh_eql"],
        klines_15m=data["klines_15m"],
        now_ts=data["now_ts"],
    )

    bear_zones = score_key_zones(
        current_price=current_price,
        direction="bearish",
        obs_15m=data["obs_15m"],
        obs_1h=data["obs_1h"],
        obs_4h=data["obs_4h"],
        fvgs_15m=data["fvgs_15m"],
        fvgs_1h=data["fvgs_1h"],
        fib=data["fib"],
        key_levels=key_levels,
        eqh_eql=data["eqh_eql"],
        klines_15m=data["klines_15m"],
        now_ts=data["now_ts"],
    )

    def build_scenario(zone: KeyZone, direction: str) -> tuple:
        """返回 (entry_str, sl_str, tp1_str, tp2_str, condition_str)"""
        entry = zone.price
        if direction == "bullish":
            sl = zone.low - atr * 0.3
        else:
            sl = zone.high + atr * 0.3

        tp_data = find_tp_levels(
            entry=entry, sl=sl, direction=direction,
            obs_15m=data["obs_15m"], fvgs_15m=data["fvgs_15m"],
            key_levels=key_levels, eqh_eql=data["eqh_eql"],
            current_price=current_price,
        )

        zone_label = " + ".join(zone.labels[:3]) if zone.labels else "OB"
        entry_str = f"{entry:,.2f}（{zone_label}）"
        sl_str = f"{sl:,.2f}（框架外側 + ATR×0.3）"
        tp1_str = f"{tp_data['tp1']:,.2f}（{tp_data['tp1_label']}）RR 1:{tp_data['tp1_rr']:.1f}"
        tp2_str = f"{tp_data['tp2']:,.2f}（{tp_data['tp2_label']}）RR 1:{tp_data['tp2_rr']:.1f}"

        if direction == "bullish":
            cond = "等待 3M 實體陽線突破近期 Swing High（MSS 確認）"
        else:
            cond = "等待 3M 實體陰線跌破近期 Swing Low（MSS 確認）"

        return entry_str, sl_str, tp1_str, tp2_str, cond

    # 判斷顯示模式
    aligned = (
        (struct_4h == "bullish" and struct_1h == "bullish") or
        (struct_4h == "bearish" and struct_1h == "bearish")
    )
    main_dir = struct_1h if struct_1h != "ranging" else None

    header = (
        f"📊 {sym} {session_label}分析 [{now_str}]\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"{bias_emoji} 當前整體偏向：【{bias_text}】\n"
        f"關鍵水平：{kl_str}\n"
    )

    if is_low_liquidity():
        header += "⚠️ 低流動性時段，訊號可信度較低\n"

    if aligned and main_dir and bull_zones and bear_zones:
        # 同向模式：主方向完整，另一方向一行備注
        if main_dir == "bullish":
            main_zone = bull_zones[0]
            entry_s, sl_s, tp1_s, tp2_s, cond_s = build_scenario(main_zone, "bullish")
            counter_zone = bear_zones[0] if bear_zones else None

            body = (
                f"\n🟢 看漲情景（主路線）：\n"
                f"   🎯 TP2：{tp2_s}\n"
                f"   🎯 TP1：{tp1_s}\n"
                f"   📍 入場：{entry_s}\n"
                f"   🛑 SL：{sl_s}\n"
                f"   入場條件：{cond_s}\n"
            )
            if counter_zone:
                body += f"\n🔴 看跌備用：若價格升至 {counter_zone.price:,.2f} 遇阻且 3M 結構轉跌，可考慮逆勢做空（建議半倉）\n"

        else:  # bearish
            main_zone = bear_zones[0]
            entry_s, sl_s, tp1_s, tp2_s, cond_s = build_scenario(main_zone, "bearish")
            counter_zone = bull_zones[0] if bull_zones else None

            body = (
                f"\n🔴 看跌情景（主路線）：\n"
                f"   🛑 SL：{sl_s}\n"
                f"   📍 入場：{entry_s}\n"
                f"   🎯 TP1：{tp1_s}\n"
                f"   🎯 TP2：{tp2_s}\n"
                f"   入場條件：{cond_s}\n"
            )
            if counter_zone:
                body += f"\n🟢 看漲備用：若價格回踩 {counter_zone.price:,.2f} 且 3M 結構轉漲，可考慮逆勢做多（建議半倉）\n"

    else:
        # 平衡模式：兩個方向均等展示
        body = "\n"
        if bull_zones:
            bz = bull_zones[0]
            entry_s, sl_s, tp1_s, tp2_s, cond_s = build_scenario(bz, "bullish")
            body += (
                f"🟢 看漲情景：\n"
                f"   🎯 TP2：{tp2_s}\n"
                f"   🎯 TP1：{tp1_s}\n"
                f"   📍 入場：{entry_s}\n"
                f"   🛑 SL：{sl_s}\n"
                f"   入場條件：{cond_s}\n\n"
            )

        if bear_zones:
            bz = bear_zones[0]
            entry_s, sl_s, tp1_s, tp2_s, cond_s = build_scenario(bz, "bearish")
            body += (
                f"🔴 看跌情景：\n"
                f"   🛑 SL：{sl_s}\n"
                f"   📍 入場：{entry_s}\n"
                f"   🎯 TP1：{tp1_s}\n"
                f"   🎯 TP2：{tp2_s}\n"
                f"   入場條件：{cond_s}\n"
            )

        body += "\n⚠️ 方向尚未同步，兩個方向均以半倉位入場，等待 4H + 1H 結構同步後再加倉\n"

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
        tag = "🟢" if ob.direction == "bullish" else "🔴"
        label = f"1H {'看漲' if ob.direction == 'bullish' else '看跌'} OB"
        all_levels.append((ob.mid, label))

    all_levels.sort(key=lambda x: x[0], reverse=True)

    # 重新排列，把現價插入正確位置
    above = [(p, n) for p, n in all_levels if p > current_price]
    below = [(p, n) for p, n in all_levels if p <= current_price]
    above.sort(key=lambda x: x[0], reverse=True)
    below.sort(key=lambda x: x[0], reverse=True)

    current_line = f"   ──── 💲{current_price:,.2f} 現價 ────\n"

    levels_final = ""
    for price, name in above:
        levels_final += f"   🔴 {price:,.2f}  {name}\n"
    levels_final += current_line
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
# 功能五：掛單建議
# ─────────────────────────────────────────

def format_limit_order(data: dict) -> str:
    """生成掛單建議"""
    now_str = datetime.now(HKT).strftime("%m-%d %H:%M HKT")
    sym = data["symbol"]
    current_price = data["current_price"]
    atr = data["atr_15m"]
    struct_1h = data["struct_1h"]
    struct_4h = data["struct_4h"]

    # 確定方向（1H 為主）
    if struct_1h == "bullish":
        direction = "bullish"
    elif struct_1h == "bearish":
        direction = "bearish"
    else:
        return (
            f"📌 {sym} 掛單建議 [{now_str}]\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"⚪ 1H 結構橫盤，暫無明確掛單方向\n"
            f"建議等待 1H 結構明確後再設掛單"
        )

    # 找高分重疊區（掛單用更嚴格條件：分數 >= 2.5）
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
        eqh_eql=data["eqh_eql"],
        klines_15m=data["klines_15m"],
        now_ts=data["now_ts"],
    )

    # 過濾：入場位必須距現價 >= 0.5%（確保有回調空間）
    min_distance = current_price * 0.005
    valid_zones = []
    for z in zones:
        if direction == "bullish" and z.price < current_price - min_distance:
            valid_zones.append(z)
        elif direction == "bearish" and z.price > current_price + min_distance:
            valid_zones.append(z)

    if not valid_zones:
        return (
            f"📌 {sym} 掛單建議 [{now_str}]\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"⚠️ 目前無合適掛單位置（現價附近無足夠距離的關鍵位）\n"
            f"建議等待價格移動後再查詢"
        )

    best = valid_zones[0]
    entry = best.price

    # SL：框架外側 + ATR × 0.3
    if direction == "bullish":
        sl = best.low - atr * 0.3
    else:
        sl = best.high + atr * 0.3

    sl_dist = abs(entry - sl)

    # TP
    tp_data = find_tp_levels(
        entry=entry, sl=sl, direction=direction,
        obs_15m=data["obs_15m"], fvgs_15m=data["fvgs_15m"],
        key_levels=data["key_levels"], eqh_eql=data["eqh_eql"],
        current_price=current_price,
    )

    # 方向標注
    is_bull = direction == "bullish"
    dir_emoji = "🟢" if is_bull else "🔴"
    dir_text = "做多" if is_bull else "做空"

    if struct_4h == direction:
        prob_tag = f"✅ {dir_emoji} 方向：{dir_text}（高勝率（4H + 1H 同向））"
    else:
        prob_tag = f"🟡 {dir_emoji} 方向：{dir_text}（注意 4H {struct_4h}，尚未與 1H 同步）"

    zone_label = " + ".join(best.labels[:4]) if best.labels else "OB"
    expiry = get_limit_order_expiry()

    # 低流動性加寬 SL
    if is_low_liquidity():
        sl_extra = atr * 0.2
        if is_bull:
            sl -= sl_extra
        else:
            sl += sl_extra
        liq_note = "（含夜間加寬 ATR×0.2）"
    else:
        liq_note = ""

    # 取消條件
    if is_bull:
        cancel_price = entry - sl_dist * 0.5
        cancel_note = f"若價格未回調直接跌破 {cancel_price:,.2f}，請取消掛單"
        trade_block = (
            f"🎯 TP2：{tp_data['tp2']:,.2f}（{tp_data['tp2_label']}）  RR 1:{tp_data['tp2_rr']:.1f}\n"
            f"🎯 TP1：{tp_data['tp1']:,.2f}（{tp_data['tp1_label']}）  RR 1:{tp_data['tp1_rr']:.1f}\n"
            f"📍 入場：{entry:,.2f}（{zone_label}）\n"
            f"🛑 SL：{sl:,.2f}（框架底部外 + ATR×0.3{liq_note}）"
        )
    else:
        cancel_price = entry + sl_dist * 0.5
        cancel_note = f"若價格未反彈直接突破 {cancel_price:,.2f}，請取消掛單"
        trade_block = (
            f"🛑 SL：{sl:,.2f}（框架頂部外 + ATR×0.3{liq_note}）\n"
            f"📍 入場：{entry:,.2f}（{zone_label}）\n"
            f"🎯 TP1：{tp_data['tp1']:,.2f}（{tp_data['tp1_label']}）  RR 1:{tp_data['tp1_rr']:.1f}\n"
            f"🎯 TP2：{tp_data['tp2']:,.2f}（{tp_data['tp2_label']}）  RR 1:{tp_data['tp2_rr']:.1f}"
        )

    tp1_note = f"\n   {tp_data['tp1_note']}" if tp_data["tp1_note"] else ""

    msg = (
        f"📌 {sym} 掛單建議 [{now_str}]\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"{prob_tag}\n"
        f"🧩 重疊區域：{zone_label}（分數 {best.score:.1f}）\n"
        f"\n"
        f"{trade_block}{tp1_note}\n"
        f"\n"
        f"📊 RR（至 TP1）：1:{tp_data['tp1_rr']:.1f}\n"
        f"⏰ 有效期：{expiry}\n"
        f"⚠️ {cancel_note}"
    )

    if is_low_liquidity():
        msg += "\n⚠️ 低流動性時段，SL 已自動加寬"

    return msg
