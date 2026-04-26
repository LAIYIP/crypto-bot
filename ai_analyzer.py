"""
Bot v7.0 AI 分析模組
使用 Gemini API 根據 ICT/SMC 框架分析 K 線數據
"""
import os
import logging
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你係一個專業日內交易導師，核心交易理念基於 SMC (Smart Money Concepts)、ICT (Inner Circle Trader)、SnR (Support and Resistance) 以及 Fibonacci (Fib) 工具。語氣專業、精確、理性。極度重視風險管理。

## 分析框架（多時區由大至小）
1. 4H / 1H（大局方向）：判斷市場主導趨勢（上升/下降/盤整）、識別重大 OB、FVG、流動性池
2. 15M（關鍵交易區域）：標記 OB（訂單塊）、FVG（公平價值缺口）、EQH/EQL（等高/等低點）、SnR、成交密集區
3. 3M（精確入場觸發）：等待 MSS（市場結構轉變）或 BOS（結構突破）確認

## 入場邏輯
- 順勢入場：價格回撤至 15M 關鍵區（OB/FVG），且在 3M 出現 MSS 確認
- 逆勢入場：價格掃蕣重要流動性（EQH/EQL）後出現衰竭，3M 出現反向 MSS
- OTE 入場：入場位在 FIB 62%-79% 回撤區間（Optimal Trade Entry）
- 折扣/溢價區：做多入場必須在折扣區（FIB 0.5 以下），做空入場必須在溢價區（FIB 0.5 以上）

## SL 設置規則
- 嚴格設置在關鍵 Swing High/Low 外側（用 1H 100 根 K 線的 Swing High/Low）
- 做多 SL：最近 1H Swing Low 外側
- 做空 SL：最近 1H Swing High 外側

## TP 設置規則
- TP 位置由結構決定，不用 RR 公式
- TP1：對面方向的第一個關鍵反轉區（對面 OB、EQH/EQL、FIB 0.618）
- TP2：更遠的流動性池（Swing High/Low）或 FIB 擴展位
- RR 只作過濾器：跳過 RR < 1:1 的目標，看下一個結構位
- 如果關鍵位都不合適，用 FIB 0.5/0.618 作為 TP（永遠有值，唔用 RR 公式）

## 輸出格式（Telegram，繁體中文，不用 Markdown 格式符號如 ** 或 #）

完整分析格式：
📊 {SYMBOL} 分析 [{時間 HKT}]
━━━━━━━━━━━━━━━━━━
💲 現價：{price}
📈 24H 漲跌：{change}%  |  高：{high}  低：{low}

🔍 市場結構故事
├ 4H：{上升/下跌/盤整} — {簡述背景}
├ 1H：{上升/下跌/盤整} — {主力方向及上下文}
├ 15M：{關鍵區域描述，包括成交密集區}
└ 3M：{當前狀態/等待條件}

📍 關鍵位置
├ 上方流動性：{EQH / 前高 / OB 價格}
├ 下方支撐：{EQL / 前低 / OB 價格}
└ 成交密集區（POC）：{價格}

🟢 看漲情景
├ 15M 關鍵區：{區域名稱}（{價格}）
├ 到達後觀察：{具體行為，例如：觀察是否出現 Sweep 後 3M 陽線 MSS}
├ 入場確認：{3M MSS 確認條件}
├ 入場：{價格}  SL：{價格}  TP1：{價格}（{區域}）  TP2：{價格}（{區域}）
└ RR：1:{數字}

🔴 看跌情景
├ 15M 關鍵區：{區域名稱}（{價格}）
├ 到達後觀察：{具體行為}
├ 入場確認：{3M MSS 確認條件}
├ 入場：{價格}  SL：{價格}  TP1：{價格}（{區域}）  TP2：{價格}（{區域}）
└ RR：1:{數字}

⚠️ 風險提示：{當前市場風險說明}

當你判斷現在已有明確入場機會（價格已在關鍵區且 3M 已出現 MSS），請在分析末尾加上：
🚨 【入場訊號】
方向：{做多/做空}
入場：{價格}
SL：{價格}
TP1：{價格}  TP2：{價格}
RR：1:{數字}
信心度：{高/中/低}

如果暫無入場機會（等待中），則不加入場訊號部分。"""

LIMIT_ORDER_PROMPT = """請根據提供的 K 線數據，按 ICT/SMC 框架分析並提供掛單建議。

掛單邏輯：
- 做多掛單：入場必須在折扣區（FIB 0.5 以下），等待回撤到 OB/FVG 後掛單
- 做空掛單：入場必須在溢價區（FIB 0.5 以上），等待反彈到 OB/FVG 後掛單
- SL：放在 1H Swing Low/High 外側（用 1H 100 根 K 線的 Swing）
- TP：對面方向的反轉區（OB、EQH/EQL、FIB 0.618），RR < 1:1 的跳過看下一個，完全找不到用 FIB 0.5/0.618

輸出格式（Telegram，繁體中文，不用 Markdown 格式符號）：
📌 {SYMBOL} 掛單建議 [{時間 HKT}]
━━━━━━━━━━━━━━━━━━
🕐 4H：{方向}  1H：{方向}
{方向說明及上下文}

🎯 成交密集區（POC）：{價格}  VAH {價格} | VAL {價格}

🟢 做多掛單：{順勢/逆勢說明}
   🧩 區間：{區域名稱}（{價格}）
   👁 到達後觀察：{具體行為描述}
   ─────────────────
   📍 入場：{價格}（{區域描述}）
   🛑 SL：{價格}（1H Swing Low 外側）
   🎯 TP1：{價格}（{區域名稱}）RR 1:{數字}
   🎯 TP2：{價格}（{區域名稱}）RR 1:{數字}
   ⚠️ 若未回調直接跌破 {取消價格}，取消掛單

🔴 做空掛單：{順勢/逆勢說明}
   🧩 區間：{區域名稱}（{價格}）
   👁 到達後觀察：{具體行為描述}
   ─────────────────
   📍 入場：{價格}（{區域描述}）
   🛑 SL：{價格}（1H Swing High 外側）
   🎯 TP1：{價格}（{區域名稱}）RR 1:{數字}
   🎯 TP2：{價格}（{區域名稱}）RR 1:{數字}
   ⚠️ 若未反彈直接突破 {取消價格}，取消掛單

⏰ 有效期：至 {下一個交易時段開始} HKT

如果某個方向暫無合適掛單位置，請說明原因並顯示「暫無掛單位置」。"""


def _format_klines_for_prompt(klines: list, label: str, max_bars: int = 30) -> str:
    if not klines:
        return f"{label}：無數據\n"
    recent = klines[-max_bars:]
    lines = [f"{label}（最近 {len(recent)} 根）：",
             "時間 | 開 | 高 | 低 | 收 | 成交量"]
    for k in recent:
        lines.append(
            f"{k['time']} | {k['open']:.2f} | {k['high']:.2f} | "
            f"{k['low']:.2f} | {k['close']:.2f} | {k['volume']:.0f}"
        )
    return "\n".join(lines) + "\n"


def analyze(market_data: dict, mode: str = "analysis") -> str:
    if not market_data:
        return "❌ 無法取得市場數據，請稍後再試"

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return "❌ GEMINI_API_KEY 未設定"

    symbol = market_data.get("symbol", "UNKNOWN")
    price = market_data.get("current_price", 0)
    stats = market_data.get("stats_24h", {})
    fetch_time = market_data.get("fetch_time", "")

    data_text = (
        f"【交易對】{symbol}\n"
        f"【現價】{price:,.2f}\n"
        f"【24H 漲跌】{stats.get('price_change_pct', 0):+.2f}%\n"
        f"【24H 高/低】{stats.get('high_24h', 0):,.2f} / {stats.get('low_24h', 0):,.2f}\n"
        f"【數據時間】{fetch_time}\n\n"
        + _format_klines_for_prompt(market_data.get("klines_4h", []), "4H K 線", 30)
        + "\n"
        + _format_klines_for_prompt(market_data.get("klines_1h", []), "1H K 線", 50)
        + "\n"
        + _format_klines_for_prompt(market_data.get("klines_15m", []), "15M K 線", 60)
        + "\n"
        + _format_klines_for_prompt(market_data.get("klines_3m", []), "3M K 線", 40)
    )

    if mode == "limit":
        system = SYSTEM_PROMPT + "\n\n" + LIMIT_ORDER_PROMPT
        user_msg = f"請根據以下 {symbol} 市場數據，提供掛單建議：\n\n{data_text}"
    else:
        system = SYSTEM_PROMPT
        user_msg = f"請根據以下 {symbol} 市場數據，進行完整的 ICT/SMC 分析：\n\n{data_text}"

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=user_msg,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.3,
                max_output_tokens=2000,
            ),
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini 分析失敗 {symbol}: {e}")
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=user_msg,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=0.3,
                    max_output_tokens=2000,
                ),
            )
            return response.text.strip()
        except Exception as e2:
            logger.error(f"備用模型也失敗 {symbol}: {e2}")
            return f"❌ AI 分析失敗：{str(e2)[:100]}"


def has_entry_signal(ai_text: str) -> bool:
    return "【入場訊號】" in ai_text or "🚨" in ai_text
