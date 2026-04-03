"""
Telegram Crypto Bot v6.0
功能一：自動入場訊號（BTC/ETH/SOL，每 3 分鐘掃描）
功能二：定時盤勢分析（每日 5 次，BTC）
功能三：按需詳細報告（打幣種名）
功能四：即時雙向分析（打「BTC分析」）
功能五：掛單建議（打「BTC掛單」）
新功能：Telegram 持久按鈕目錄 + 48 小時訊息清理
"""

import os
import sys
import asyncio
import logging
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(__file__))

from telegram import (
    Update,
    ReplyKeyboardMarkup,
    KeyboardButton,
)
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

from core_engine import analyze_symbol, HKT
from signals import (
    generate_auto_signal,
    format_auto_signal,
    format_directional_analysis,
    format_on_demand_report,
    format_limit_order,
    get_session_label,
)

# ─────────────────────────────────────────
# 設定
# ─────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = int(os.environ.get("TELEGRAM_CHAT_ID", "0"))

# 自動掃描幣種
AUTO_SCAN_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

# 定時分析時間（HKT）
SCHEDULED_TIMES = [
    (8, 0, "早盤"),
    (12, 0, "午盤"),
    (17, 0, "歐洲盤"),
    (20, 30, "美盤"),
    (23, 30, "深夜盤"),
]

# 幣種名稱對應
SYMBOL_MAP = {
    "BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT",
    "BNB": "BNBUSDT", "XRP": "XRPUSDT", "ADA": "ADAUSDT",
    "DOGE": "DOGEUSDT", "AVAX": "AVAXUSDT", "DOT": "DOTUSDT",
    "MATIC": "MATICUSDT", "LINK": "LINKUSDT", "UNI": "UNIUSDT",
    "ATOM": "ATOMUSDT", "LTC": "LTCUSDT", "BCH": "BCHUSDT",
}

# 訊息 ID 記錄（用於 48 小時清理）
message_log: list[tuple[int, datetime]] = []  # [(msg_id, sent_time)]

# 上次訊號記錄（防重複，30 分鐘內同方向不重複發）
last_signal: dict[str, tuple[str, datetime]] = {}  # {symbol: (direction, time)}

# ─────────────────────────────────────────
# 持久按鈕鍵盤
# ─────────────────────────────────────────

MAIN_KEYBOARD = ReplyKeyboardMarkup(
    [
        [
            KeyboardButton("📊 BTC分析"),
            KeyboardButton("📌 BTC掛單"),
            KeyboardButton("📋 BTC報告"),
        ],
        [
            KeyboardButton("📊 ETH分析"),
            KeyboardButton("📌 ETH掛單"),
            KeyboardButton("📋 ETH報告"),
        ],
        [
            KeyboardButton("📊 SOL分析"),
            KeyboardButton("📌 SOL掛單"),
            KeyboardButton("📋 SOL報告"),
        ],
    ],
    resize_keyboard=True,
    is_persistent=True,
)

HELP_TEXT = (
    "📱 其他幣種查詢：直接輸入幣種名稱\n"
    "例如：BNB、XRP、ADA、DOGE、AVAX\n"
    "加上「分析」→ 雙向情景分析\n"
    "加上「掛單」→ 掛單建議\n"
    "（不加後綴）→ 詳細報告"
)

# ─────────────────────────────────────────
# 輔助函數
# ─────────────────────────────────────────

async def send_msg(bot, text: str, parse_mode: str = None) -> int | None:
    """發送訊息並記錄 ID"""
    try:
        msg = await bot.send_message(
            chat_id=CHAT_ID,
            text=text,
            parse_mode=parse_mode,
        )
        message_log.append((msg.message_id, datetime.now(HKT)))
        return msg.message_id
    except Exception as e:
        logger.error(f"發送訊息失敗: {e}")
        return None


async def reply_msg(update: Update, text: str) -> None:
    """回覆訊息並記錄 ID"""
    try:
        msg = await update.message.reply_text(
            text,
            reply_markup=MAIN_KEYBOARD,
        )
        message_log.append((msg.message_id, datetime.now(HKT)))
    except Exception as e:
        logger.error(f"回覆訊息失敗: {e}")


def get_symbol_from_text(text: str) -> str | None:
    """從文字中提取幣種代碼"""
    text_upper = text.upper().strip()
    # 移除後綴
    for suffix in ["分析", "掛單", "報告", "LIMIT", "ANALYSIS", "REPORT"]:
        text_upper = text_upper.replace(suffix, "").strip()
    # 移除 emoji
    for emoji in ["📊", "📌", "📋"]:
        text_upper = text_upper.replace(emoji, "").strip()

    return SYMBOL_MAP.get(text_upper)


def is_duplicate_signal(symbol: str, direction: str) -> bool:
    """檢查是否在 30 分鐘內已發過同方向訊號"""
    if symbol in last_signal:
        last_dir, last_time = last_signal[symbol]
        if last_dir == direction:
            elapsed = (datetime.now(HKT) - last_time).total_seconds()
            if elapsed < 1800:  # 30 分鐘
                return True
    return False


# ─────────────────────────────────────────
# 指令處理
# ─────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """處理 /start 指令"""
    welcome = (
        "👋 歡迎使用 ICT/SMC 加密貨幣分析 Bot v6.0\n\n"
        "📊 功能說明：\n"
        "• 自動入場訊號（BTC/ETH/SOL，每 3 分鐘掃描）\n"
        "• 定時盤勢分析（每日 5 次：08:00, 12:00, 17:00, 20:30, 23:30）\n"
        "• 按需詳細報告（輸入幣種名稱）\n"
        "• 即時雙向分析（輸入「BTC分析」）\n"
        "• 掛單建議（輸入「BTC掛單」）\n\n"
        f"{HELP_TEXT}"
    )
    await update.message.reply_text(welcome, reply_markup=MAIN_KEYBOARD)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """處理 /help 指令"""
    await update.message.reply_text(HELP_TEXT, reply_markup=MAIN_KEYBOARD)


# ─────────────────────────────────────────
# 訊息處理
# ─────────────────────────────────────────

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """處理用戶輸入"""
    if not update.message or not update.message.text:
        return

    text = update.message.text.strip()
    text_upper = text.upper()

    # 判斷操作類型
    is_analysis = any(s in text for s in ["分析", "ANALYSIS"])
    is_limit = any(s in text for s in ["掛單", "LIMIT"])
    is_report = any(s in text for s in ["報告", "REPORT"])

    # 提取幣種
    symbol = get_symbol_from_text(text)

    if not symbol:
        # 未識別的輸入
        await reply_msg(update,
            f"❓ 未識別的指令：{text}\n\n{HELP_TEXT}"
        )
        return

    coin_name = text_upper.replace("分析", "").replace("掛單", "").replace("報告", "")
    coin_name = coin_name.replace("📊", "").replace("📌", "").replace("📋", "").strip()

    await update.message.reply_text(f"⏳ 正在分析 {coin_name}，請稍候...")

    # 拉取數據
    data = analyze_symbol(symbol)
    if not data:
        await reply_msg(update, f"❌ 無法取得 {coin_name} 數據，請稍後再試")
        return

    # 生成對應報告
    if is_analysis:
        msg = format_directional_analysis(data)
    elif is_limit:
        msg = format_limit_order(data)
    elif is_report:
        msg = format_on_demand_report(data)
    else:
        # 預設：詳細報告
        msg = format_on_demand_report(data)

    await reply_msg(update, msg)


# ─────────────────────────────────────────
# 背景任務：自動訊號掃描
# ─────────────────────────────────────────

async def signal_scan_loop(bot) -> None:
    """每 3 分鐘掃描 BTC/ETH/SOL，偵測入場訊號"""
    logger.info("自動訊號掃描已啟動（每 3 分鐘）")
    while True:
        try:
            for symbol in AUTO_SCAN_SYMBOLS:
                data = analyze_symbol(symbol)
                if not data:
                    continue

                sig = generate_auto_signal(data)
                if sig is None:
                    continue

                # 防重複：30 分鐘內同方向不重複發
                if is_duplicate_signal(symbol, sig["direction"]):
                    continue

                # 記錄並發送
                last_signal[symbol] = (sig["direction"], datetime.now(HKT))
                msg = format_auto_signal(sig)
                await send_msg(bot, msg)
                logger.info(f"發出訊號：{symbol} {sig['direction']}")

                # 避免 API 限速
                await asyncio.sleep(2)

        except Exception as e:
            logger.error(f"訊號掃描錯誤: {e}")

        await asyncio.sleep(180)  # 3 分鐘


# ─────────────────────────────────────────
# 背景任務：定時盤勢分析
# ─────────────────────────────────────────

async def scheduled_analysis_loop(bot) -> None:
    """每日 5 個固定時間發送 BTC 雙向分析"""
    logger.info("定時分析已啟動")
    sent_today: set[tuple[int, int]] = set()

    while True:
        try:
            now_hkt = datetime.now(HKT)
            current_time = (now_hkt.hour, now_hkt.minute)

            for hour, minute, session in SCHEDULED_TIMES:
                key = (hour, minute)
                if current_time == key and key not in sent_today:
                    # 重置每日記錄（新的一天）
                    if now_hkt.hour == 0 and now_hkt.minute == 0:
                        sent_today.clear()

                    data = analyze_symbol("BTCUSDT")
                    if data:
                        msg = format_directional_analysis(data, session_label=session)
                        await send_msg(bot, msg)
                        sent_today.add(key)
                        logger.info(f"定時分析已發送：{session}")

        except Exception as e:
            logger.error(f"定時分析錯誤: {e}")

        await asyncio.sleep(60)  # 每分鐘檢查一次


# ─────────────────────────────────────────
# 背景任務：48 小時訊息清理
# ─────────────────────────────────────────

async def message_cleanup_loop(bot) -> None:
    """每小時清理超過 48 小時的舊訊息"""
    logger.info("訊息清理已啟動（48 小時）")
    while True:
        try:
            now = datetime.now(HKT)
            cutoff = now - timedelta(hours=48)
            to_delete = [(mid, t) for mid, t in message_log if t < cutoff]

            for msg_id, _ in to_delete:
                try:
                    await bot.delete_message(chat_id=CHAT_ID, message_id=msg_id)
                    await asyncio.sleep(0.5)  # 避免 API 限速
                except Exception:
                    pass  # 訊息可能已被手動刪除

            # 從記錄中移除已刪除的訊息
            for item in to_delete:
                if item in message_log:
                    message_log.remove(item)

            if to_delete:
                logger.info(f"已清理 {len(to_delete)} 條舊訊息")

        except Exception as e:
            logger.error(f"訊息清理錯誤: {e}")

        await asyncio.sleep(3600)  # 每小時執行一次


# ─────────────────────────────────────────
# 啟動後初始化
# ─────────────────────────────────────────

async def post_init(app: Application) -> None:
    """Bot 啟動後初始化背景任務"""
    bot = app.bot

    # 啟動背景任務
    asyncio.create_task(signal_scan_loop(bot))
    asyncio.create_task(scheduled_analysis_loop(bot))
    asyncio.create_task(message_cleanup_loop(bot))

    # 發送啟動通知
    now_str = datetime.now(HKT).strftime("%m-%d %H:%M HKT")
    await send_msg(
        bot,
        f"✅ Bot v6.0 已啟動 [{now_str}]\n"
        f"• 自動訊號掃描：BTC/ETH/SOL（每 3 分鐘）\n"
        f"• 定時分析：08:00, 12:00, 17:00, 20:30, 23:30\n"
        f"• 按鈕目錄已啟用"
    )
    logger.info("Bot v6.0 啟動完成")


# ─────────────────────────────────────────
# 主程式
# ─────────────────────────────────────────

def main() -> None:
    if not BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN 未設定")
        return

    app = (
        Application.builder()
        .token(BOT_TOKEN)
        .post_init(post_init)
        .build()
    )

    # 指令處理
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))

    # 訊息處理（所有文字訊息）
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )

    logger.info("Bot 正在啟動...")
    app.run_polling(
        allowed_updates=["message"],
        drop_pending_updates=True,
    )


if __name__ == "__main__":
    main()
