"""
Bot v7.0 主程式
AI 驅動的加密貨幣交易訊號機械人
- 每 3 分鐘輪流偵測 BTC/ETH/SOL（每幣每 9 分鐘偵測一次）
- 6 個按鈕：BTC分析 / ETH分析 / SOL分析 / BTC掛單 / ETH掛單 / SOL掛單
- 有入場訊號時自動發送 Telegram 通知
"""
import os
import asyncio
import logging
import pytz
from datetime import datetime
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler,
    ContextTypes
)

from core_engine import fetch_market_data, HKT
from ai_analyzer import analyze, has_entry_signal

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID   = int(os.environ.get("TELEGRAM_CHAT_ID", "0"))

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

SYMBOL_LABELS = {
    "BTCUSDT": "BTC",
    "ETHUSDT": "ETH",
    "SOLUSDT": "SOL",
}

_scan_index = 0
_last_signal_time: dict[str, float] = {}
SIGNAL_COOLDOWN = 30 * 60  # 30 分鐘

def main_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("📊 BTC 分析", callback_data="analysis_BTCUSDT"),
            InlineKeyboardButton("📊 ETH 分析", callback_data="analysis_ETHUSDT"),
            InlineKeyboardButton("📊 SOL 分析", callback_data="analysis_SOLUSDT"),
        ],
        [
            InlineKeyboardButton("📌 BTC 掛單", callback_data="limit_BTCUSDT"),
            InlineKeyboardButton("📌 ETH 掛單", callback_data="limit_ETHUSDT"),
            InlineKeyboardButton("📌 SOL 掛單", callback_data="limit_SOLUSDT"),
        ],
    ])

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    now = datetime.now(HKT).strftime("%m-%d %H:%M HKT")
    await update.message.reply_text(
        f"🤖 *Bot v7.0 已啟動* [{now}]\n\n"
        "AI 驅動 ICT/SMC 交易訊號機械人\n"
        "每 9 分鐘自動偵測 BTC/ETH/SOL 入場機會\n\n"
        "請選擇功能：",
        parse_mode="Markdown",
        reply_markup=main_keyboard()
    )

async def cmd_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "請選擇功能：",
        reply_markup=main_keyboard()
    )

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    data = query.data
    parts = data.split("_", 1)
    if len(parts) != 2:
        return

    mode, symbol = parts[0], parts[1]
    label = SYMBOL_LABELS.get(symbol, symbol)

    wait_msg = await query.message.reply_text(
        f"⏳ 正在分析 {label}，請稍候（AI 分析約需 10-20 秒）..."
    )

    try:
        market_data = fetch_market_data(symbol)
        if not market_data:
            await wait_msg.edit_text(f"❌ 無法取得 {label} 數據，請稍後再試")
            return

        result = analyze(market_data, mode=mode)
        await wait_msg.delete()

        if len(result) <= 4096:
            await query.message.reply_text(result, reply_markup=main_keyboard())
        else:
            chunks = [result[i:i+4000] for i in range(0, len(result), 4000)]
            for i, chunk in enumerate(chunks):
                if i == len(chunks) - 1:
                    await query.message.reply_text(chunk, reply_markup=main_keyboard())
                else:
                    await query.message.reply_text(chunk)

    except Exception as e:
        logger.error(f"按鈕回調錯誤 {symbol}: {e}")
        await wait_msg.edit_text(f"❌ 分析失敗：{str(e)[:100]}")

async def auto_scan(context: ContextTypes.DEFAULT_TYPE):
    global _scan_index

    if not CHAT_ID:
        return

    symbol = SYMBOLS[_scan_index % len(SYMBOLS)]
    _scan_index += 1
    label = SYMBOL_LABELS.get(symbol, symbol)

    logger.info(f"自動偵測 {symbol}...")

    try:
        market_data = fetch_market_data(symbol)
        if not market_data:
            logger.warning(f"無法取得 {symbol} 數據")
            return

        result = analyze(market_data, mode="analysis")

        if has_entry_signal(result):
            import time
            now = time.time()
            last = _last_signal_time.get(symbol, 0)
            if now - last < SIGNAL_COOLDOWN:
                logger.info(f"{symbol} 訊號冷卻中，跳過")
                return

            _last_signal_time[symbol] = now
            logger.info(f"🚨 {symbol} 入場訊號！發送 Telegram 通知")

            header = f"🚨 *【自動偵測訊號】{label}*\n\n"
            full_msg = header + result

            if len(full_msg) <= 4096:
                await context.bot.send_message(
                    chat_id=CHAT_ID,
                    text=full_msg,
                    parse_mode="Markdown",
                    reply_markup=main_keyboard()
                )
            else:
                chunks = [full_msg[i:i+4000] for i in range(0, len(full_msg), 4000)]
                for i, chunk in enumerate(chunks):
                    if i == len(chunks) - 1:
                        await context.bot.send_message(
                            chat_id=CHAT_ID,
                            text=chunk,
                            parse_mode="Markdown",
                            reply_markup=main_keyboard()
                        )
                    else:
                        await context.bot.send_message(
                            chat_id=CHAT_ID,
                            text=chunk,
                            parse_mode="Markdown"
                        )
        else:
            logger.info(f"{symbol} 暫無入場訊號")

    except Exception as e:
        logger.error(f"自動偵測錯誤 {symbol}: {e}")

def main():
    if not BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN 未設定")
        return

    app = (
        Application.builder()
        .token(BOT_TOKEN)
        .build()
    )

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("menu",  cmd_menu))
    app.add_handler(CallbackQueryHandler(button_callback))

    app.job_queue.run_repeating(
        auto_scan,
        interval=180,   
        first=30,       
    )

    logger.info("Bot v7.0 啟動中...")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
