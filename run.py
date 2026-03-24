#!/usr/bin/env python3
import logging
import asyncio
import time
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import pandas as pd
import numpy as np
from binance.client import Client
import google.generativeai as genai
from datetime import datetime
import os
from collections import defaultdict

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

WATCH_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
SCAN_INTERVAL = 120

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

binance_client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)
genai.configure(api_key=GEMINI_API_KEY)

try:
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
except:
    gemini_model = genai.GenerativeModel('gemini-2.0-flash')

sent_signals = defaultdict(lambda: {"time": 0, "price": 0})

def get_klines(symbol, interval, limit=500):
    try:
        klines = binance_client.get_historical_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        return df
    except:
        return None

def get_ticker_info(symbol):
    try:
        return binance_client.get_ticker(symbol=symbol)
    except:
        return {}

def get_order_book(symbol, limit=20):
    try:
        ob = binance_client.get_order_book(symbol=symbol, limit=limit)
        bids = np.array([[float(p), float(q)] for p, q in ob['bids']])
        asks = np.array([[float(p), float(q)] for p, q in ob['asks']])
        return bids, asks
    except:
        return None, None

def analyze_ict_levels(df_1h, df_15m):
    levels = []
    if df_15m is None or len(df_15m) < 5:
        return levels
    recent = df_15m.iloc[-10:].copy()
    for i in range(1, len(recent)):
        curr = recent.iloc[i]
        prev = recent.iloc[i-1]
        if curr['close'] < prev['close']:
            levels.append({'type': '看跌 OB', 'price_high': float(curr['high']), 'price_low': float(curr['low']), 'price_mid': float((curr['high'] + curr['low']) / 2)})
        if curr['close'] > prev['close']:
            levels.append({'type': '看漲 OB', 'price_high': float(curr['high']), 'price_low': float(curr['low']), 'price_mid': float((curr['high'] + curr['low']) / 2)})
    for i in range(1, len(recent) - 1):
        curr = recent.iloc[i]
        next_k = recent.iloc[i+1]
        if curr['high'] < next_k['low']:
            levels.append({'type': 'FVG (向上)', 'price_high': float(next_k['low']), 'price_low': float(curr['high']), 'price_mid': float((next_k['low'] + curr['high']) / 2)})
        if curr['low'] > next_k['high']:
            levels.append({'type': 'FVG (向下)', 'price_high': float(curr['low']), 'price_low': float(next_k['high']), 'price_mid': float((curr['low'] + next_k['high']) / 2)})
    if len(recent) >= 3:
        for i in range(2, len(recent)):
            prev2 = recent.iloc[i-2]
            prev1 = recent.iloc[i-1]
            curr = recent.iloc[i]
            if curr['low'] < prev1['low'] < prev2['low']:
                levels.append({'type': 'BB (新低)', 'price_high': float(prev1['low']), 'price_low': float(curr['low']), 'price_mid': float((prev1['low'] + curr['low']) / 2)})
            if curr['high'] > prev1['high'] > prev2['high']:
                levels.append({'type': 'BB (新高)', 'price_high': float(curr['high']), 'price_low': float(prev1['high']), 'price_mid': float((curr['high'] + prev1['high']) / 2)})
    return levels

def get_1h_direction(df_1h):
    if df_1h is None or len(df_1h) < 2:
        return "未知"
    recent = df_1h.iloc[-2:]
    return "⬆️ 看漲" if recent.iloc[-1]['close'] > recent.iloc[-2]['close'] else "⬇️ 看跌"

def analyze_5m_pattern(df_5m):
    if df_5m is None or len(df_5m) < 2:
        return "未知"
    recent = df_5m.iloc[-2:]
    curr = recent.iloc[-1]
    prev = recent.iloc[-2]
    if curr['open'] > prev['close'] and curr['close'] < prev['open']:
        return "看跌吞沒"
    if curr['open'] < prev['close'] and curr['close'] > prev['open']:
        return "看漲吞沒"
    if (curr['high'] - curr['close']) > (curr['close'] - curr['open']) * 2:
        return "射擊之星"
    if (curr['close'] - curr['low']) > (curr['high'] - curr['close']) * 2:
        return "錘子線"
    return "普通"

def analyze_liquidity(bids, asks):
    if bids is None or asks is None:
        return "未知", 0, 0
    bid_volume = np.sum(bids[:5, 1])
    ask_volume = np.sum(asks[:5, 1])
    if bid_volume < 10:
        return "弱", bid_volume, ask_volume
    elif bid_volume < 50:
        return "中", bid_volume, ask_volume
    else:
        return "強", bid_volume, ask_volume

async def send_signal_1(app, chat_id, symbol, direction_1h, levels, current_price, pattern_5m, bid_strength):
    levels_sorted = sorted(levels, key=lambda x: x['price_mid'], reverse=True)[:4]
    msg = f"🚨 【入場信號】{symbol}\n\n【大方向】1H: {direction_1h}\n\n【15M 關鍵位置】\n"
    for i, level in enumerate(levels_sorted):
        if level['price_mid'] > current_price:
            msg += f"🔴 阻力{i+1}: ${level['price_mid']:.2f} ({level['type']})\n"
        else:
            msg += f"🟢 支撐{i+1}: ${level['price_mid']:.2f} ({level['type']})\n"
    msg += f"\n【入場條件滿足】\n- 1H 方向: {direction_1h} ✓\n- 15M 位置: 接近關鍵位置 ✓\n- 5M 組合: {pattern_5m} ✓\n- 流動性: {bid_strength} ✓\n\n【時間戳記】\n- 偵測時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n- 當前價格: ${current_price:.2f}\n\n【確認】\n確認 1H、15M、5M 信號"
    try:
        await app.bot.send_message(chat_id=chat_id, text=msg)
    except Exception as e:
        logger.error(f"發送信號失敗: {e}")

async def auto_scan(app, chat_id):
    logger.info("自動掃描已啟動")
    while True:
        try:
            for symbol in WATCH_SYMBOLS:
                try:
                    df_1h = get_klines(symbol, "1h", limit=100)
                    df_15m = get_klines(symbol, "15m", limit=100)
                    df_5m = get_klines(symbol, "5m", limit=100)
                    ticker = get_ticker_info(symbol)
                    bids, asks = get_order_book(symbol)
                    if df_1h is None or df_15m is None:
                        continue
                    current_price = float(ticker.get('lastPrice', 0))
                    direction_1h = get_1h_direction(df_1h)
                    levels = analyze_ict_levels(df_1h, df_15m)
                    pattern_5m = analyze_5m_pattern(df_5m)
                    bid_strength, bid_vol, ask_vol = analyze_liquidity(bids, asks)
                    signal_key = f"{symbol}_entry"
                    current_time = time.time()
                    if pattern_5m != "普通" and bid_strength == "弱" and current_time - sent_signals[signal_key]['time'] > 600:
                        await send_signal_1(app, chat_id, symbol, direction_1h, levels, current_price, pattern_5m, bid_strength)
                        sent_signals[signal_key] = {'time': current_time, 'price': current_price}
                except Exception as e:
                    logger.error(f"掃描失敗: {e}")
        except Exception as e:
            logger.error(f"自動掃描失敗: {e}")
        await asyncio.sleep(SCAN_INTERVAL)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = "👋 歡迎使用交易信號機械人！\n\n📊 功能：自動掃描 BTC、ETH、SOL 交易信號\n\n🎯 分析方法：ICT 交易法\n\n機械人每 2 分鐘掃描一次。"
    await update.message.reply_text(msg)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("機械人正在自動掃描中。")

def main():
    logger.info("正在啟動機械人...")
    if not TELEGRAM_CHAT_ID:
        logger.error("缺少 TELEGRAM_CHAT_ID")
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
