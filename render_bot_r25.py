import asyncio, json, os, logging
from datetime import datetime, timezone, date
import numpy as np
import pandas as pd
import websockets
import aiohttp
from aiohttp import web

# ============================================================
# CONFIGURATION
# ============================================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TOKEN_HERE")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID_HERE")

ENDPOINT = "wss://api.derivws.com/trading/v1/options/ws/public?app_id=1089"
M15_GRAN = 900
SYMBOLS = ["R_25"] # Dedicated to Volatility 25 Index

ATR_PERIOD = 14
MAX_BARS_OPEN = 20 # Mean reversion must be fast
GRADE_THRESHOLDS = {"A+": 2.5, "A": 2.0, "B": 1.5, "C": 1.0}

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# ============================================================
# DAILY JOURNAL TRACKER
# ============================================================
daily_counts = {"A+": 0, "A": 0, "B": 0, "C": 0}
last_summary_date = datetime.now(timezone.utc).date() 

# ============================================================
# V3.2 STRATEGY LOGIC (Bollinger Mean Reversion)
# ============================================================

def atr(df, p):
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"]-df["low"], (df["high"]-pc).abs(), (df["low"]-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/p, adjust=False).mean()

def bollinger_bands(df, window=20, num_std=2.0):
    sma = df["close"].rolling(window=window).mean()
    std = df["close"].rolling(window=window).std()
    df["bb_mid"] = sma
    df["bb_upper"] = sma + (std * num_std)
    df["bb_lower"] = sma - (std * num_std)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma
    return df

def detect_fvgs(df):
    fvgs = []
    highs, lows = df["high"].values, df["low"].values
    for i in range(2, len(df)):
        if lows[i] > highs[i-2]:
            fvgs.append({"type": "bull", "low": float(highs[i-2]), "high": float(lows[i]), "formed_at": i})
        elif highs[i] < lows[i-2]:
            fvgs.append({"type": "bear", "low": float(highs[i]), "high": float(lows[i-2]), "formed_at": i})
    return fvgs

def detect_obs(df):
    obs = []
    for i in range(2, len(df)):
        prev = df.iloc[i - 1]
        if df.iloc[i]["close"] < prev["open"]: # Bullish OB (Down candle before up move)
            zl, zh = float(min(prev["open"], prev["close"])), float(max(prev["open"], prev["close"]))
            obs.append({"type": "bull", "low": zl, "high": zh, "formed_at": i - 1})
        elif df.iloc[i]["close"] > prev["open"]: # Bearish OB
            zl, zh = float(min(prev["open"], prev["close"])), float(max(prev["open"], prev["close"]))
            obs.append({"type": "bear", "low": zl, "high": zh, "formed_at": i - 1})
    return obs

def evaluate_signal_r25(m15):
    if len(m15) < 100: return None
    
    # Use all but the last (currently forming) candle
    m15_closed = m15.iloc[:-1].copy().reset_index(drop=True)
    idx = len(m15_closed) - 1
    if idx < 60: return None
    
    row = m15_closed.iloc[idx]
    cur_atr = row["atr"]
    bb_upper = row["bb_upper"]
    bb_lower = row["bb_lower"]
    bb_mid = row["bb_mid"]
    bb_width = row["bb_width"]
    bb_width_avg = row["bb_width_avg"]
    
    if np.isnan(cur_atr) or cur_atr == 0 or np.isnan(bb_upper) or np.isnan(bb_width_avg): return None
    
    o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
    rng = h - l
    if rng == 0: return None
    
    # 1. Squeeze Filter
    if bb_width > bb_width_avg * 1.2: return None
    
    # 2. Identify Bollinger Pierces & Rejections
    swept_lower = l < bb_lower and c > bb_lower and c > o
    swept_upper = h > bb_upper and c < bb_upper and c < o
    
    if not (swept_lower or swept_upper): return None
    
    # 3. Calculate Dynamic Scores
    if swept_lower:
        pierce_depth = (bb_lower - l) / cur_atr
        bb_score = min(1.0, pierce_depth / 0.5)
        lower_wick = min(o, c) - l
        pa_score = min(1.0, (lower_wick / rng) * 1.5)
    else:
        pierce_depth = (h - bb_upper) / cur_atr
        bb_score = min(1.0, pierce_depth / 0.5)
        upper_wick = h - max(o, c)
        pa_score = min(1.0, (upper_wick / rng) * 1.5)
        
    if bb_score < 0.2 or pa_score < 0.3: return None
    
    # 4. SNR Score (Confluence)
    snr_score = 0.4
    want_type = "bull" if swept_lower else "bear"
    
    for ob in row.get('obs', []):
        if ob["formed_at"] >= idx: continue
        if ob["type"] == want_type:
            if ob["low"] <= h and ob["high"] >= l:
                snr_score = 1.0; break
    if snr_score < 1.0:
        for f in row.get('fvgs', []):
            if f["formed_at"] >= idx: continue
            if f["type"] == want_type:
                if f["low"] <= h and f["high"] >= l:
                    snr_score = 0.8; break
                    
    total = bb_score + pa_score + snr_score
    
    # 5. Calculate Levels with STRICT R:R
    direction = 1 if swept_lower else -1
    entry = c
    
    if direction == 1:
        structural_stop = l - (cur_atr * 0.1)
        stop = max(structural_stop, entry - (cur_atr * 1.5)) 
        target = bb_mid
    else:
        structural_stop = h + (cur_atr * 0.1)
        stop = min(structural_stop, entry + (cur_atr * 1.5))
        target = bb_mid
        
    risk = abs(entry - stop)
    reward = abs(target - entry)
    
    if risk == 0 or reward < (risk * 1.5): return None 
    if reward > (cur_atr * 3.0): 
        target = entry + (cur_atr * 3.0) if direction == 1 else entry - (cur_atr * 3.0)
    
    # Determine Grade
    grade = "D"
    for g, thresh in sorted(GRADE_THRESHOLDS.items(), key=lambda x: x[1], reverse=True):
        if total >= thresh:
            grade = g
            break
    if grade == "D": return None
    
    return {
        "grade": grade, "direction": direction, "entry": entry, "stop": stop,
        "tp1": entry + (risk if direction == 1 else -risk), # 1R for partial
        "tp2": target, "risk_pct": (risk / entry) * 100,
        "rr_ratio": reward / risk, "score_total": total,
        "score_bb": bb_score, "score_pa": pa_score, "score_snr": snr_score, "atr": cur_atr,
    }

# ============================================================
# DATA FETCHER
# ============================================================
async def fetch(symbol, gran, total):
    all_c, seen = [], set()
    try:
        async with websockets.connect(ENDPOINT, ping_interval=None, ping_timeout=None) as ws:
            end, req, last = "latest", 0, None
            while len(all_c) < total:
                req += 1
                await ws.send(json.dumps({"ticks_history": symbol, "adjust_start_time": 1, "count": 2000,
                                          "end": end, "granularity": gran, "style": "candles", "req_id": req}))
                msg = json.loads(await asyncio.wait_for(ws.recv(), 30))
                if "error" in msg: return pd.DataFrame()
                c = msg.get("candles", [])
                if not c: break
                new = [x for x in c if int(x["epoch"]) not in seen]
                if not new: break
                for x in new: seen.add(int(x["epoch"]))
                all_c = new + all_c
                oldest = int(new[0]["epoch"])
                if last is not None and oldest >= last: break
                last, end = oldest, str(oldest - 1)
                await asyncio.sleep(0.05)
    except Exception as e:
        logging.error(f"Fetch error {symbol}: {e}")
        return pd.DataFrame()
    df = pd.DataFrame([{"epoch": int(x["epoch"]), "open": float(x["open"]), "high": float(x["high"]),
                        "low": float(x["low"]), "close": float(x["close"])} for x in all_c])
    return df.drop_duplicates("epoch").sort_values("epoch").reset_index(drop=True)

# ============================================================
# TELEGRAM & JOURNAL
# ============================================================
async def send_message(text, session):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status == 200: logging.info("📤 Telegram message sent.")
    except Exception as e:
        logging.error(f"Telegram error: {e}")

from datetime import datetime, timezone

# --- Triple Checked Small Caps Converter ---
SMALL_CAPS_MAP = str.maketrans(
    "abcdefghijklmnopqrstuvwxyz",
    "ᴀʙᴄᴅᴇꜰɢʜɪᴊᴋʟᴍɴᴏᴘqʀꜱᴛᴜᴠᴡxʏᴢ"
)

def to_small_caps(text: str) -> str:
    return text.lower().translate(SMALL_CAPS_MAP)

# --- Triple Checked Signal Function ---
async def send_signal(symbol, signal, session):
    global daily_counts
    grade = signal["grade"]
    if grade in daily_counts: daily_counts[grade] += 1

    direction_text = "🟢 ʟᴏɴɢ / ʙᴜʏ" if signal["direction"] == 1 else "🔴 ꜱʜᴏʀᴛ / ꜱᴇʟʟ"
    grade_emoji = {"A+": "🏆", "A": "⭐", "B": "✅", "C": "📊"}.get(grade, "📊")
    grade_label = {"A+": "ᴇʟɪᴛᴇ", "A": "ʜɪɢʜ", "B": "ꜱᴛᴀɴᴅᴀʀᴅ", "C": "ꜱᴘᴇᴄᴜʟᴀᴛɪᴠᴇ"}.get(grade, "")
    
    lbl_grade = to_small_caps("GRADE")
    lbl_entry = to_small_caps("ENTRY")
    lbl_sl = to_small_caps("STOP LOSS")
    lbl_tp1 = to_small_caps("TAKE PROFIT 1")
    lbl_tp2 = to_small_caps("TAKE PROFIT 2")
    lbl_rr = to_small_caps("RISK/REWARD")
    lbl_score = to_small_caps("CONFLUENCE SCORE")
    
    msg = (
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{grade_emoji} <b>{lbl_grade}: {grade} ({grade_label})</b>\n"
        f"📈 <b>Volatility 25 Index</b> <code>(R_25)</code>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"{direction_text}\n"
        f"📍 <b>{lbl_entry}:</b> <code>{signal['entry']:.5g}</code>\n"
        f"🛑 <b>{lbl_sl}:</b> <code>{signal['stop']:.5g}</code> <i>(Risk: {signal['risk_pct']:.2f}%)</i>\n\n"
        f"🎯 <b>{lbl_tp1}:</b> <code>{signal['tp1']:.5g}</code> <i>(50% Close)</i>\n"
        f"🎯 <b>{lbl_tp2}:</b> <code>{signal['tp2']:.5g}</code> <i>(Runner)</i>\n"
        f"⚖️ <b>{lbl_rr}:</b> 1 : {signal['rr_ratio']:.2f}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    )
    await send_message(msg, session)

# --- Triple Checked Daily Journal Function ---
async def send_daily_journal(session):
    global daily_counts, last_summary_date
    total_signals = sum(daily_counts.values())
    
    lbl_title = to_small_caps("R_25 Daily Journal")
    lbl_date = to_small_caps("Date")
    lbl_setups = to_small_caps("Total Setups")
    lbl_bot = to_small_caps("V3.2 Mean Reversion")
    
    msg = (
        f"📊 <b>{lbl_title}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📅 <b>{lbl_date}:</b> {last_summary_date.strftime('%Y-%m-%d')}\n"
        f"🎯 <b>{lbl_setups}:</b> {total_signals}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🏆 A+: {daily_counts['A+']} | ⭐ A: {daily_counts['A']}\n"
        f"✅ B: {daily_counts['B']} | 📊 C: {daily_counts['C']}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<i>{lbl_bot}</i>"
    )
    await send_message(msg, session)
    daily_counts = {"A+": 0, "A": 0, "B": 0, "C": 0}
    last_summary_date = datetime.now(timezone.utc).date()

# ============================================================
# MAIN LOOP
# ============================================================
async def run_trading_loop():
    logging.info("🚀 V3.2 R_25 Bot started. Checking every 15 minutes.")
    while True:
        try:
            now_utc = datetime.now(timezone.utc)
            current_date = now_utc.date()
            
            if now_utc.hour == 0 and now_utc.minute < 15 and current_date != last_summary_date:
                async with aiohttp.ClientSession() as session:
                    await send_daily_journal(session)

            async with aiohttp.ClientSession() as session:
                for symbol in SYMBOLS:
                    m15 = await fetch(symbol, M15_GRAN, 500)
                    if m15.empty: continue
                    
                    # Calculate indicators for the whole dataframe
                    m15 = bollinger_bands(m15, 20, 2.0)
                    m15["atr"] = atr(m15, ATR_PERIOD)
                    m15["bb_width_avg"] = m15["bb_width"].rolling(50).mean()
                    m15['fvgs'] = [detect_fvgs(m15)] * len(m15) # Simplified for live bot
                    m15['obs'] = [detect_obs(m15)] * len(m15)
                    
                    signal = evaluate_signal_r25(m15)
                    if signal:
                        await send_signal(symbol, signal, session)
                    else:
                        logging.info(f"✅ No signal for {symbol}")
        except Exception as e:
            logging.error(f"Trading loop error: {e}")
        
        logging.info("💤 Sleeping for 15 minutes...")
        await asyncio.sleep(900)

# ============================================================
# WEB SERVER (Keeps Render Awake)
# ============================================================
async def handle(request):
    return web.Response(text="R_25 Mean Reversion Bot is Alive 🚀")

async def start_background_tasks(app):
    app['trading_task'] = asyncio.ensure_future(run_trading_loop())

async def cleanup_background_tasks(app):
    app['trading_task'].cancel()
    await app['trading_task']

def create_app():
    app = web.Application()
    app.router.add_get('/', handle)
    app.on_startup.append(start_background_tasks)
    app.on_cleanup.append(cleanup_background_tasks)
    return app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app = create_app()
    web.run_app(app, host='0.0.0.0', port=port)
