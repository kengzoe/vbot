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
H1_GRAN, M15_GRAN, M5_GRAN = 3600, 900, 300
SYMBOLS = ["R_50", "R_100", "1HZ75V", "stpRNG5", "R_75", "stpRNG4", "1HZ25V"]

COST_PCT = 0.03
SWING_WINDOW = 5
FIB_LOW, FIB_HIGH = 0.50, 0.786
FIB_SHALLOW = 0.382
ATR_PERIOD = 14
GRADE_THRESHOLDS = {"A+": 2.4, "A": 2.1, "B": 1.8, "C": 1.5}

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# ============================================================
# DAILY JOURNAL TRACKER
# ============================================================
daily_counts = {"A+": 0, "A": 0, "B": 0, "C": 0}
# FIX: Strictly use UTC date to prevent timezone mismatch bugs
last_summary_date = datetime.now(timezone.utc).date() 

# ============================================================
# STRATEGY HELPER FUNCTIONS (V2.2 Logic)
# ============================================================
def atr(df, p):
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"]-df["low"], (df["high"]-pc).abs(), (df["low"]-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/p, adjust=False).mean()

def detect_swings(df, window=SWING_WINDOW):
    if len(df) < window * 2 + 1: return []
    highs, lows = df["high"].values, df["low"].values
    out = []
    for i in range(window, len(df) - window):
        if highs[i] == max(highs[i-window:i+window+1]) and highs[i] > highs[i-1]:
            out.append({"idx": i, "type": 1, "price": float(highs[i]), "confirmed_at": i + window})
        if lows[i] == min(lows[i-window:i+window+1]) and lows[i] < lows[i-1]:
            out.append({"idx": i, "type": -1, "price": float(lows[i]), "confirmed_at": i + window})
    return out

def compute_h1_bias(h1, swings):
    n = len(h1); bias = np.zeros(n, dtype=int)
    ims = np.full(n, np.nan); ime = np.full(n, np.nan)
    ch, cl = [], []
    cur = 0; bull = (np.nan, np.nan); bear = (np.nan, np.nan); si = 0
    for i in range(n):
        while si < len(swings) and swings[si]["confirmed_at"] <= i:
            s = swings[si]; (ch if s["type"] == 1 else cl).append(s["price"]); si += 1
        close = h1["close"].iloc[i]
        if ch and close > ch[-1]:
            cur = 1
            lows = [p for p in cl if p < ch[-1]]
            if lows: bull = (lows[-1], ch[-1])
        elif cl and close < cl[-1]:
            cur = -1
            highs = [p for p in ch if p > cl[-1]]
            if highs: bear = (highs[-1], cl[-1])
        bias[i] = cur
        if cur == 1: ims[i], ime[i] = bull
        elif cur == -1: ims[i], ime[i] = bear
    return pd.DataFrame({"epoch": h1["epoch"].values, "bias": bias, "imp_s": ims, "imp_e": ime})

def detect_obs(df, swings):
    obs = []
    closes = df["close"].values
    for s in swings:
        bp = s["idx"]
        if bp < 2: continue
        prev = df.iloc[bp - 1]
        if s["type"] == 1 and prev["close"] < prev["open"]:
            zl, zh = float(min(prev["open"], prev["close"])), float(max(prev["open"], prev["close"]))
            obs.append({"type": "bull", "low": zl, "high": zh, "formed_at": bp - 1})
        elif s["type"] == -1 and prev["close"] > prev["open"]:
            zl, zh = float(min(prev["open"], prev["close"])), float(max(prev["open"], prev["close"]))
            obs.append({"type": "bear", "low": zl, "high": zh, "formed_at": bp - 1})
    return obs

def detect_fvgs(df):
    fvgs = []
    highs, lows = df["high"].values, df["low"].values
    for i in range(2, len(df)):
        if lows[i] > highs[i-2]:
            fvgs.append({"type": "bull", "low": float(highs[i-2]), "high": float(lows[i]), "formed_at": i})
        elif highs[i] < lows[i-2]:
            fvgs.append({"type": "bear", "low": float(highs[i]), "high": float(lows[i-2]), "formed_at": i})
    return fvgs

def classify_pa(o, h, l, c, po, pc, ph, pl):
    body = abs(c - o); rng = h - l
    if rng == 0: return 0.0
    br = body / rng
    is_bull = c > o; is_bear = c < o
    prev_bull = pc > po; prev_bear = pc < po
    if is_bull and prev_bear and c > po and o < pc:
        return 1.0 if br >= 0.6 else 0.7
    if is_bear and prev_bull and c < po and o > pc:
        return 1.0 if br >= 0.6 else 0.7
    uw = h - max(o, c); lw = min(o, c) - l
    if lw > 2*body and lw > rng*0.5: return 0.9
    if uw > 2*body and uw > rng*0.5: return 0.9
    if h <= ph and l >= pl: return 0.6
    if br < 0.15: return 0.3
    return 0.0

def fib_zone_score(close, imp_s, imp_e, direction):
    if direction == 1:
        lo, hi = imp_s, imp_e; rng = hi - lo
        ret = (hi - close) / rng if rng > 0 else 0
        zl = hi - rng * FIB_HIGH; zh = hi - rng * FIB_LOW
    else:
        hi, lo = imp_s, imp_e; rng = hi - lo
        ret = (close - lo) / rng if rng > 0 else 0
        zl = lo + rng * FIB_LOW; zh = lo + rng * FIB_HIGH
    if FIB_LOW <= ret <= FIB_HIGH: return 1.0, zl, zh
    if FIB_SHALLOW <= ret < FIB_LOW: return 0.75, zl, zh
    return 0.0, zl, zh

def check_structure(swings, direction, idx):
    confirmed = [s for s in swings if s["confirmed_at"] <= idx]
    highs = sorted([s for s in confirmed if s["type"] == 1], key=lambda s: s["confirmed_at"])
    lows = sorted([s for s in confirmed if s["type"] == -1], key=lambda s: s["confirmed_at"])
    if len(highs) < 2 or len(lows) < 2: return False
    rh, ph = highs[-1]["price"], highs[-2]["price"]
    rl, pl = lows[-1]["price"], lows[-2]["price"]
    if direction == 1: return rh > ph and rl > pl
    return rh < ph and rl < pl

def compute_mtf(idx, direction, m15_swings, m5_swings):
    hits = sum([check_structure(m15_swings, direction, idx),
                check_structure(m5_swings, direction, idx)])
    return 1.0 if hits == 2 else 0.5 if hits == 1 else 0.0

def compute_snr(idx, low, high, direction, obs, fvgs, m15_swings, atr_val):
    mid = (low + high) / 2
    best = 0.0
    want = "bull" if direction == 1 else "bear"
    for s in m15_swings:
        if s["type"] != direction or s["confirmed_at"] > idx: continue
        if abs(s["price"] - mid) < atr_val * 0.3:
            if 1.0 > best: best = 1.0
    for ob in obs:
        if ob["type"] != want or ob["formed_at"] >= idx: continue
        if not (ob["high"] < low or ob["low"] > high):
            if 0.9 > best: best = 0.9
    for f in fvgs:
        if f["type"] != want or f["formed_at"] >= idx: continue
        if not (f["high"] < low or f["low"] > high):
            if 0.8 > best: best = 0.8
    if best == 0.0:
        for s in m15_swings:
            if s["type"] != direction or s["confirmed_at"] > idx: continue
            if abs(s["price"] - mid) < atr_val * 1.0:
                best = 0.5; break
    return best

def get_grade(total_score):
    if total_score >= GRADE_THRESHOLDS["A+"]: return "A+"
    elif total_score >= GRADE_THRESHOLDS["A"]: return "A"
    elif total_score >= GRADE_THRESHOLDS["B"]: return "B"
    elif total_score >= GRADE_THRESHOLDS["C"]: return "C"
    else: return "D"

def evaluate_signal(h1, m15, m5):
    if len(h1) < 200 or len(m15) < 100 or len(m5) < 200: return None
    m15_closed = m15.iloc[:-1].copy().reset_index(drop=True)
    m5_closed = m5.iloc[:-1].copy().reset_index(drop=True)
    idx = len(m15_closed) - 1
    if idx < max(60, SWING_WINDOW + 5): return None
    row = m15_closed.iloc[idx]
    prev = m15_closed.iloc[idx - 1]
    h1_sw = detect_swings(h1)
    h1_bias = compute_h1_bias(h1, h1_sw)
    m15_sw = detect_swings(m15_closed)
    m5_sw = detect_swings(m5_closed)
    obs = detect_obs(m15_closed, m15_sw)
    fvgs = detect_fvgs(m15_closed)
    m15_atr = atr(m15_closed, ATR_PERIOD)
    epoch = int(row["epoch"])
    hep = h1_bias["epoch"].values
    hi = np.searchsorted(hep, epoch, side="right") - 1
    if hi < 0: return None
    h1r = h1_bias.iloc[hi]
    if h1r["bias"] == 0 or np.isnan(h1r["imp_s"]) or np.isnan(h1r["imp_e"]): return None
    direction = int(h1r["bias"])
    imp_s, imp_e = float(h1r["imp_s"]), float(h1r["imp_e"])
    fib_score, zl, zh = fib_zone_score(float(row["close"]), imp_s, imp_e, direction)
    pa_score = classify_pa(float(row["open"]), float(row["high"]), float(row["low"]),
                            float(row["close"]), float(prev["open"]), float(prev["close"]),
                            float(prev["high"]), float(prev["low"]))
    if pa_score < 0.6: return None
    mtf_score = compute_mtf(idx, direction, m15_sw, m5_sw)
    cur_atr = float(m15_atr.iloc[idx])
    if np.isnan(cur_atr) or cur_atr == 0: return None
    snr_score = compute_snr(idx, zl, zh, direction, obs, fvgs, m15_sw, cur_atr)
    if snr_score < 0.5: return None
    total = mtf_score + pa_score + snr_score
    grade = get_grade(total)
    if grade == "D": return None
    entry = float(row["close"])
    if direction == 1:
        stop = min(zl, float(row["low"])) - cur_atr * 0.1
    else:
        stop = max(zh, float(row["high"])) + cur_atr * 0.1
    risk = abs(entry - stop)
    raw_target_dist = abs(imp_e - entry)
    max_target_dist = cur_atr * 3.0
    effective_target_dist = min(raw_target_dist, max_target_dist)
    if effective_target_dist < risk: return None
    if direction == 1:
        target_1 = entry + risk
        target_final = entry + effective_target_dist
    else:
        target_1 = entry - risk
        target_final = entry - effective_target_dist
    return {
        "grade": grade, "direction": direction, "entry": entry, "stop": stop,
        "tp1": target_1, "tp2": target_final, "risk_pct": (risk / entry) * 100,
        "rr_ratio": effective_target_dist / risk, "score_total": total,
        "score_mtf": mtf_score, "score_pa": pa_score, "score_snr": snr_score, "atr": cur_atr,
    }

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

async def send_message(text, session):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status == 200: logging.info("📤 Telegram message sent.")
    except Exception as e:
        logging.error(f"Telegram error: {e}")

async def send_signal(symbol, signal, session):
    global daily_counts
    grade = signal["grade"]
    if grade in daily_counts:
        daily_counts[grade] += 1

    direction = "🟢 BUY" if signal["direction"] == 1 else "🔴 SELL"
    grade_emoji = {"A+": "🏆", "A": "⭐", "B": "✅", "C": "📊"}.get(grade, "📊")
    msg = (
        f"{grade_emoji} <b>{grade} SIGNAL DETECTED</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Symbol:</b> <code>{symbol}</code>\n"
        f"<b>Direction:</b> {direction}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>📍 Entry:</b> <code>{signal['entry']:.5g}</code>\n"
        f"<b>🛑 Stop Loss:</b> <code>{signal['stop']:.5g}</code>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>🎯 TP1:</b> <code>{signal['tp1']:.5g}</code>\n"
        f"<b>🎯 TP2:</b> <code>{signal['tp2']:.5g}</code>\n"
        f"<i>V2.2 Render Bot | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</i>"
    )
    await send_message(msg, session)

async def send_daily_journal(session):
    global daily_counts, last_summary_date
    
    total_signals = sum(daily_counts.values())
    today_str = last_summary_date.strftime("%Y-%m-%d")
    
    msg = (
        f"📊 <b>Daily Market Journal</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📅 <b>Date:</b> {today_str}\n"
        f"🎯 <b>Total Setups Found:</b> {total_signals}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🏆 <b>A+ Signals:</b> {daily_counts['A+']}\n"
        f"⭐ <b>A Signals:</b> {daily_counts['A']}\n"
        f"✅ <b>B Signals:</b> {daily_counts['B']}\n"
        f"📊 <b>C Signals:</b> {daily_counts['C']}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<i>Review your Telegram history to check trade outcomes.</i>\n"
        f"<i>Bot is alive and scanning...</i>"
    )
    await send_message(msg, session)
    
    # FIX: Reset using strict UTC date
    daily_counts = {"A+": 0, "A": 0, "B": 0, "C": 0}
    last_summary_date = datetime.now(timezone.utc).date()

# ============================================================
# BACKGROUND TRADING TASK
# ============================================================
async def run_trading_loop():
    logging.info("🚀 Trading loop started. Checking every 15 minutes.")
    while True:
        try:
            now_utc = datetime.now(timezone.utc)
            current_date = now_utc.date()
            
            # Check if it's a new day (runs between 00:00 and 00:14 UTC)
            if now_utc.hour == 0 and now_utc.minute < 15 and current_date != last_summary_date:
                logging.info("🕛 Midnight UTC reached. Sending Daily Journal...")
                async with aiohttp.ClientSession() as session:
                    await send_daily_journal(session)

            async with aiohttp.ClientSession() as session:
                for symbol in SYMBOLS:
                    h1 = await fetch(symbol, H1_GRAN, 500)
                    m15 = await fetch(symbol, M15_GRAN, 500)
                    m5 = await fetch(symbol, M5_GRAN, 500)
                    if h1.empty or m15.empty or m5.empty: continue
                    signal = evaluate_signal(h1, m15, m5)
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
    return web.Response(text="Deriv Bot is Alive and Running 🚀")

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
