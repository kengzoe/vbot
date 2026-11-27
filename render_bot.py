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
SYMBOLS = ["R_50", "R_100", "1HZ75V", "stpRNG5", "R_75", "stpRNG4", "1HZ25V", "1HZ30V"]

# Map API symbols to their official Deriv names
SYMBOL_NAMES = {
    "R_50": "Volatility 50 Index",
    "R_100": "Volatility 100 Index",
    "1HZ75V": "Volatility 75 (1s) Index",
    "stpRNG5": "Step Index 500",
    "R_10": "Volatility 10 Index",
    "R_25": "Volatility 25 Index",
    "R_75": "Volatility 75 Index",
    "stpRNG": "Step Index",
    "1HZ10V": "Volatility 10 (1s) Index",
    "1HZ25V": "Volatility 25 (1s) Index",
    "1HZ50V": "Volatility 50 (1s) Index",
    "1HZ100V": "Volatility 100 (1s) Index",
    "1HZ30V": "Volatility 30 (1s) Index"
}

# Custom parameters for each symbol based on backtest optimization
SYMBOL_PARAMS = {
    "R_50": {"target_cap": 3.0, "stop_buffer": 0.1},
    "R_100": {"target_cap": 3.0, "stop_buffer": 0.1},
    "1HZ75V": {"target_cap": 3.0, "stop_buffer": 0.1},
    "stpRNG5": {"target_cap": 3.0, "stop_buffer": 0.1},
    "R_75": {"target_cap": 4.0, "stop_buffer": 0.2}, # Optimized for R_75
}

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

# Track the last M15 candle epoch we sent a signal for, per symbol.
# Prevents re-alerting the SAME closed candle after the 15-minute loop wakes up.
last_signal_epoch = {}

# Track live/open signals so we can alert on TP1 / TP2 / SL hits.
# Structure: { symbol: {"signal": dict, "entry_epoch": int, "tp1_done": bool, "tp2_done": bool, "created_at": datetime} }
active_signals = {}

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

def evaluate_symbol(symbol, h1, m15, m5):
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
        # Get custom parameters for this specific symbol
    params = SYMBOL_PARAMS.get(symbol, {"target_cap": 3.0, "stop_buffer": 0.1})
    stop_buffer = params["stop_buffer"]
    target_cap = params["target_cap"]

    entry = float(row["close"])
    if direction == 1:
        stop = min(zl, float(row["low"])) - (cur_atr * stop_buffer)
    else:
        stop = max(zh, float(row["high"])) + (cur_atr * stop_buffer)
    
    risk = abs(entry - stop)
    raw_target_dist = abs(imp_e - entry)
    max_target_dist = cur_atr * target_cap
    effective_target_dist = min(raw_target_dist, max_target_dist)
    if effective_target_dist < risk: return None
    if direction == 1:
        target_1 = entry + risk
        target_final = entry + effective_target_dist
    else:
        target_1 = entry - risk
        target_final = entry - effective_target_dist
    return {
        "epoch": epoch,
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
            body = await resp.text()
            if resp.status == 200:
                logging.info("📤 Telegram message sent.")
            else:
                logging.error(f"❌ Telegram send failed: {resp.status} {body}")
    except Exception as e:
        logging.error(f"Telegram error: {e}", exc_info=True)
        
from datetime import datetime, timezone

# Ensure this is defined at the top of your file
SMALL_CAPS_MAP = str.maketrans(
    "abcdefghijklmnopqrstuvwxyz",
    "ᴀʙᴄᴅᴇꜰɢʜɪᴊᴋʟᴍɴᴏᴘqʀꜱᴛᴜᴠᴡxʏᴢ"
)

def to_small_caps(text: str) -> str:
    return text.lower().translate(SMALL_CAPS_MAP)

# --- Updated Signal Function (just for context) ---
async def send_signal(symbol, signal, session):
    global daily_counts
    grade = signal["grade"]
    if grade in daily_counts:
        daily_counts[grade] += 1

    real_name = SYMBOL_NAMES.get(symbol, symbol)
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
        f"📈 <b>{real_name}</b> <code>({symbol})</code>\n"
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


# --- Updated Daily Journal Function ---
async def send_daily_journal(session):
    global daily_counts
    
    # 1. Apply small caps to the main heading and labels
    heading = to_small_caps("Daily Market Journal")
    lbl_date = to_small_caps("Date")
    lbl_setups = to_small_caps("Total Setups Found")
    lbl_aplus = to_small_caps("A+ Signals")
    lbl_a = to_small_caps("A Signals")
    lbl_b = to_small_caps("B Signals")
    lbl_c = to_small_caps("C Signals")
    
    # Get current date
    today_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    
    # Calculate total setups
    total_setups = sum(daily_counts.values())

    # 2. Build the message
    msg = (
        f"📊 <b>{heading}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📅 <b>{lbl_date}:</b> {today_date}\n"
        f"🎯 <b>{lbl_setups}:</b> {total_setups}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"🏆 <b>{lbl_aplus}:</b> {daily_counts.get('A+', 0)}\n"
        f"⭐ <b>{lbl_a}:</b> {daily_counts.get('A', 0)}\n"
        f"✅ <b>{lbl_b}:</b> {daily_counts.get('B', 0)}\n"
        f"📊 <b>{lbl_c}:</b> {daily_counts.get('C', 0)}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    )

    # Send the message
    await send_message(msg, session)


# ============================================================
# OUTCOME ALERTS (TP1 / TP2 / SL HIT)
# ============================================================
async def send_outcome_message(symbol, sig, event, session):
    """Send a Telegram update when a previously-sent signal hits TP1 / TP2 / SL."""
    real_name = SYMBOL_NAMES.get(symbol, symbol)
    direction_text = "🟢 ʟᴏɴɢ / ʙᴜʏ" if sig["direction"] == 1 else "🔴 ꜱʜᴏʀᴛ / ꜱᴇʟʟ"
    entry = sig["entry"]

    if event == "tp1":
        title = "🎯 " + to_small_caps("TAKE PROFIT 1 HIT")
        price_line = f"✅ <b>{to_small_caps('TP1')}:</b> <code>{sig['tp1']:.5g}</code>"
        footer = "💡 " + to_small_caps("50% closed · move SL to breakeven")
    elif event == "tp2":
        title = "🏆 " + to_small_caps("TAKE PROFIT 2 HIT · TRADE CLOSED")
        price_line = f"🎯 <b>{to_small_caps('TP2')}:</b> <code>{sig['tp2']:.5g}</code> ✅"
        footer = "🏆 " + to_small_caps(f"full target reached · R/R 1 : {sig['rr_ratio']:.2f}")
    elif event == "sl_after_tp1":
        title = "⚖️ " + to_small_caps("STOPPED AFTER TP1")
        price_line = f"🛑 <b>{to_small_caps('SL')}:</b> <code>{sig['stop']:.5g}</code>"
        footer = "🛡️ " + to_small_caps("tp1 banked · remainder closed at BE")
    else:  # "sl"
        title = "🛑 " + to_small_caps("STOP LOSS HIT")
        price_line = f"❌ <b>{to_small_caps('SL')}:</b> <code>{sig['stop']:.5g}</code>"
        footer = "💼 " + to_small_caps(f"trade closed · risk {sig['risk_pct']:.2f}%")

    msg = (
        f"{title}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📈 <b>{real_name}</b> <code>({symbol})</code>\n"
        f"{direction_text} · <b>{to_small_caps('ENTRY')}:</b> <code>{entry:.5g}</code>\n"
        f"{price_line}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{footer}\n"
    )
    await send_message(msg, session)


async def check_signal_outcomes(session):
    """Scan M5 candles since each open signal's entry to detect TP1 / TP2 / SL hits."""
    global active_signals
    if not active_signals:
        return

    now_utc = datetime.now(timezone.utc)

    for symbol in list(active_signals.keys()):
        trade = active_signals[symbol]
        sig = trade["signal"]
        direction = sig["direction"]
        tp1, tp2, stop = sig["tp1"], sig["tp2"], sig["stop"]
        entry_epoch = trade["entry_epoch"]

        # Safety: expire trades older than 24h so a stuck symbol can't block new signals
        if (now_utc - trade["created_at"]).total_seconds() > 24 * 3600:
            logging.info(f"⌛ Expiring stale signal for {symbol} after 24h.")
            del active_signals[symbol]
            continue

        m5 = await fetch(symbol, M5_GRAN, 100)
        if m5.empty:
            continue

        # Only CLOSED M5 candles, and only those formed at/after entry time
        m5_closed = m5.iloc[:-1]
        recent = m5_closed[m5_closed["epoch"] >= entry_epoch].reset_index(drop=True)
        if recent.empty:
            continue

        finished = False
        for _, c in recent.iterrows():
            o, h, l, cl = float(c["open"]), float(c["high"]), float(c["low"]), float(c["close"])

            if direction == 1:  # long
                sl_hit  = l <= stop
                tp1_hit = h >= tp1
                tp2_hit = h >= tp2
            else:               # short
                sl_hit  = h >= stop
                tp1_hit = l <= tp1
                tp2_hit = l <= tp2

            # Heuristic for ambiguous candles (both TP and SL touched within same M5):
            # long  + bullish candle  → low formed first  → SL first
            # long  + bearish candle  → high formed first → TP first
            # short + bullish candle  → low formed first  → TP first
            # short + bearish candle  → high formed first → SL first
            tp_first = (direction == 1 and cl < o) or (direction == -1 and cl > o)

            if tp_first:
                events = []
                if tp1_hit: events.append("tp1")
                if tp2_hit: events.append("tp2")
                if sl_hit:  events.append("sl")
            else:
                events = []
                if sl_hit:  events.append("sl")
                if tp1_hit: events.append("tp1")
                if tp2_hit: events.append("tp2")

            for ev in events:
                if ev == "sl":
                    tag = "sl_after_tp1" if trade["tp1_done"] else "sl"
                    await send_outcome_message(symbol, sig, tag, session)
                    logging.info(f"📣 {symbol} SL hit → removed from active.")
                    finished = True
                    break
                elif ev == "tp1" and not trade["tp1_done"]:
                    trade["tp1_done"] = True
                    await send_outcome_message(symbol, sig, "tp1", session)
                    logging.info(f"📣 {symbol} TP1 hit.")
                elif ev == "tp2" and not trade["tp2_done"]:
                    trade["tp2_done"] = True
                    await send_outcome_message(symbol, sig, "tp2", session)
                    logging.info(f"📣 {symbol} TP2 hit → removed from active.")
                    finished = True
                    break

            if finished:
                break

        if finished:
            del active_signals[symbol]


# ============================================================
# BACKGROUND TRADING TASK
# ============================================================
async def run_trading_loop():
    global last_summary_date, active_signals, last_signal_epoch
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
                last_summary_date = current_date

            async with aiohttp.ClientSession() as session:
                # 1) First check TP/SL outcomes on any currently open signals
                await check_signal_outcomes(session)

                # 2) Then scan for fresh signals
                for symbol in SYMBOLS:
                    h1 = await fetch(symbol, H1_GRAN, 500)
                    m15 = await fetch(symbol, M15_GRAN, 500)
                    m5 = await fetch(symbol, M5_GRAN, 500)
                    if h1.empty or m15.empty or m5.empty: continue
                    signal = evaluate_symbol(symbol, h1, m15, m5)
                    if signal:
                        # Deduplicate: skip if this is the same M15 candle we already alerted on
                        if last_signal_epoch.get(symbol) == signal["epoch"]:
                            logging.info(
                                f"⏭️ Duplicate signal for {symbol} on candle "
                                f"{signal['epoch']} — already sent, skipping."
                            )
                        elif symbol in active_signals:
                            logging.info(f"⏳ {symbol} already has an open trade — skipping new signal.")
                        else:
                            await send_signal(symbol, signal, session)
                            last_signal_epoch[symbol] = signal["epoch"]
                            active_signals[symbol] = {
                                "signal": signal,
                                # M15 candle with epoch E closes at E+900; that's when the trade "starts"
                                "entry_epoch": signal["epoch"] + M15_GRAN,
                                "tp1_done": False,
                                "tp2_done": False,
                                "created_at": now_utc,
                            }
                            logging.info(f"🆕 Active trade opened for {symbol} (candle {signal['epoch']}).")
                    else:
                        logging.info(f"✅ No signal for {symbol}")
        except Exception as e:
            logging.error(f"Trading loop error: {e}", exc_info=True)
        
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
