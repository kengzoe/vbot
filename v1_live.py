import asyncio, json, logging, time
import numpy as np
import pandas as pd
import websockets
import aiohttp

# ============================================================
# CONFIGURATION - FILL THESE IN
# ============================================================
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"
TELEGRAM_CHAT_ID   = "YOUR_CHAT_ID_HERE"

ENDPOINT = "wss://api.derivws.com/trading/v1/options/ws/public?app_id=1089"
H1_GRAN, M15_GRAN, M5_GRAN = 3600, 900, 300

SYMBOLS = ["R_50", "R_100", "1HZ75V", "stpRNG5"]

COST_PCT = 0.03
SWING_WINDOW = 5
FIB_LOW, FIB_HIGH = 0.50, 0.786
FIB_SHALLOW = 0.382
ATR_PERIOD = 14
MAX_BARS_OPEN = 40

GRADE_THRESHOLDS = {
    "A+": 2.4,
    "A":  2.1,
    "B":  1.8,
    "C":  1.5
}

GRADE_COOLDOWNS = {
    "A+": MAX_BARS_OPEN * M15_GRAN,
    "A":  int(MAX_BARS_OPEN * M15_GRAN * 0.75),
    "B":  int(MAX_BARS_OPEN * M15_GRAN * 0.5),
    "C":  int(MAX_BARS_OPEN * M15_GRAN * 0.25)
}

H1_BUFFER, M15_BUFFER, M5_BUFFER = 2000, 4000, 10000

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("live_bot")


class TelegramSender:
    def __init__(self, token, chat_id):
        self.token = token
        self.chat_id = chat_id
        self.url = f"https://api.telegram.org/bot{token}/sendMessage"
    
    async def send(self, text, session):
        try:
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True
            }
            async with session.post(self.url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    log.warning(f"Telegram send failed: {resp.status}")
        except Exception as e:
            log.warning(f"Telegram error: {e}")


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
    if total_score >= GRADE_THRESHOLDS["A+"]:
        return "A+"
    elif total_score >= GRADE_THRESHOLDS["A"]:
        return "A"
    elif total_score >= GRADE_THRESHOLDS["B"]:
        return "B"
    elif total_score >= GRADE_THRESHOLDS["C"]:
        return "C"
    else:
        return "D"


def evaluate_signal(h1, m15, m5):
    if len(h1) < 200 or len(m15) < 100 or len(m5) < 200:
        return None
    
    m15_closed = m15.iloc[:-1].copy().reset_index(drop=True)
    m5_closed = m5.iloc[:-1].copy().reset_index(drop=True)
    
    idx = len(m15_closed) - 1
    if idx < max(60, SWING_WINDOW + 5):
        return None
    
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
    if h1r["bias"] == 0 or np.isnan(h1r["imp_s"]) or np.isnan(h1r["imp_e"]):
        return None
    
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
    
    if grade == "D":
        return None
    
    entry = float(row["close"])
    if direction == 1:
        stop = min(zl, float(row["low"])) - cur_atr * 0.1
    else:
        stop = max(zh, float(row["high"])) + cur_atr * 0.1
    
    risk = abs(entry - stop)
    raw_target_dist = abs(imp_e - entry)
    max_target_dist = cur_atr * 3.0
    effective_target_dist = min(raw_target_dist, max_target_dist)
    
    if effective_target_dist < risk:
        return None
    
    if direction == 1:
        target_1 = entry + risk
        target_final = entry + effective_target_dist
    else:
        target_1 = entry - risk
        target_final = entry - effective_target_dist
    
    return {
        "grade": grade,
        "direction": direction,
        "entry": entry,
        "stop": stop,
        "tp1": target_1,
        "tp2": target_final,
        "risk_pct": (risk / entry) * 100,
        "rr_ratio": effective_target_dist / risk,
        "score_total": total,
        "score_mtf": mtf_score,
        "score_pa": pa_score,
        "score_snr": snr_score,
        "atr": cur_atr,
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
                msg = json.loads(await asyncio.wait_for(ws.recv(), 60))
                if "error" in msg:
                    return pd.DataFrame()
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
        log.error(f"Fetch error {symbol}: {e}")
        return pd.DataFrame()
    df = pd.DataFrame([{"epoch": int(x["epoch"]), "open": float(x["open"]), "high": float(x["high"]),
                        "low": float(x["low"]), "close": float(x["close"])} for x in all_c])
    return df.drop_duplicates("epoch").sort_values("epoch").reset_index(drop=True)


async def monitor_symbol(symbol, data_store, signal_tracker, tg, session):
    while True:
        try:
            # Disable automatic ping/pong to avoid connection issues
            async with websockets.connect(ENDPOINT, ping_interval=None, ping_timeout=None, close_timeout=10) as ws:
                # Subscribe to tick stream (not candles - we'll aggregate ourselves)
                await ws.send(json.dumps({
                    "ticks_history": symbol,
                    "adjust_start_time": 1,
                    "count": 1,
                    "end": "latest",
                    "granularity": M15_GRAN,
                    "style": "candles",
                    "subscribe": 1
                }))
                
                last_evaluated_epoch = None
                msg_count = 0
                
                while True:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=180)  # 3 minute timeout
                        msg_data = json.loads(msg)
                        msg_count += 1
                        
                        # Skip error messages
                        if "error" in msg_data:
                            log.warning(f"{symbol} stream error: {msg_data['error']}")
                            break
                        
                        # Skip subscription acknowledgment
                        if msg_data.get("msg_type") == "subscription":
                            continue
                        
                        # Extract OHLC data
                        ohlc = msg_data.get("ohlc")
                        if not ohlc:
                            continue
                        
                        epoch = int(ohlc.get("epoch", 0))
                        if epoch == 0:
                            continue
                        
                        o = float(ohlc["open"])
                        h = float(ohlc["high"])
                        l = float(ohlc["low"])
                        c = float(ohlc["close"])
                        
                        df = data_store[symbol]["m15"]
                        
                        # Check if this is a new candle
                        if len(df) > 0 and df.iloc[-1]["epoch"] < epoch:
                            closed_epoch = df.iloc[-1]["epoch"]
                            
                            # Evaluate if we haven't already
                            if last_evaluated_epoch != closed_epoch:
                                last_evaluated_epoch = closed_epoch
                                
                                now = time.time()
                                should_evaluate = False
                                
                                if symbol not in signal_tracker:
                                    should_evaluate = True
                                else:
                                    last_signal_time, last_grade = signal_tracker[symbol]
                                    cooldown = GRADE_COOLDOWNS.get(last_grade, GRADE_COOLDOWNS["C"])
                                    if now - last_signal_time >= cooldown:
                                        should_evaluate = True
                                
                                if should_evaluate:
                                    signal = evaluate_signal(
                                        data_store[symbol]["h1"],
                                        data_store[symbol]["m15"],
                                        data_store[symbol]["m5"]
                                    )
                                    if signal:
                                        signal_tracker[symbol] = (now, signal["grade"])
                                        await send_signal(symbol, signal, tg, session)
                            
                            # Add new candle
                            new_row = pd.DataFrame([{"epoch": epoch, "open": o, "high": h, "low": l, "close": c}])
                            data_store[symbol]["m15"] = pd.concat([df, new_row], ignore_index=True).tail(M15_BUFFER)
                        
                        elif len(df) > 0 and df.iloc[-1]["epoch"] == epoch:
                            # Update current candle
                            df.at[df.index[-1], "high"] = max(df.iloc[-1]["high"], h)
                            df.at[df.index[-1], "low"] = min(df.iloc[-1]["low"], l)
                            df.at[df.index[-1], "close"] = c
                    
                    except asyncio.TimeoutError:
                        # No message received in 3 minutes - send ping to keep alive
                        try:
                            pong = await ws.ping()
                            await asyncio.wait_for(pong, timeout=5)
                        except:
                            log.warning(f"{symbol} ping failed, reconnecting...")
                            break
        
        except (websockets.exceptions.ConnectionClosed, websockets.exceptions.ConnectionClosedError) as e:
            log.warning(f"{symbol} connection closed: {e}. Reconnecting in 5s...")
            await asyncio.sleep(5)
        except Exception as e:
            log.exception(f"{symbol} monitor crashed: {e}. Restarting in 10s...")
            await asyncio.sleep(10)


async def send_signal(symbol, signal, tg, session):
    direction = "🟢 BUY" if signal["direction"] == 1 else "🔴 SELL"
    grade = signal["grade"]
    
    grade_emoji = {
        "A+": "🏆",
        "A":  "⭐",
        "B":  "✅",
        "C":  "📊"
    }.get(grade, "📊")
    
    msg = (
        f"{grade_emoji} <b>{grade} SIGNAL DETECTED</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Symbol:</b> <code>{symbol}</code>\n"
        f"<b>Direction:</b> {direction}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>📍 Entry:</b> <code>{signal['entry']:.5g}</code>\n"
        f"<b>🛑 Stop Loss:</b> <code>{signal['stop']:.5g}</code>\n"
        f"   <i>(Risk: {signal['risk_pct']:.2f}%)</i>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>🎯 TP1 (Partial 50%):</b> <code>{signal['tp1']:.5g}</code>\n"
        f"<b>🎯 TP2 (Final):</b> <code>{signal['tp2']:.5g}</code>\n"
        f"   <i>(R:R = 1:{signal['rr_ratio']:.2f})</i>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>📊 Trade Management:</b>\n"
        f"• At TP1: close 50%, move SL to breakeven\n"
        f"• Trail remaining 50% by 1.5× ATR\n"
        f"• ATR: <code>{signal['atr']:.5g}</code>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>🧠 Score Breakdown:</b>\n"
        f"• MTF: {signal['score_mtf']:.2f}\n"
        f"• PA:  {signal['score_pa']:.2f}\n"
        f"• SNR: {signal['score_snr']:.2f}\n"
        f"• <b>Total: {signal['score_total']:.2f} ({grade})</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<i>V2.2 Strategy | {time.strftime('%Y-%m-%d %H:%M:%S')}</i>"
    )
    
    await tg.send(msg, session)
    log.info(f"📤 {grade} signal sent for {symbol} | {direction} @ {signal['entry']:.5g} | Score: {signal['score_total']:.2f}")


async def main():
    log.info("=" * 60)
    log.info("🚀 V2.2 LIVE SIGNAL BOT STARTING")
    log.info(f"📡 Monitoring {len(SYMBOLS)} symbols: {', '.join(SYMBOLS)}")
    log.info(f"📊 Sending signals for grades: A+, A, B, C")
    log.info("=" * 60)
    
    if TELEGRAM_BOT_TOKEN == "YOUR_BOT_TOKEN_HERE" or TELEGRAM_CHAT_ID == "YOUR_CHAT_ID_HERE":
        log.error("❌ Please set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in the script!")
        return
    
    tg = TelegramSender(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    data_store = {s: {"h1": None, "m15": None, "m5": None} for s in SYMBOLS}
    signal_tracker = {}
    
    async with aiohttp.ClientSession() as session:
        await tg.send(
            f"🟢 <b>Live Bot Started</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Monitoring: {', '.join(SYMBOLS)}\n"
            f"Strategy: V2.2 No-Fib\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>Signal Grades:</b>\n"
            f"🏆 A+ (Score ≥ 2.4) — Cooldown: 10h\n"
            f"⭐ A  (Score ≥ 2.1) — Cooldown: 7.5h\n"
            f"✅ B  (Score ≥ 1.8) — Cooldown: 5h\n"
            f"📊 C  (Score ≥ 1.5) — Cooldown: 2.5h\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"<i>Waiting for M15 candle closes...</i>",
            session
        )
        log.info("✅ Startup Telegram message sent")
        
        for symbol in SYMBOLS:
            log.info(f"📥 Fetching historical data for {symbol}...")
            h1 = await fetch(symbol, H1_GRAN, H1_BUFFER)
            m15 = await fetch(symbol, M15_GRAN, M15_BUFFER)
            m5 = await fetch(symbol, M5_GRAN, M5_BUFFER)
            
            if h1.empty or m15.empty or m5.empty:
                log.error(f"❌ Failed to fetch data for {symbol}. Skipping.")
                continue
            
            data_store[symbol] = {"h1": h1, "m15": m15, "m5": m5}
            log.info(f"  ✅ {symbol}: h1={len(h1)} m15={len(m15)} m5={len(m5)}")
        
        tasks = []
        for symbol in SYMBOLS:
            if data_store[symbol]["m15"] is not None and not data_store[symbol]["m15"].empty:
                tasks.append(asyncio.create_task(
                    monitor_symbol(symbol, data_store, signal_tracker, tg, session)
                ))
        
        if not tasks:
            log.error("❌ No symbols could be monitored. Exiting.")
            return
        
        log.info(f"🎯 Live monitoring started for {len(tasks)} symbols")
        log.info("⏰ Signals will be sent on M15 candle closes")
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            log.info("🛑 Shutting down gracefully...")
            await tg.send("🔴 <b>Live Bot Stopped</b>\n<i>Manual shutdown</i>", session)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("🛑 Bot terminated by user")
