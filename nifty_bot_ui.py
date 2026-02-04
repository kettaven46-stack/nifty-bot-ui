import os
import sys
import json
import time
import requests
import pandas as pd
import numpy as np
from urllib.parse import quote
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# ==========================================================
# CONFIG
# ==========================================================
# Local default (Streamlit Cloud will override via Secrets)
ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiI2UEE2QzUiLCJqdGkiOiI2OTgyYTQ1YjFmNWJkMTYyNzRhMDQyMTciLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNQbHVzUGxhbiI6dHJ1ZSwiaWF0IjoxNzcwMTY5NDM1LCJpc3MiOiJ1ZGFwaS1nYXRld2F5LXNlcnZpY2UiLCJleHAiOjE3NzAyNDI0MDB9.FE02xu-nqq8xWsPaGNTCn59TmbzFWtE7Cj_m1xHplOE"

UNDERLYING_KEY = "NSE_INDEX|Nifty 50"
INTERVAL = "1minute"
UPDATE_EVERY_SECONDS = 60

# Local persistence (on Streamlit Cloud this may not persist long-term)
SAVE_PATH = os.environ.get("BOT_SAVE_PATH", "bot_data")
STATE_FILE = os.path.join(SAVE_PATH, "bot_state.json")
TRADES_FILE = os.path.join(SAVE_PATH, "bot_trades.csv")
LOG_FILE = os.path.join(SAVE_PATH, "bot_log.txt")

# NSE hours (IST)
MARKET_OPEN = "09:15"
MARKET_CLOSE = "15:30"
MAX_LOOKBACK_DAYS = 15

DEFAULT_LIVE_TRADING = False  # safety OFF

# ==========================================================
# DEFAULT STRATEGY RULES (editable in UI)
# ==========================================================
DEFAULT_RULES = {
    "name": "NIFTY + ATM confirmation",
    "live_trading": DEFAULT_LIVE_TRADING,

    "min_adx": 18,
    "rsi_buy": 55,
    "rsi_sell": 45,

    # "cross" => macd cross on last candle, "above" => macd above signal
    "macd_mode": "cross",

    "confirm_with_options": True,
    "ce_confirm_rsi": 52,
    "pe_confirm_rsi": 52,

    "one_trade_at_a_time": True,
    "cooldown_minutes": 5,

    "paper_mode": True,
}

# ==========================================================
# FILE / STATE (NO RECURSION)
# ==========================================================
def ensure_paths():
    os.makedirs(SAVE_PATH, exist_ok=True)

    if not os.path.exists(TRADES_FILE):
        pd.DataFrame(columns=["ts_ist", "signal", "symbol", "price", "note"]).to_csv(TRADES_FILE, index=False)

    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write("")

    if not os.path.exists(STATE_FILE):
        init_state = {
            "status": "INIT",
            "last_update_ist": "",
            "market_state": "",
            "today_ref": "",
            "expiry": "",
            "spot": None,
            "atm_strike": None,
            "index_mode": "",
            "ce_mode": "",
            "pe_mode": "",
            "last_signal": "NONE",
            "last_signal_ts_ist": "",
            "last_index_candle": "",
            "position": "FLAT",  # FLAT / LONG_CE / LONG_PE
            "last_trade_ts_ist": "",
            "rules": DEFAULT_RULES,
            "errors": "",
            "last_reason": "",
            "last_exec_status": "",
            "last_exec_note": ""
        }
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(init_state, f, indent=2, ensure_ascii=False)


def read_state():
    ensure_paths()
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def write_state(state: dict):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def append_trade(row: dict):
    ensure_paths()
    df = pd.read_csv(TRADES_FILE)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(TRADES_FILE, index=False)


def log_line(msg: str):
    ensure_paths()
    ts = ist_now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# ==========================================================
# TIME HELPERS
# ==========================================================
def ist_now():
    return datetime.now(ZoneInfo("Asia/Kolkata"))

def ist_today():
    return ist_now().date()

def date_str(d):
    return d.strftime("%Y-%m-%d")

def market_state():
    now = ist_now().time()
    open_t = datetime.strptime(MARKET_OPEN, "%H:%M").time()
    close_t = datetime.strptime(MARKET_CLOSE, "%H:%M").time()
    return "OPEN" if (open_t <= now <= close_t) else "CLOSED"

# ==========================================================
# API HELPERS
# ==========================================================
def api_get(url, params=None):
    if not ACCESS_TOKEN or "PASTE_YOUR" in ACCESS_TOKEN:
        raise RuntimeError("ACCESS_TOKEN missing. Set Streamlit Secrets ACCESS_TOKEN or paste it in code (not recommended).")

    headers = {"Accept": "application/json", "Authorization": f"Bearer {ACCESS_TOKEN}"}
    r = requests.get(url, params=params, headers=headers, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
    return r.json()


def force_numeric_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    for c in ["open", "high", "low", "close", "volume", "oi"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "volume" in df.columns:
        df["volume"] = df["volume"].fillna(0)
    if "oi" in df.columns:
        df["oi"] = df["oi"].fillna(0)
    df = df.dropna(subset=["time", "high", "low", "close"])
    return df


def fetch_candles_day(instrument_key: str, day) -> pd.DataFrame:
    """
    Fetch candles for a given day.
    - If day == today (IST): use intraday endpoint (live running candles)
    - Else: use historical day endpoint
    """
    encoded = quote(instrument_key, safe="")

    # TODAY (intraday live)
    if day == ist_today():
        url = f"https://api.upstox.com/v2/historical-candle/intraday/{encoded}/{INTERVAL}"
        j = api_get(url)
        candles = j.get("data", {}).get("candles", [])
        if not candles:
            return pd.DataFrame()

        df = pd.DataFrame(candles, columns=["time", "open", "high", "low", "close", "volume", "oi"])

    # PAST DAY (historical)
    else:
        d = date_str(day)
        url = f"https://api.upstox.com/v2/historical-candle/{encoded}/{INTERVAL}/{d}/{d}"
        j = api_get(url)
        candles = j.get("data", {}).get("candles", [])
        if not candles:
            return pd.DataFrame()

        df = pd.DataFrame(candles, columns=["time", "open", "high", "low", "close", "volume", "oi"])

    # Convert time -> IST and timezone-naive (Excel/Streamlit friendly)
    df["time"] = (
        pd.to_datetime(df["time"], utc=True, errors="coerce")
        .dt.tz_convert("Asia/Kolkata")
        .dt.tz_localize(None)
    )

    df = force_numeric_ohlcv(df)
    df = df.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)
    return df

# ==========================================================
# INDICATORS
# ==========================================================
def ema(series: pd.Series, span: int):
    return series.ewm(span=span, adjust=False).mean()

def macd(close: pd.Series, fast=12, slow=26, signal=9):
    close = close.astype(float)
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def rsi(close: pd.Series, length=14):
    close = close.astype(float)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def dmi_adx(high: pd.Series, low: pd.Series, close: pd.Series, length=14):
    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)

    up = high.diff()
    down = -low.diff()

    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    prev_close = close.shift(1)
    tr1 = (high - low).abs().to_numpy()
    tr2 = (high - prev_close).abs().to_numpy()
    tr3 = (low - prev_close).abs().to_numpy()
    tr = np.nanmax(np.vstack([tr1, tr2, tr3]), axis=0)
    tr = pd.Series(tr, index=high.index)

    atr = tr.ewm(alpha=1/length, adjust=False).mean()
    plus_dm_s = pd.Series(plus_dm, index=high.index).ewm(alpha=1/length, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm, index=high.index).ewm(alpha=1/length, adjust=False).mean()

    plus_di = 100 * (plus_dm_s / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm_s / atr.replace(0, np.nan))

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1/length, adjust=False).mean()
    return plus_di, minus_di, adx

def mfi(high, low, close, volume, length=14):
    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)
    volume = volume.astype(float)

    tp = (high + low + close) / 3.0
    mf = tp * volume

    direction = tp.diff()
    pos_mf = mf.where(direction > 0, 0.0)
    neg_mf = mf.where(direction < 0, 0.0)

    pos_sum = pos_mf.rolling(length).sum()
    neg_sum = neg_mf.rolling(length).sum()

    mfr = pos_sum / neg_sum.replace(0, np.nan)
    return 100 - (100 / (1 + mfr))

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    df = force_numeric_ohlcv(df)
    if len(df) < 35:
        return df
    m_line, s_line, h_line = macd(df["close"], 12, 26, 9)
    df["macd"] = m_line
    df["macd_signal"] = s_line
    df["macd_hist"] = h_line
    df["rsi"] = rsi(df["close"], 14)
    pdi, mdi, adx = dmi_adx(df["high"], df["low"], df["close"], 14)
    df["+di"] = pdi
    df["-di"] = mdi
    df["adx"] = adx
    df["mfi"] = mfi(df["high"], df["low"], df["close"], df["volume"], 14)
    return df

# ==========================================================
# SIGNAL LOGIC (sample)
# ==========================================================
def macd_cross_up(df):
    if len(df) < 3:
        return False
    a0 = df["macd"].iloc[-2] - df["macd_signal"].iloc[-2]
    a1 = df["macd"].iloc[-1] - df["macd_signal"].iloc[-1]
    return (a0 <= 0) and (a1 > 0)

def macd_cross_down(df):
    if len(df) < 3:
        return False
    a0 = df["macd"].iloc[-2] - df["macd_signal"].iloc[-2]
    a1 = df["macd"].iloc[-1] - df["macd_signal"].iloc[-1]
    return (a0 >= 0) and (a1 < 0)

def compute_signal(rules, idx_df, ce_df, pe_df):
    if idx_df is None or idx_df.empty or len(idx_df) < 35:
        return "NONE", "Index data insufficient"

    last = idx_df.iloc[-1]
    adx_v = float(last.get("adx", np.nan))
    rsi_v = float(last.get("rsi", np.nan))

    if adx_v < float(rules["min_adx"]):
        return "NONE", f"ADX<{rules['min_adx']}"

    macd_mode = rules.get("macd_mode", "cross")
    bullish_macd = macd_cross_up(idx_df) if macd_mode == "cross" else (last["macd"] > last["macd_signal"])
    bearish_macd = macd_cross_down(idx_df) if macd_mode == "cross" else (last["macd"] < last["macd_signal"])

    buy_ce = bullish_macd and (rsi_v >= float(rules["rsi_buy"]))
    buy_pe = bearish_macd and (rsi_v <= float(rules["rsi_sell"]))

    if rules.get("confirm_with_options", True):
        if buy_ce:
            if ce_df is None or ce_df.empty or len(ce_df) < 35:
                return "NONE", "CE data insufficient"
            ce_rsi = float(ce_df["rsi"].iloc[-1])
            if ce_rsi < float(rules["ce_confirm_rsi"]):
                return "NONE", f"CE confirm failed (RSI<{rules['ce_confirm_rsi']})"

        if buy_pe:
            if pe_df is None or pe_df.empty or len(pe_df) < 35:
                return "NONE", "PE data insufficient"
            pe_rsi = float(pe_df["rsi"].iloc[-1])
            if pe_rsi < float(rules["pe_confirm_rsi"]):
                return "NONE", f"PE confirm failed (RSI<{rules['pe_confirm_rsi']})"

    if buy_ce:
        return "BUY_CE", "Conditions met"
    if buy_pe:
        return "BUY_PE", "Conditions met"
    return "NONE", "No setup"

# ==========================================================
# LOCAL BOT LOOP (optional)
# ==========================================================
def cooldown_ok(state, rules):
    last_ts = state.get("last_trade_ts_ist", "")
    if not last_ts:
        return True
    try:
        last_dt = datetime.strptime(last_ts, "%Y-%m-%d %H:%M:%S")
        now = ist_now().replace(tzinfo=None)
        mins = (now - last_dt).total_seconds() / 60.0
        return mins >= float(rules.get("cooldown_minutes", 0))
    except Exception:
        return True

def execute_signal(state, rules, signal, ce_key, pe_key, ce_df, pe_df):
    position = state.get("position", "FLAT")

    if rules.get("one_trade_at_a_time", True) and position != "FLAT":
        return "SKIP", f"Already in position: {position}"

    if not cooldown_ok(state, rules):
        return "SKIP", "Cooldown active"

    if signal == "BUY_CE":
        price = float(ce_df["close"].iloc[-1]) if (ce_df is not None and not ce_df.empty) else np.nan
        append_trade({
            "ts_ist": ist_now().strftime("%Y-%m-%d %H:%M:%S"),
            "signal": "BUY_CE",
            "symbol": ce_key,
            "price": price,
            "note": "PAPER BUY CE" if rules.get("paper_mode", True) else "LIVE BUY CE (not implemented)"
        })
        state["position"] = "LONG_CE"
        state["last_trade_ts_ist"] = ist_now().strftime("%Y-%m-%d %H:%M:%S")
        return "EXECUTED", "BUY_CE"

    if signal == "BUY_PE":
        price = float(pe_df["close"].iloc[-1]) if (pe_df is not None and not pe_df.empty) else np.nan
        append_trade({
            "ts_ist": ist_now().strftime("%Y-%m-%d %H:%M:%S"),
            "signal": "BUY_PE",
            "symbol": pe_key,
            "price": price,
            "note": "PAPER BUY PE" if rules.get("paper_mode", True) else "LIVE BUY PE (not implemented)"
        })
        state["position"] = "LONG_PE"
        state["last_trade_ts_ist"] = ist_now().strftime("%Y-%m-%d %H:%M:%S")
        return "EXECUTED", "BUY_PE"

    return "NOOP", "No action"

def bot_loop():
    ensure_paths()
    log_line("BOT STARTED (LOCAL MODE)")

    # Bot always uses AUTO mode (today else fallback)
    data_mode = "AUTO"

    while True:
        state = read_state()
        rules = state.get("rules", DEFAULT_RULES)

        try:
            now = ist_now()
            mkt = market_state()
            today_ref = date_str(ist_today())

            expiry = get_nearest_expiry()
            spot, strike, ce_key, pe_key = get_atm_ce_pe_keys(expiry)

            idx_day, idx_df, idx_mode = get_data_by_mode(UNDERLYING_KEY, data_mode, ist_today())
            ce_day, ce_df, ce_mode = get_data_by_mode(ce_key, data_mode, ist_today())
            pe_day, pe_df, pe_mode = get_data_by_mode(pe_key, data_mode, ist_today())

            idx_df = add_indicators(idx_df)
            ce_df = add_indicators(ce_df)
            pe_df = add_indicators(pe_df)

            signal, reason = compute_signal(rules, idx_df, ce_df, pe_df)

            exec_status, exec_note = "NONE", ""
            if signal != "NONE":
                exec_status, exec_note = execute_signal(state, rules, signal, ce_key, pe_key, ce_df, pe_df)

            last_idx_time = idx_df["time"].iloc[-1] if (idx_df is not None and not idx_df.empty) else None

            state.update({
                "status": "RUNNING",
                "last_update_ist": now.strftime("%Y-%m-%d %H:%M:%S"),
                "market_state": mkt,
                "today_ref": today_ref,
                "expiry": expiry,
                "spot": spot,
                "atm_strike": strike,
                "index_mode": idx_mode,
                "ce_mode": ce_mode,
                "pe_mode": pe_mode,
                "last_signal": signal,
                "last_signal_ts_ist": now.strftime("%Y-%m-%d %H:%M:%S"),
                "last_index_candle": str(last_idx_time) if last_idx_time else "",
                "errors": "",
                "last_reason": reason,
                "last_exec_status": exec_status,
                "last_exec_note": exec_note
            })
            write_state(state)

            log_line(f"Market={mkt} Spot={spot:.2f} ATM={strike} | IDX={idx_mode} | SIG={signal} ({reason}) | EXEC={exec_status} {exec_note}")

        except Exception as e:
            state["status"] = "ERROR"
            state["errors"] = str(e)
            state["last_update_ist"] = ist_now().strftime("%Y-%m-%d %H:%M:%S")
            write_state(state)
            log_line(f"ERROR: {e}")

        time.sleep(UPDATE_EVERY_SECONDS)

# ==========================================================
# UI APP (Streamlit Cloud Safe)
# ==========================================================
def ui_app():
    import streamlit as st

    # Use token from Streamlit Secrets if available
    global ACCESS_TOKEN
    if "ACCESS_TOKEN" in st.secrets:
        ACCESS_TOKEN = st.secrets["ACCESS_TOKEN"]

    ensure_paths()
    st.set_page_config(page_title="NIFTY Bot UI", layout="wide")
    st.title("📈 NIFTY Bot UI (Index + ATM CE/PE)")
    st.caption("Cloud UI mode: refreshes on demand. For auto trading run bot locally:  python nifty_bot_ui.py bot")

    colA, colB = st.columns([1, 4])
    if colA.button("🔄 Refresh now"):
        st.rerun()
    colB.caption("Tip: LIVE_TODAY shows only today's candles; AUTO falls back to last available day; PICK lets you choose any date.")

    state = read_state()
    rules = state.get("rules", DEFAULT_RULES)

    st.write("---")
    st.subheader("Data Mode")

    mode_ui = st.radio(
        "Select what candles you want",
        ["LIVE_TODAY (only today running)", "AUTO (today else last day)", "PICK A DATE (past day)"],
        index=0
    )

    picked_date = st.date_input("Pick a date (used only for PICK A DATE)", value=ist_today())
    picked_day = picked_date  # date object

    if mode_ui.startswith("LIVE_TODAY"):
        data_mode = "LIVE_TODAY"
    elif mode_ui.startswith("AUTO"):
        data_mode = "AUTO"
    else:
        data_mode = "PICK"

    err = ""
    try:
        mkt = market_state()
        expiry = get_nearest_expiry()
        spot, strike, ce_key, pe_key = get_atm_ce_pe_keys(expiry)

        idx_day, idx_df, idx_mode = get_data_by_mode(UNDERLYING_KEY, data_mode, picked_day)
        ce_day, ce_df, ce_mode = get_data_by_mode(ce_key, data_mode, picked_day)
        pe_day, pe_df, pe_mode = get_data_by_mode(pe_key, data_mode, picked_day)

        idx_df = add_indicators(idx_df)
        ce_df = add_indicators(ce_df)
        pe_df = add_indicators(pe_df)

        signal, reason = compute_signal(rules, idx_df, ce_df, pe_df)

        last_idx_time = idx_df["time"].iloc[-1] if (idx_df is not None and not idx_df.empty) else None

        state.update({
            "status": "UI_LIVE",
            "last_update_ist": ist_now().strftime("%Y-%m-%d %H:%M:%S"),
            "market_state": mkt,
            "today_ref": date_str(ist_today()),
            "expiry": expiry,
            "spot": spot,
            "atm_strike": strike,
            "index_mode": idx_mode,
            "ce_mode": ce_mode,
            "pe_mode": pe_mode,
            "last_signal": signal,
            "last_signal_ts_ist": ist_now().strftime("%Y-%m-%d %H:%M:%S"),
            "last_index_candle": str(last_idx_time) if last_idx_time else "",
            "errors": "",
            "last_reason": reason
        })
        write_state(state)

    except Exception as e:
        err = str(e)
        state["status"] = "ERROR"
        state["errors"] = err
        state["last_update_ist"] = ist_now().strftime("%Y-%m-%d %H:%M:%S")
        write_state(state)

    c1, c2, c3 = st.columns(3)
    c1.metric("Status", state.get("status", ""))
    c2.metric("Market", state.get("market_state", ""))
    c3.metric("Last Update (IST)", state.get("last_update_ist", ""))

    st.write("### Snapshot")
    a, b, c, d = st.columns(4)
    a.metric("Today(ref)", state.get("today_ref", ""))
    b.metric("Expiry", state.get("expiry", ""))
    c.metric("Spot", f"{state.get('spot', 0):.2f}" if state.get("spot") else "—")
    d.metric("ATM Strike", str(state.get("atm_strike", "—")))

    st.write("**Last Index Candle Time:**", state.get("last_index_candle", ""))

    st.write("### Signal")
    s1, s2 = st.columns(2)
    s1.metric("Last Signal", state.get("last_signal", "NONE"))
    s2.metric("Reason", state.get("last_reason", ""))

    if state.get("errors"):
        st.error(state["errors"])

    st.write("---")
    st.subheader("Strategy Rules (edit & save)")
    with st.form("rules_form"):
        rules["min_adx"] = st.number_input("Min ADX (Index)", value=float(rules.get("min_adx", 18)))
        rules["rsi_buy"] = st.number_input("RSI Buy threshold (Index)", value=float(rules.get("rsi_buy", 55)))
        rules["rsi_sell"] = st.number_input("RSI Sell threshold (Index)", value=float(rules.get("rsi_sell", 45)))
        rules["macd_mode"] = st.selectbox("MACD Mode", ["cross", "above"], index=0 if rules.get("macd_mode", "cross") == "cross" else 1)

        rules["confirm_with_options"] = st.checkbox("Confirm with Options", value=bool(rules.get("confirm_with_options", True)))
        rules["ce_confirm_rsi"] = st.number_input("CE confirm RSI >=", value=float(rules.get("ce_confirm_rsi", 52)))
        rules["pe_confirm_rsi"] = st.number_input("PE confirm RSI >=", value=float(rules.get("pe_confirm_rsi", 52)))

        rules["one_trade_at_a_time"] = st.checkbox("One trade at a time", value=bool(rules.get("one_trade_at_a_time", True)))
        rules["cooldown_minutes"] = st.number_input("Cooldown minutes", value=float(rules.get("cooldown_minutes", 5)))

        rules["paper_mode"] = st.checkbox("Paper Mode (no real orders)", value=bool(rules.get("paper_mode", True)))
        rules["live_trading"] = st.checkbox("LIVE trading (not implemented here)", value=bool(rules.get("live_trading", False)))

        if st.form_submit_button("Save Rules"):
            state["rules"] = rules
            write_state(state)
            st.success("Saved ✅")

    st.write("---")
    st.subheader("Latest Candles (with indicators)")

    try:
        if 'idx_df' in locals() and idx_df is not None and not idx_df.empty:
            st.write(f"#### INDEX ({state.get('index_mode','')})")
            st.dataframe(idx_df.tail(200), use_container_width=True)
        else:
            st.info("INDEX: No candles returned for selected mode/date.")

        if 'ce_df' in locals() and ce_df is not None and not ce_df.empty:
            st.write(f"#### ATM CE ({state.get('ce_mode','')})")
            st.dataframe(ce_df.tail(200), use_container_width=True)
        else:
            st.info("ATM CE: No candles returned for selected mode/date.")

        if 'pe_df' in locals() and pe_df is not None and not pe_df.empty:
            st.write(f"#### ATM PE ({state.get('pe_mode','')})")
            st.dataframe(pe_df.tail(200), use_container_width=True)
        else:
            st.info("ATM PE: No candles returned for selected mode/date.")

    except Exception as e:
        st.warning(str(e))

    st.write("---")
    st.subheader("Trades Log (local bot writes here)")
    try:
        tdf = pd.read_csv(TRADES_FILE)
        st.dataframe(tdf.tail(200), use_container_width=True)
    except Exception:
        st.info("No trades file yet (or not persisted on cloud).")


# ==========================================================
# ENTRYPOINT
# ==========================================================
if __name__ == "__main__":
    ensure_paths()
    if len(sys.argv) >= 2 and sys.argv[1].lower() == "bot":
        bot_loop()
    else:
        # Local UI run: streamlit run nifty_bot_ui.py
        ui_app()
