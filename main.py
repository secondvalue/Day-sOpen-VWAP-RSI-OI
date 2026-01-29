"""
NIFTY 50 OPTIONS INTRADAY TRADING BOT - FIXED VERSION V4.1 (TUESDAY EXPIRY)
- OI interpretation corrected (PE > CE => Bullish; CE > PE => Bearish)
- OI delta confirmation (Strong Bullish / Strong Bearish)
- VWAP per day, explicit 09:15 open candle, robust RSI
- NIFTY expiry fixed to TUESDAY
- No environment variables — direct Python config (replace placeholders)
"""

import os
import requests
import pandas as pd
import numpy as np
import datetime as dt
import time
import csv
import logging
from zoneinfo import ZoneInfo
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ==================== CONFIGURATION ====================
ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiI1NUJBOVgiLCJqdGkiOiI2OTdhZDYyOGNiMDhkNjFmZjg5NmE1NzQiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNQbHVzUGxhbiI6ZmFsc2UsImlhdCI6MTc2OTY1Nzg5NiwiaXNzIjoidWRhcGktZ2F0ZXdheS1zZXJ2aWNlIiwiZXhwIjoxNzY5NzI0MDAwfQ.3fy6jzd6BNKgCBXIlqTz0mfTXUzjVP8kT_25jYJ97vE"        # <<-- Replace with your Upstox API token
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1412386951474057299/Jgft_nxzGxcfWOhoLbSWMde-_bwapvqx8l3VQGQwEoR7_8n4b9Q9zN242kMoXsVbLdvG"  # <<-- Optional, leave blank if unused
NIFTY_SYMBOL = "NSE_INDEX|Nifty 50"

SIGNAL_COOLDOWN = 300     # seconds
LOT_SIZE = 75
TAKE_PROFIT = 1000
STOP_LOSS = 1500
TRAILING_STOP = 500
MIN_5MIN_BARS = 1

TRADE_LOGS_DIR = "trade_logs"
TERMINAL_LOGS_DIR = "terminal_logs"
os.makedirs(TRADE_LOGS_DIR, exist_ok=True)
os.makedirs(TERMINAL_LOGS_DIR, exist_ok=True)

KOLKATA = ZoneInfo("Asia/Kolkata")
timestamp = dt.datetime.now(tz=KOLKATA).strftime('%Y-%m-%d_%I%M%S_%p')
CSV_FILE = os.path.join(TRADE_LOGS_DIR, f"{timestamp}_trades.csv")
TERMINAL_LOG_FILE = os.path.join(TERMINAL_LOGS_DIR, f"{timestamp}_terminal.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %I:%M:%S %p',
    handlers=[
        logging.FileHandler(TERMINAL_LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== STATE ====================
last_signal_time = None
current_expiry_date = None
contracts_cache = []
open_position = None
days_open_cache = None
last_oi_snapshot = {'ce': 0, 'pe': 0}

# ==================== HTTP SESSION ====================
session = requests.Session()
retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429,500,502,503,504])
adapter = HTTPAdapter(max_retries=retries)
session.mount('https://', adapter)
session.mount('http://', adapter)

# ==================== HELPERS ====================
def encode_symbol(sym): return sym.replace('|', '%7C').replace(' ', '%20')
def get_now_kolkata(): return dt.datetime.now(tz=KOLKATA)

def get_next_weekly_expiry(weekday_target=1):
    """
    Returns next weekly expiry date (default: Tuesday for NIFTY 50)
    weekday_target: 0=Mon, 1=Tue, 2=Wed, 3=Thu, ...
    """
    today = get_now_kolkata()
    days_ahead = (weekday_target - today.weekday()) % 7
    if days_ahead == 0:
        if today.hour < 15 or (today.hour == 15 and today.minute < 30):
            expiry = today
        else:
            expiry = today + dt.timedelta(days=7)
    else:
        expiry = today + dt.timedelta(days=days_ahead)
    return expiry.strftime('%Y-%m-%d')

def get_arrow(cur, ref): return "🔺" if cur > ref else "🔻" if cur < ref else "➡️"

def get_rsi_label(rsi):
    if rsi > 70:
        return "OVERBOUGHT ⚠️"
    elif rsi > 60:
        return "STRONG BULL ✅"
    elif rsi < 30:
        return "OVERSOLD ⚠️"
    elif rsi < 40:
        return "STRONG BEAR ✅"
    else:
        return "NEUTRAL"

# ==================== POSITION CLASS ====================
class Position:
    def __init__(self, signal_type, strike, entry_premium, instrument_key, timestamp):
        self.signal_type = signal_type
        self.strike = strike
        self.entry_premium = float(entry_premium)
        self.instrument_key = instrument_key
        self.timestamp = timestamp
        self.lot_size = LOT_SIZE
        self.highest_pnl = 0
        self.trailing_stop_active = False
        self.trailing_stop_price = None

    def calculate_pnl(self, current_premium):
        diff = current_premium - self.entry_premium
        pnl = diff * self.lot_size
        if pnl > self.highest_pnl: self.highest_pnl = pnl
        return pnl, diff

    def check_exit(self, current_premium):
        pnl, diff = self.calculate_pnl(current_premium)
        if pnl <= -STOP_LOSS:
            return True, f"STOP LOSS (Loss: ₹{abs(pnl):.2f})", pnl, diff
        if pnl >= TAKE_PROFIT:
            if not self.trailing_stop_active:
                self.trailing_stop_active = True
                self.trailing_stop_price = current_premium - (TRAILING_STOP / self.lot_size)
                logger.info(f"  🎯 Take Profit reached! Trailing stop: ₹{self.trailing_stop_price:.2f}")
        if self.trailing_stop_active:
            if current_premium <= self.trailing_stop_price:
                return True, f"TRAILING STOP (Profit: ₹{pnl:.2f})", pnl, diff
            new_trail = current_premium - (TRAILING_STOP / self.lot_size)
            if new_trail > self.trailing_stop_price:
                self.trailing_stop_price = new_trail
                logger.info(f"  📈 Trailing stop updated: ₹{self.trailing_stop_price:.2f}")
        return False, None, pnl, diff

# ==================== DISCORD ====================
def send_discord_alert(title, description, color=0x00ff00, fields=None):
    if not DISCORD_WEBHOOK_URL or "YOUR_DISCORD_WEBHOOK_URL" in DISCORD_WEBHOOK_URL:
        logger.debug("Discord webhook not configured; skipping alert")
        return
    embed = {
        "title": title,
        "description": description,
        "color": color,
        "timestamp": get_now_kolkata().astimezone(dt.timezone.utc).isoformat(),
        "footer": {"text": f"Day's Open | Lot: {LOT_SIZE}"}
    }
    if fields:
        embed["fields"] = fields
    try:
        r = session.post(DISCORD_WEBHOOK_URL, json={"embeds": [embed]}, timeout=10)
        if r.status_code in (200,204):
            logger.info("  ✅ Discord alert sent")
        else:
            logger.warning(f"  ⚠️ Discord webhook returned {r.status_code}")
    except Exception as e:
        logger.debug(f"  ❌ Discord send error: {e}")

# ==================== DATA FETCHING & PARSING ====================
def get_spot_price():
    try:
        encoded_symbol = encode_symbol(NIFTY_SYMBOL)
        url = f"https://api.upstox.com/v2/market-quote/quotes?instrument_key={encoded_symbol}"
        headers = {"accept": "application/json", "Authorization": f"Bearer {ACCESS_TOKEN}"}
        r = session.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            payload = data.get('data', {})
            for k, v in payload.items():
                if isinstance(v, dict) and ('last_price' in v or 'ltp' in v):
                    return v.get('last_price') or v.get('ltp')
        return None
    except Exception:
        return None

def get_option_instruments():
    global current_expiry_date, contracts_cache
    current_expiry_date = get_next_weekly_expiry()  # Tuesday
    encoded_symbol = encode_symbol(NIFTY_SYMBOL)
    url = f"https://api.upstox.com/v2/option/contract?instrument_key={encoded_symbol}&expiry_date={current_expiry_date}"
    headers = {"accept": "application/json", "Authorization": f"Bearer {ACCESS_TOKEN}"}
    try:
        r = session.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            contracts_cache = data.get('data') or []
        if not contracts_cache:
            url2 = f"https://api.upstox.com/v2/option/contract?instrument_key={encoded_symbol}"
            r2 = session.get(url2, headers=headers, timeout=10)
            if r2.status_code == 200:
                data2 = r2.json()
                all_contracts = data2.get('data') or []
                if all_contracts:
                    expiries = sorted(set([c.get('expiry') for c in all_contracts if c.get('expiry')]))
                    if expiries:
                        nearest = expiries[0]
                        current_expiry_date = nearest
                        contracts_cache = [c for c in all_contracts if c.get('expiry') == nearest]
        if not contracts_cache:
            return []
        spot = get_spot_price()
        if spot is not None:
            filtered = [c.get('instrument_key') for c in contracts_cache if c.get('instrument_key') and abs(c.get('strike_price', 0) - spot) <= 500]
            return filtered if filtered else [c.get('instrument_key') for c in contracts_cache[:50] if c.get('instrument_key')]
        else:
            return [c.get('instrument_key') for c in contracts_cache[:50] if c.get('instrument_key')]
    except Exception as e:
        logger.error(f"  ❌ Option instruments error: {e}")
        return []

def get_current_premium(instrument_key):
    try:
        url = f"https://api.upstox.com/v2/market-quote/quotes?instrument_key={instrument_key}"
        headers = {"accept": "application/json", "Authorization": f"Bearer {ACCESS_TOKEN}"}
        r = session.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json().get('data', {})
            for k, v in data.items():
                if isinstance(v, dict):
                    premium = v.get('last_price') or v.get('ltp') or v.get('last_traded_price')
                    if premium is not None:
                        return float(premium)
        return None
    except Exception:
        return None

# ==================== FIXED: GET LIVE OI with DELTA CONFIRMATION ====================
def detect_ce_pe_from_key(instrument_key: str):
    ik = (instrument_key or '').upper()
    if ik.endswith('_CE') or '_CE_' in ik or ik.split('|')[-1].startswith('CE') or ik.split('|')[-1].endswith('CE'):
        return 'CE'
    if ik.endswith('_PE') or '_PE_' in ik or ik.split('|')[-1].startswith('PE') or ik.split('|')[-1].endswith('PE'):
        return 'PE'
    if ' CE ' in ik or 'CE|' in ik:
        return 'CE'
    if ' PE ' in ik or 'PE|' in ik:
        return 'PE'
    return None

def get_live_oi_from_quotes(instrument_keys):
    """
    Returns: (trend, ce_oi_total, pe_oi_total, confirmation)
    confirmation: 'Strong Bullish', 'Strong Bearish', 'Weak / Neutral', or 'Neutral' if no data
    """
    global last_oi_snapshot
    if not instrument_keys:
        return None, 0, 0, 'Neutral'

    ce_oi_total = 0
    pe_oi_total = 0
    batch_size = 80

    for i in range(0, len(instrument_keys), batch_size):
        batch = instrument_keys[i:i+batch_size]
        instrument_param = ",".join(batch)
        url = f"https://api.upstox.com/v2/market-quote/quotes?instrument_key={instrument_param}"
        headers = {"accept": "application/json", "Authorization": f"Bearer {ACCESS_TOKEN}"}
        try:
            r = session.get(url, headers=headers, timeout=10)
            if r.status_code != 200:
                logger.debug(f"  ⚠️ Quotes batch returned {r.status_code}")
                continue
            data = r.json().get('data', {})
            for ik, q in data.items():
                if not isinstance(q, dict):
                    continue
                oi_value = q.get('oi') or q.get('open_interest') or 0
                try:
                    oi_value = int(float(oi_value))
                except Exception:
                    oi_value = 0
                kind = detect_ce_pe_from_key(ik)
                if kind == 'CE':
                    ce_oi_total += oi_value
                elif kind == 'PE':
                    pe_oi_total += oi_value
        except Exception as e:
            logger.debug(f"  ❌ OI batch error: {e}")
            continue

    if ce_oi_total == 0 and pe_oi_total == 0:
        return None, 0, 0, 'Neutral'

    # Interpretation: PE > CE => Bullish (support); CE > PE => Bearish (resistance)
    if pe_oi_total > ce_oi_total * 1.05:
        trend = 'Bullish'
    elif ce_oi_total > pe_oi_total * 1.05:
        trend = 'Bearish'
    else:
        trend = 'Sideways'

    # Delta OI confirmation
    delta_ce = ce_oi_total - last_oi_snapshot.get('ce', 0)
    delta_pe = pe_oi_total - last_oi_snapshot.get('pe', 0)

    if trend == 'Bullish' and delta_pe > delta_ce:
        confirmation = 'Strong Bullish'
    elif trend == 'Bearish' and delta_ce > delta_pe:
        confirmation = 'Strong Bearish'
    else:
        confirmation = 'Weak / Neutral'

    last_oi_snapshot = {'ce': ce_oi_total, 'pe': pe_oi_total}

    logger.info(f"  🧮 OI Δ (CE: {delta_ce:+,}, PE: {delta_pe:+,}) → Confirmation: {confirmation}")
    return trend, ce_oi_total, pe_oi_total, confirmation

# ==================== INDICATORS ====================
def calculate_vwap_rsi(df):
    df = df.copy()
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
    else:
        df = df.reset_index().rename(columns={'index': 'time'})
        df['time'] = pd.to_datetime(df['time'])

    df['date'] = df['time'].dt.date
    df['TP'] = (df['high'] + df['low'] + df['close']) / 3.0
    df['TPV'] = df['TP'] * df['volume']
    df['Cumulative_TPV'] = df.groupby('date')['TPV'].cumsum()
    df['Cumulative_Volume'] = df.groupby('date')['volume'].cumsum()
    df['VWAP'] = df['Cumulative_TPV'] / df['Cumulative_Volume']
    df['VWAP'] = df['VWAP'].fillna(df['close'])

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()

    rs = pd.Series(np.zeros(len(df)), index=df.index)
    mask_loss_pos = avg_loss > 0
    rs[mask_loss_pos] = avg_gain[mask_loss_pos] / avg_loss[mask_loss_pos]

    rsi = pd.Series(np.full(len(df), 50.0), index=df.index)
    rsi[mask_loss_pos] = 100 - (100 / (1 + rs[mask_loss_pos]))

    mask_loss_zero_gain_pos = (avg_loss == 0) & (avg_gain > 0)
    rsi[mask_loss_zero_gain_pos] = 100.0

    mask_gain_zero_loss_pos = (avg_gain == 0) & (avg_loss > 0)
    rsi[mask_gain_zero_loss_pos] = 0.0

    df['RSI'] = rsi.fillna(50.0)

    df.drop(columns=['TPV', 'Cumulative_TPV', 'Cumulative_Volume'], errors='ignore', inplace=True)
    return df

# ==================== FIND ATM & PREMIUM ====================
def is_option_type_match(contract, optype):
    if not contract:
        return False
    t = str(contract.get('instrument_type') or contract.get('option_type') or '').upper()
    if t in ('CE', 'CALL'):
        return optype == 'CE'
    if t in ('PE', 'PUT'):
        return optype == 'PE'
    ik = str(contract.get('instrument_key') or '').upper()
    if '_CE' in ik or ik.split('|')[-1].startswith('CE'):
        return optype == 'CE'
    if '_PE' in ik or ik.split('|')[-1].startswith('PE'):
        return optype == 'PE'
    return False

def find_atm_strike_and_premium(spot_price, option_type):
    global contracts_cache
    try:
        if not contracts_cache:
            return None, None, None
        strikes = [c for c in contracts_cache if is_option_type_match(c, option_type)]
        if not strikes:
            return None, None, None
        atm_contract = min(strikes, key=lambda x: abs(x.get('strike_price', 1e9) - spot_price))
        atm_strike = atm_contract.get('strike_price')
        instrument_key = atm_contract.get('instrument_key')
        premium = get_current_premium(instrument_key)
        if premium is not None:
            return atm_strike, premium, instrument_key
        return atm_strike, 0.0, instrument_key
    except Exception:
        return None, None, None

# ==================== SIGNAL LOGIC ====================
def check_signal_conditions(spot, day_open, vwap, rsi, oi_trend, oi_confirmation):
    conditions = {
        'CE': {
            'price_above_open': spot > day_open,
            'price_above_vwap': spot > vwap,
            'rsi_bullish': rsi > 60,
            'oi_bullish': oi_trend == 'Bullish' and oi_confirmation.startswith('Strong')
        },
        'PE': {
            'price_below_open': spot < day_open,
            'price_below_vwap': spot < vwap,
            'rsi_bearish': rsi < 40,
            'oi_bearish': oi_trend == 'Bearish' and oi_confirmation.startswith('Strong')
        }
    }

    if all(conditions['CE'].values()):
        return 'BUY CE', conditions
    if all(conditions['PE'].values()):
        return 'BUY PE', conditions
    return None, conditions

# ==================== DISPLAY & LOGGING ====================
def print_startup_banner():
    session_time = get_now_kolkata().strftime('%Y-%m-%d %I:%M:%S %p')
    banner = f"""
{'=' * 85}
🚀 NIFTY 50 OPTIONS INTRADAY TRADING BOT - PATCHED V4.1
{'=' * 85}
Strategy:    Day's Open + VWAP + RSI + OI + OI-Delta Confirmation
Timeframe:   5-Minute Candles (1-min resampled)
Data Source: Live from NSE via Upstox API
Session:     {session_time}
Trade Log:   {CSV_FILE}
Terminal Log: {TERMINAL_LOG_FILE}
Expiry:      {current_expiry_date}
Lot Size:    {LOT_SIZE} quantity
Take Profit: ₹{TAKE_PROFIT} | Stop Loss: ₹{STOP_LOSS} | Trail: ₹{TRAILING_STOP}

✅ FIXED: OI Interpretation (PE>CE => Bullish; CE>PE => Bearish)
✅ NEW: OI Δ Confirmation
✅ FIXED: Day's Open (explicit 09:15 candle)
✅ FIXED: RSI & VWAP calculation robustness
{'=' * 85}
"""
    logger.info(banner)

def print_market_snapshot(spot, day_open, vwap, rsi, oi_trend, oi_ce, oi_pe):
    snapshot = f"""
📊 MARKET SNAPSHOT
{'-' * 85}
  Spot Price:    {spot:8.2f}  |  Day's Open:   {day_open:8.2f}  {get_arrow(spot, day_open)}
  VWAP:          {vwap:8.2f}  |  Position:     {'ABOVE ✅' if spot > vwap else 'BELOW ❌'}
  RSI:           {rsi:8.2f}  |  Momentum:     {get_rsi_label(rsi)}
  OI Trend:      {oi_trend:>8}  |  CE OI: {oi_ce:,} | PE OI: {oi_pe:,}
"""
    logger.info(snapshot)

def print_signal_evaluation(conditions):
    ce = conditions['CE']
    pe = conditions['PE']
    ce_result = "🔔 TRIGGER!" if all(ce.values()) else "❌ NO"
    pe_result = "🔔 TRIGGER!" if all(pe.values()) else "❌ NO"
    evaluation = f"""
🔍 SIGNAL EVALUATION (All ✅ required for trade)
{'-' * 85}
  CALL: {'✅' if ce['price_above_open'] else '❌'} Open  {'✅' if ce['price_above_vwap'] else '❌'} VWAP  {'✅' if ce['rsi_bullish'] else '❌'} RSI>60  {'✅' if ce['oi_bullish'] else '❌'} OI-Bull  →  {ce_result}
  PUT:  {'✅' if pe['price_below_open'] else '❌'} Open  {'✅' if pe['price_below_vwap'] else '❌'} VWAP  {'✅' if pe['rsi_bearish'] else '❌'} RSI<40  {'✅' if pe['oi_bearish'] else '❌'} OI-Bear  →  {pe_result}
"""
    logger.info(evaluation)

def print_trade_alert(timestamp, signal, strike, premium, spot):
    alert = f"""
{'=' * 85}
🔔 TRADE SIGNAL GENERATED!
{'=' * 85}
  Time:        {timestamp}
  Action:      {signal}
  Strike:      {strike}
  Premium:     ₹{premium:.2f}
  Lot Size:    {LOT_SIZE}
  Investment:  ₹{premium * LOT_SIZE:.2f}
  Spot:        {spot:.2f}
  Expiry:      {current_expiry_date}
  CSV Logged:  ✅
{'=' * 85}
"""
    logger.info(alert)

def log_trade_to_csv(timestamp, signal, strike, premium, spot, rsi, vwap, day_open, oi_trend, exit_reason=None, pnl=None, premium_diff=None):
    with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp,
            signal,
            strike,
            premium,
            round(spot, 2) if spot is not None else "",
            round(rsi, 2) if rsi is not None else "",
            round(vwap, 2) if vwap is not None else "",
            round(day_open, 2) if day_open is not None else "",
            oi_trend or "",
            exit_reason or "",
            round(pnl, 2) if pnl is not None else "",
            round(premium_diff, 2) if premium_diff is not None else ""
        ])

# ==================== GET DAY'S OPEN (explicit 09:15) ====================
def get_days_open_from_intraday():
    global days_open_cache
    if days_open_cache is not None:
        return days_open_cache
    try:
        encoded_symbol = encode_symbol(NIFTY_SYMBOL)
        url = f"https://api.upstox.com/v2/historical-candle/intraday/{encoded_symbol}/1minute"
        headers = {"accept": "application/json", "Authorization": f"Bearer {ACCESS_TOKEN}"}
        r = session.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            logger.warning(f"  ⚠️  Day's open API returned status {r.status_code}")
            return None
        data = r.json()
        candles = data.get('data', {}).get('candles', [])
        if not candles:
            logger.warning("  ⚠️  Empty candles list")
            return None
        df = pd.DataFrame(candles, columns=["time","open","high","low","close","volume","oi"])
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").reset_index(drop=True)
        today = get_now_kolkata().date()
        df['date'] = df['time'].dt.date
        today_candles = df[df['date'] == today]
        if today_candles.empty:
            logger.warning("  ⚠️  No candles found for today")
            return None
        candle_at_915 = today_candles[today_candles['time'].dt.time == dt.time(9,15)]
        if not candle_at_915.empty:
            first_candle = candle_at_915.iloc[0]
        else:
            after_915 = today_candles[today_candles['time'].dt.time >= dt.time(9,15)]
            if not after_915.empty:
                first_candle = after_915.iloc[0]
            else:
                first_candle = today_candles.iloc[0]
        day_open = float(first_candle['open'])
        days_open_cache = day_open
        candle_time = first_candle['time']
        logger.info(f"  ✅ Day's Open (09:15): {day_open:.2f} (from {candle_time.strftime('%I:%M %p')})")
        return day_open
    except Exception as e:
        logger.error(f"  ❌ Day's open fetch error: {e}")
        return None

# ==================== FETCH LIVE SPOT CANDLES (1-min -> 5-min) ====================
def fetch_live_spot_candles(symbol):
    encoded_symbol = encode_symbol(symbol)
    url = f"https://api.upstox.com/v2/historical-candle/intraday/{encoded_symbol}/1minute"
    headers = {"accept": "application/json", "Authorization": f"Bearer {ACCESS_TOKEN}"}
    try:
        r = session.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        candles = data.get('data', {}).get('candles', [])
        if not candles:
            return None
        df = pd.DataFrame(candles, columns=["time","open","high","low","close","volume","oi"])
        df["time"] = pd.to_datetime(df["time"])
        df["volume"] = df["volume"].replace(0, 1)
        df = df.sort_values("time").reset_index(drop=True)
        df.set_index("time", inplace=True)
        df_5min = df.resample('5min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        df_5min.reset_index(inplace=True)
        logger.info(f"  ✅ Fetched {len(candles)} 1-min → {len(df_5min)} 5-min candles")
        return df_5min
    except Exception as e:
        logger.error(f"  ❌ Candle fetch error: {e}")
        return None

# ==================== MAIN LOOP ====================
def main():
    global last_signal_time, open_position, days_open_cache, current_expiry_date, contracts_cache

    with open(CSV_FILE, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Time", "Signal", "Strike", "Premium",
            "Spot", "RSI", "VWAP", "Day_Open", "OI_Trend",
            "Exit_Reason", "PnL", "Premium_Diff"
        ])

    logger.info("CSV file initialized: " + CSV_FILE)
    logger.info("\n📥 Initializing...")
    option_instruments = get_option_instruments()

    if len(option_instruments) == 0:
        logger.error("❌ Failed to fetch option instruments")
        return

    logger.info(f"✅ Loaded {len(option_instruments)} instruments")
    current_expiry_date = current_expiry_date or get_next_weekly_expiry()
    print_startup_banner()

    iteration = 0
    try:
        while True:
            iteration += 1
            now = get_now_kolkata()

            logger.info("\n" + "="*85)
            logger.info(f"⏰ [{now.strftime('%d-%b-%Y %I:%M:%S %p')}] Iteration #{iteration}")
            logger.info("="*85)

            # Market not open
            if now.hour < 9 or (now.hour == 9 and now.minute < 15):
                logger.info("⏸  Market not open yet (Opens 9:15 AM)")
                time.sleep(60)
                continue

            # Market closed after 15:30
            if (now.hour == 15 and now.minute > 30) or now.hour > 15:
                logger.info("⏸  Market Closed (Closes 3:30 PM)")
                if open_position:
                    current_premium = get_current_premium(open_position.instrument_key)
                    if current_premium:
                        pnl, premium_diff = open_position.calculate_pnl(current_premium)
                        timestamp = now.strftime('%Y-%m-%d %I:%M:%S %p')

                        logger.info("\n💼 CLOSING POSITION AT MARKET CLOSE")
                        logger.info(f"   P&L: ₹{pnl:.2f} (₹{premium_diff:.2f} × {LOT_SIZE})")

                        log_trade_to_csv(timestamp, f"EXIT {open_position.signal_type}", open_position.strike,
                                         current_premium, 0, 0, 0, 0, "", "MARKET CLOSE", pnl, premium_diff)

                        send_discord_alert(
                            "🔔 Position Closed - Market Close",
                            f"**{open_position.signal_type}** | Strike: {open_position.strike}",
                            0xffff00,
                            [
                                {"name": "Entry", "value": f"₹{open_position.entry_premium:.2f}", "inline": True},
                                {"name": "Exit", "value": f"₹{current_premium:.2f}", "inline": True},
                                {"name": "P&L", "value": f"₹{pnl:.2f}", "inline": False}
                            ]
                        )

                        open_position = None

                days_open_cache = None
                time.sleep(60)
                continue

            # If position open - manage
            if open_position:
                logger.info(f"\n💼 OPEN POSITION: {open_position.signal_type} {open_position.strike}")
                logger.info(f"   Entry: ₹{open_position.entry_premium:.2f} | Lot: {LOT_SIZE}")

                current_premium = get_current_premium(open_position.instrument_key)
                if current_premium:
                    pnl, premium_diff = open_position.calculate_pnl(current_premium)

                    logger.info(f"   Current: ₹{current_premium:.2f} | Diff: ₹{premium_diff:.2f}")
                    logger.info(f"   P&L: ₹{pnl:.2f} (₹{premium_diff:.2f} × {LOT_SIZE})")

                    if open_position.trailing_stop_active:
                        logger.info(f"   🎯 Trailing Stop: ₹{open_position.trailing_stop_price:.2f}")

                    should_exit, exit_reason, final_pnl, final_premium_diff = open_position.check_exit(current_premium)

                    if should_exit:
                        timestamp = now.strftime('%Y-%m-%d %I:%M:%S %p')

                        logger.info("\n" + "="*85)
                        logger.info(f"🔔 POSITION CLOSED: {exit_reason}")
                        logger.info("="*85)
                        logger.info(f"  Entry:       ₹{open_position.entry_premium:.2f}")
                        logger.info(f"  Exit:        ₹{current_premium:.2f}")
                        logger.info(f"  Premium Diff: ₹{final_premium_diff:.2f}")
                        logger.info(f"  Total P&L:   ₹{final_pnl:.2f} (₹{final_premium_diff:.2f} × {LOT_SIZE})")
                        logger.info("=" * 85)

                        log_trade_to_csv(timestamp, f"EXIT {open_position.signal_type}", open_position.strike,
                                       current_premium, 0, 0, 0, 0, "", exit_reason, final_pnl, final_premium_diff)

                        color = 0x00ff00 if final_pnl > 0 else 0xff0000
                        send_discord_alert(
                            f"🔔 {exit_reason}",
                            f"**{open_position.signal_type}** | Strike: {open_position.strike}",
                            color,
                            [
                                {"name": "Entry", "value": f"₹{open_position.entry_premium:.2f}", "inline": True},
                                {"name": "Exit", "value": f"₹{current_premium:.2f}", "inline": True},
                                {"name": "P&L", "value": f"₹{final_pnl:.2f}", "inline": False}
                            ]
                        )

                        open_position = None
                        last_signal_time = now

                time.sleep(60)
                continue

            # Normal flow: get day's open and market data
            day_open = get_days_open_from_intraday()
            if day_open is None:
                logger.warning("⚠️  Could not fetch day's open. Retrying in 60s...")
                time.sleep(60)
                continue

            df = fetch_live_spot_candles(NIFTY_SYMBOL)
            if df is None or len(df) == 0:
                logger.warning("\n❌ Failed to fetch candles. Retrying in 60s...")
                time.sleep(60)
                continue

            if len(df) < MIN_5MIN_BARS:
                logger.info(f"⏳ Not enough 5-min bars yet ({len(df)}) — waiting for at least {MIN_5MIN_BARS}")
                time.sleep(60)
                continue

            df = calculate_vwap_rsi(df)
            latest = df.iloc[-1]
            spot = float(latest["close"])
            vwap = float(latest["VWAP"])
            rsi = float(latest["RSI"])

            logger.info(f"  ✅ Spot: {spot:.2f} | VWAP: {vwap:.2f} | RSI: {rsi:.2f}")

            oi_trend, oi_ce, oi_pe, oi_conf = get_live_oi_from_quotes(option_instruments)
            if oi_trend is None:
                oi_trend = "Unknown"
                oi_ce, oi_pe, oi_conf = 0, 0, 'Neutral'
            else:
                logger.info(f"  ✅ Live OI: CE={oi_ce:,} | PE={oi_pe:,} → {oi_trend} ({oi_conf})")

            print_market_snapshot(spot, day_open, vwap, rsi, oi_trend, oi_ce, oi_pe)

            if last_signal_time:
                elapsed = (now - last_signal_time).seconds
                if elapsed < SIGNAL_COOLDOWN:
                    remaining = SIGNAL_COOLDOWN - elapsed
                    logger.info(f"\n⏳ COOLDOWN ACTIVE: {remaining}s remaining until next signal")
                    time.sleep(60)
                    continue

            signal, conditions = check_signal_conditions(spot, day_open, vwap, rsi, oi_trend, oi_conf)
            print_signal_evaluation(conditions)

            if signal:
                option_type = "CE" if signal == "BUY CE" else "PE"
                strike, premium, instrument_key = find_atm_strike_and_premium(spot, option_type)

                if strike is not None and premium is not None and instrument_key:
                    timestamp = now.strftime('%Y-%m-%d %I:%M:%S %p')

                    print_trade_alert(timestamp, signal, strike, premium, spot)

                    open_position = Position(signal, strike, premium, instrument_key, timestamp)

                    log_trade_to_csv(timestamp, signal, strike, premium, spot, rsi, vwap, day_open, oi_trend)

                    send_discord_alert(
                        f"🚀 NEW SIGNAL - {signal}",
                        f"Strike: {strike} | Lot: {LOT_SIZE}",
                        0x00ff00,
                        [
                            {"name": "Premium", "value": f"₹{premium:.2f}", "inline": True},
                            {"name": "Spot", "value": f"{spot:.2f}", "inline": True},
                            {"name": "Investment", "value": f"₹{premium * LOT_SIZE:.2f}", "inline": True}
                        ]
                    )

                    last_signal_time = now
                else:
                    logger.warning("\n⚠️  Signal generated but strike/premium unavailable")
            else:
                logger.info("\n⏸  NO SIGNAL - Waiting for all conditions to align...")

            logger.info("\n⏱  Next check in 60 seconds...")
            time.sleep(60)

    except KeyboardInterrupt:
        logger.info("\n\n" + "="*85)
        logger.info("⏹  BOT STOPPED BY USER")
        logger.info("="*85)
        logger.info(f"All signals saved to: {CSV_FILE}")
        logger.info("="*85)
        logger.info("\n✅ Thank you for using Nifty Options Trading Bot!\n")

    except Exception as e:
        logger.critical(f"\n\n❌ CRITICAL ERROR: {e}", exc_info=True)

if __name__ == "__main__":
    # Preload option instruments used throughout the day
    option_instruments = get_option_instruments()
    if not option_instruments:
        logger.error("Exiting: couldn't fetch option instruments.")
    else:
        main()
