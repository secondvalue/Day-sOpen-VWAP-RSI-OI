"""
NIFTY 50 OPTIONS INTRADAY TRADING BOT - V5.0 LIVE TRADING (TUESDAY EXPIRY)
- ✅ LIVE ORDER EXECUTION via Upstox API
- OI interpretation corrected (PE > CE => Bullish; CE > PE => Bearish)
- OI delta confirmation (Strong Bullish / Strong Bearish)
- VWAP per day, explicit 09:15 open candle, robust RSI
- NIFTY expiry fixed to TUESDAY
- Paper trading mode available (set LIVE_TRADING = False)
"""

import os
import sys
import requests
import threading
import pandas as pd
import numpy as np
import datetime as dt
import time  
import csv
import logging
from zoneinfo import ZoneInfo
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib.parse

# ==================== CONFIGURATION ====================
# Dual token support: Use SANDBOX token for testing, LIVE token for production
SANDBOX_ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiI1NUJBOVgiLCJqdGkiOiI2OTljMTk5ODk1M2M5YjFjMDIxNjRjMzAiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNQbHVzUGxhbiI6dHJ1ZSwiaWF0IjoxNzcxODM3ODQ4LCJpc3MiOiJ1ZGFwaS1nYXRld2F5LXNlcnZpY2UiLCJleHAiOjE3NzQzODk2MDB9.2pU16hshwl2MEdLtm8h3XOc6axRlCTmAO0VSKtnYhbo"  # <<-- Add your SANDBOX token here (do not commit to Git)
LIVE_ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiI1NUJBOVgiLCJqdGkiOiI2OTlmYmY5ZTVjNTdjODY5OTEwOTQwMmUiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNQbHVzUGxhbiI6ZmFsc2UsImlhdCI6MTc3MjA3Njk1OCwiaXNzIjoidWRhcGktZ2F0ZXdheS1zZXJ2aWNlIiwiZXhwIjoxNzcyMTQzMjAwfQ.a_b06OpClZTeXW_3AmJSDivhE422PTUw3nILjL22EP0"  # <<-- Add your LIVE Upstox API token here (do not commit to Git)
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1412386951474057299/Jgft_nxzGxcfWOhoLbSWMde-_bwapvqx8l3VQGQwEoR7_8n4b9Q9zN242kMoXsVbLdvG"
NIFTY_SYMBOL = "NSE_INDEX|Nifty 50"

# ==================== LIVE TRADING SETTINGS ====================
SANDBOX_MODE = False         # <<-- SET TO True FOR SANDBOX TESTING
LIVE_TRADING = True         # <<-- SET TO True to send orders (to Sandbox if SANDBOX_MODE=True)

# Separate tokens for Data (Live) and Orders (Sandbox/Live)
DATA_ACCESS_TOKEN = LIVE_ACCESS_TOKEN
ORDER_ACCESS_TOKEN = SANDBOX_ACCESS_TOKEN if SANDBOX_MODE else LIVE_ACCESS_TOKEN
ACCESS_TOKEN = DATA_ACCESS_TOKEN # Default for data calls, but specific functions will specific tokens

# API endpoints - automatically set based on SANDBOX_MODE
if SANDBOX_MODE:
    ORDER_PLACE_URL = "https://api-sandbox.upstox.com/v3/order/place"
    ORDER_CANCEL_URL = "https://api-sandbox.upstox.com/v3/order/cancel"
else:
    ORDER_PLACE_URL = "https://api-hft.upstox.com/v3/order/place"
    ORDER_CANCEL_URL = "https://api-hft.upstox.com/v3/order/cancel"

SIGNAL_COOLDOWN = 300     # seconds
LOT_SIZE = 65             # NIFTY 50 lot size


# ==================== RISK MANAGEMENT (FIXED) ====================
TAKE_PROFIT = 1000.0    # Activates Trailing Stop when profit reaches ₹500
STOP_LOSS = 1000.0     # Fixed Stop Loss (exit if loss > ₹1500)
TRAILING_STOP = 500.0  # Trailing Step (lock profit in ₹500 chunks)

MIN_5MIN_BARS = 1

# ==================== POLLING INTERVALS ====================
SIGNAL_CHECK_INTERVAL = 60    # seconds - interval when waiting for signals (SLOWER: 60s)
POSITION_MONITOR_INTERVAL = 1 # seconds - interval when position is open (faster for trailing)

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
    handlers=[]
)
logger = logging.getLogger(__name__)

# File Handler: Gets everything
file_handler = logging.FileHandler(TERMINAL_LOG_FILE, encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%Y-%m-%d %I:%M:%S %p'))
logger.addHandler(file_handler)

# Stream Handler (Terminal): Filters out logs tagged with file_only=True
class NoFileOnlyFilter(logging.Filter):
    def filter(self, record):
        return not getattr(record, 'file_only', False)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%Y-%m-%d %I:%M:%S %p'))
stream_handler.addFilter(NoFileOnlyFilter())
logger.addHandler(stream_handler)

# ==================== STATE ====================
last_signal_time = None
current_expiry_date = None
contracts_cache = []
open_position = None
days_open_cache = None
last_oi_snapshot = {'ce': 0, 'pe': 0}
trade_completed_today = False   # One trade per day limit

# ==================== HTTP SESSION ====================
session = requests.Session()
retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429,500,502,503,504])
adapter = HTTPAdapter(max_retries=retries)
session.mount('https://', adapter)
session.mount('http://', adapter)

# ==================== LIVE ORDER EXECUTION ====================
def place_order(instrument_key, transaction_type, quantity, order_tag="ALGO_BOT"):
    """
    Place a live order via Upstox API.
    
    Args:
        instrument_key: The instrument to trade (e.g., 'NFO_OPT|...')
        transaction_type: 'BUY' or 'SELL'
        quantity: Number of shares/lots to trade
        order_tag: Custom tag for order identification
    
    Returns:
        (success: bool, order_id: str or None, message: str)
    """
    if not LIVE_TRADING:
        logger.info(f"  📝 [PAPER TRADE] {transaction_type} {quantity} of {instrument_key}")
        return True, f"PAPER_{int(time.time())}", "Paper trade logged"
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {ORDER_ACCESS_TOKEN}"
    }
    
    payload = {
        "quantity": quantity,
        "product": "I",                    # Intraday
        "validity": "DAY",
        "price": 0,                        # Market order
        "tag": order_tag,
        "instrument_token": instrument_key,
        "order_type": "MARKET",
        "transaction_type": transaction_type,
        "disclosed_quantity": 0,
        "trigger_price": 0,
        "is_amo": False,
        "slice": False                     # Disable auto-slicing for V3 API
    }
    
    try:
        logger.info(f"  🔄 Placing {transaction_type} order for {quantity} qty...")
        r = session.post(ORDER_PLACE_URL, json=payload, headers=headers, timeout=15)
        response_data = r.json()
        
        if r.status_code == 200 and response_data.get('status') == 'success':
            data = response_data.get('data', {})
            # V3 API returns order_ids array, V2 returns order_id
            order_ids = data.get('order_ids', [])
            order_id = order_ids[0] if order_ids else data.get('order_id')
            logger.info(f"  ✅ ORDER PLACED: {transaction_type} | Order ID: {order_id}")
            return True, order_id, "Order placed successfully"
        else:
            error_msg = response_data.get('message', response_data.get('errors', 'Unknown error'))
            logger.error(f"  ❌ ORDER FAILED: {error_msg}")
            return False, None, str(error_msg)
            
    except requests.exceptions.Timeout:
        logger.error("  ❌ ORDER TIMEOUT: Request timed out")
        return False, None, "Request timeout"
    except Exception as e:
        logger.error(f"  ❌ ORDER ERROR: {e}")
        return False, None, str(e)

def exit_position(position):
    """
    Exit an open position by placing a SELL order.
    
    Args:
        position: The Position object to close
    
    Returns:
        (success: bool, order_id: str or None, message: str)
    """
    return place_order(
        instrument_key=position.instrument_key,
        transaction_type="SELL",
        quantity=position.lot_size,
        order_tag=f"EXIT_{position.signal_type.replace(' ', '_')}"
    )

def place_sl_order(instrument_key, quantity, trigger_price, order_tag="SL_ORDER"):
    """
    Place a Stop Loss Limit (SL) order on the exchange.
    This order will automatically trigger when price hits the trigger_price.
    
    Args:
        instrument_key: The instrument to trade
        quantity: Number to sell when SL triggers
        trigger_price: Price at which SL will trigger
        order_tag: Custom tag for identification
    
    Returns:
        (success: bool, order_id: str or None, message: str)
    """
    if not LIVE_TRADING:
        logger.info(f"  📝 [PAPER TRADE] SL Order: SELL {quantity} @ trigger ₹{trigger_price:.2f}")
        return True, f"PAPER_SL_{int(time.time())}", "Paper SL order logged"
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {ORDER_ACCESS_TOKEN}"
    }
    

    # Helper to round to nearest tick size (0.05)
    def round_to_tick(price):
        return round(round(price / 0.05) * 0.05, 2)

    # For SL (Stop Loss Limit), we need both trigger_price and limit_price
    # Limit price is set slightly below trigger to ensure execution
    # SAFETY: Ensure limit_price is always positive (minimum ₹0.05)
    
    # Calculate target limit price (trigger - 2.0)
    raw_limit_price = trigger_price - 2.0
    
    # Round both trigger and limit to tick size
    trigger_price = round_to_tick(trigger_price)
    limit_price = max(0.05, round_to_tick(raw_limit_price))
    
    # Truncate order tag to 20 chars (common API limit)
    safe_order_tag = order_tag[:20] if order_tag else "SL_ORDER"
    
    payload = {
        "quantity": quantity,
        "product": "I",                    # Intraday
        "validity": "DAY",
        "price": limit_price,              # Limit price for SL order
        "tag": safe_order_tag,
        "instrument_token": instrument_key,
        "order_type": "SL",                # Stop Loss Limit (not SL-M)
        "transaction_type": "SELL",
        "trigger_price": trigger_price
    }
    
    try:
        logger.info(f"  🛡️ Placing SL order: Trigger @ ₹{trigger_price:.2f}, Limit @ ₹{limit_price:.2f}...")
        r = session.post(ORDER_PLACE_URL, json=payload, headers=headers, timeout=15)
        response_data = r.json()
        
        if r.status_code == 200 and response_data.get('status') == 'success':
            data = response_data.get('data', {})
            # V3 API returns order_ids array, V2 returns order_id
            order_ids = data.get('order_ids', [])
            order_id = order_ids[0] if order_ids else data.get('order_id')
            logger.info(f"  ✅ SL ORDER PLACED: Trigger @ ₹{trigger_price:.2f} | Order ID: {order_id}")
            return True, order_id, "SL order placed successfully"
        else:
            # Capture full error details for debugging
            error_msg = response_data.get('message', response_data.get('errors', 'Unknown error'))
            logger.error(f"  ❌ SL ORDER FAILED: {error_msg} | Full Response: {response_data}")
            return False, None, str(error_msg)
            
    except Exception as e:
        logger.error(f"  ❌ SL ORDER ERROR: {e}")
        return False, None, str(e)

def cancel_order(order_id):
    """
    Cancel an existing order on the exchange.
    
    Args:
        order_id: The order ID to cancel
    
    Returns:
        (success: bool, message: str)
    """
    if not LIVE_TRADING:
        logger.info(f"  📝 [PAPER TRADE] Cancel order: {order_id}")
        return True, "Paper order cancelled"
    
    if not order_id or order_id.startswith("PAPER"):
        return True, "No real order to cancel"
    
    cancel_url = f"{ORDER_CANCEL_URL}?order_id={order_id}"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {ORDER_ACCESS_TOKEN}"
    }
    
    try:
        logger.info(f"  🚫 Cancelling order: {order_id}...")
        r = session.delete(cancel_url, headers=headers, timeout=15)
        response_data = r.json()
        
        if r.status_code == 200 and response_data.get('status') == 'success':
            logger.info(f"  ✅ ORDER CANCELLED: {order_id}")
            return True, "Order cancelled successfully"
        else:
            error_msg = response_data.get('message', response_data.get('errors', 'Unknown error'))
            logger.warning(f"  ⚠️ CANCEL FAILED: {error_msg}")
            return False, str(error_msg)
            
    except Exception as e:
        logger.error(f"  ❌ CANCEL ERROR: {e}")
        return False, str(e)

# ==================== HELPERS ====================
def check_order_status(order_id):
    """
    Check the status of an order by ID via Upstox API.
    Returns: status string (e.g., 'complete', 'rejected', 'open', 'cancelled') or None
    """
    if not LIVE_TRADING or not order_id or str(order_id).startswith("PAPER"):
        return "complete"

    # Use order history endpoint which is reliable for checking status
    # Use sandbox URL when SANDBOX_MODE is enabled
    if SANDBOX_MODE:
        url = "https://api-sandbox.upstox.com/v2/order/history"
    else:
        url = "https://api.upstox.com/v2/order/history"
    
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {ORDER_ACCESS_TOKEN}"
    }
    params = {"order_id": order_id}

    try:
        r = session.get(url, headers=headers, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            # The API returns a list of history for the order.
            # We want the latest status.
            order_history = data.get('data', [])
            if order_history:
                # Check for any terminal state in history, otherwise take the latest state
                # The history is usually sorted latest first? logic: iterate 
                # If any entry is 'complete', 'rejected', 'cancelled', return that.
                
                # Iterate through history to find a final state
                for entry in order_history:
                    s = str(entry.get('status') or '').lower()
                    if s in ('rejected', 'cancelled', 'complete'):
                        return s
                
                # If no terminal state found, return the status of the first (latest) item
                # This covers 'open', 'trigger pending', 'validation pending' etc.
                if order_history:
                    return str(order_history[0].get('status') or '').lower()
        else:
             logger.error(f"  ❌ Check Order Status Failed. Status: {r.status_code}, Response: {r.text}")
             
    except Exception as e:
        logger.error(f"  ❌ Check Order Status Error: {e}")
        pass
        
    return None

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
        
    def _send():
        try:
            # Using independent requests.post instead of shared Upstox session to avoid pool blocking/timeouts
            r = requests.post(DISCORD_WEBHOOK_URL, json={"embeds": [embed]}, timeout=20)
            if r.status_code in (200,204):
                logger.info("  ✅ Discord alert sent")
            else:
                logger.warning(f"  ⚠️ Discord webhook returned {r.status_code}")
        except requests.exceptions.ReadTimeout:
            logger.warning("  ⚠️ Discord webhook send timeout (ignored as background task)")
        except requests.exceptions.ConnectTimeout:
            logger.warning("  ⚠️ Discord webhook connection timeout (ignored as background task)")
        except Exception as e:
            logger.error(f"  ❌ Discord send error: {e}")
            
    # Run discord webhook asynchronously so it doesn't block the main trading loop or exit logic
    t = threading.Thread(target=_send, daemon=True)
    t.start()
    return t

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
        logger.info(f"Fetching option instruments for {current_expiry_date}...")
        r = session.get(url, headers=headers, timeout=10)
        logger.info(f"API Response Status: {r.status_code}")
        
        if r.status_code == 200:
            data = r.json()
            contracts_cache = data.get('data') or []
            logger.info(f"Fetched {len(contracts_cache)} contracts for {current_expiry_date}")
        else:
            logger.error(f"API returned status {r.status_code}: {r.text[:200]}")
            
        if not contracts_cache:
            logger.warning(f"No contracts for {current_expiry_date}, fetching all expiries...")
            url2 = f"https://api.upstox.com/v2/option/contract?instrument_key={encoded_symbol}"
            r2 = session.get(url2, headers=headers, timeout=10)
            logger.info(f"All expiries API Status: {r2.status_code}")
            
            if r2.status_code == 200:
                data2 = r2.json()
                all_contracts = data2.get('data') or []
                logger.info(f"Total contracts available: {len(all_contracts)}")
                
                if all_contracts:
                    expiries = sorted(set([c.get('expiry') for c in all_contracts if c.get('expiry')]))
                    logger.info(f"Available expiries: {expiries[:5]}")
                    if expiries:
                        nearest = expiries[0]
                        current_expiry_date = nearest
                        contracts_cache = [c for c in all_contracts if c.get('expiry') == nearest]
                        logger.info(f"Using nearest expiry: {nearest} ({len(contracts_cache)} contracts)")
            else:
                logger.error(f"All expiries API failed: {r2.status_code}: {r2.text[:200]}")
                
        if not contracts_cache:
            logger.error("No contracts found after all attempts!")
            return []
            
        spot = get_spot_price()
        if spot is not None:
            filtered = [c.get('instrument_key') for c in contracts_cache if c.get('instrument_key') and abs(c.get('strike_price', 0) - spot) <= 500]
            logger.info(f"Filtered to {len(filtered)} contracts near spot price {spot}")
            return filtered if filtered else [c.get('instrument_key') for c in contracts_cache[:50] if c.get('instrument_key')]
        else:
            logger.warning("Could not get spot price, returning first 50 contracts")
            return [c.get('instrument_key') for c in contracts_cache[:50] if c.get('instrument_key')]
    except Exception as e:
        logger.error(f"  ❌ Option instruments error: {e}", exc_info=True)
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
def calculate_atr(df, period=14):
    """
    Calculate Average True Range (ATR) for volatility-based stops.
    """
    df = df.copy()
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = np.abs(df['high'] - df['close'].shift())
    df['low_close'] = np.abs(df['low'] - df['close'].shift())
    df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['ATR'] = df['tr'].rolling(window=period).mean()
    return df

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

    # V4.1 RSI Logic (SMA)
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

    df.drop(columns=['TPV', 'Cumulative_TPV', 'Cumulative_Volume', 'high_low', 'high_close', 'low_close', 'tr'], errors='ignore', inplace=True)
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



# ==================== POSITION CLASS ====================
class Position:
    def __init__(self, signal_type, strike, entry_premium, instrument_key, timestamp, entry_order_id=None):
        self.signal_type = signal_type
        self.strike = strike
        self.entry_premium = float(entry_premium)
        self.instrument_key = instrument_key
        self.timestamp = timestamp
        self.lot_size = LOT_SIZE
        self.highest_pnl = -999999.0 # Initialize low
        self.trailing_stop_active = False
        self.trailing_stop_price = None
        self.entry_order_id = entry_order_id    # Track entry order ID
        
        # Note: No exchange SL order ID needed for software-based SL
        # But we keep the field to avoid breaking logging if referenced, 
        # though we won't use it for logic.
        self.sl_order_id = None 

    def calculate_pnl(self, current_premium):
        diff = current_premium - self.entry_premium
        pnl = diff * self.lot_size
        
        # Track highest P&L for potential stats, though trailing uses price levels mostly?
        # Wait, snippet used P&L for check but calculated trail based on premium?
        # Snippet: self.trailing_stop_price = current_premium - (TRAILING_STOP / self.lot_size)
        if pnl > self.highest_pnl: self.highest_pnl = pnl
        return pnl, diff

    def check_exit(self, current_premium):
        pnl, diff = self.calculate_pnl(current_premium)
        
        # 1. FIXED STOP LOSS
        if pnl <= -STOP_LOSS:
            return True, f"STOP LOSS (Loss: ₹{abs(pnl):.2f})", pnl, diff

        # 1.5. PROFIT LOCK (700 -> 200 reversal)
        if self.highest_pnl >= 700.0 and pnl <= 200.0:
            return True, f"PROFIT LOCK (Reached ₹700, hit ₹200 lock)", pnl, diff

        # 2. TAKE PROFIT -> ACTIVATE TRAILING
        if pnl >= TAKE_PROFIT:
            if not self.trailing_stop_active:
                self.trailing_stop_active = True
                # Initial Trail: Current Price - (Trailing Step in Points)
                # e.g. If Step is 500rs, and Lot is 65: 500/65 = 7.7 pts distance
                trail_gap_pts = TRAILING_STOP / self.lot_size
                self.trailing_stop_price = current_premium - trail_gap_pts
                logger.info(f"  🎯 Take Profit reached! Trailing stop activated @ ₹{self.trailing_stop_price:.2f}")
        
        # 3. TRAILING STOP CHECK
        if self.trailing_stop_active:
            # Check exit
            if current_premium <= self.trailing_stop_price:
                return True, f"TRAILING STOP (Profit: ₹{pnl:.2f})", pnl, diff
            
            # Update Trail
            # Logic: If price moves up, pull trail up?
            # Snippet: new_trail = current_premium - (TRAILING_STOP / self.lot_size)
            # This is a fixed distance trail from *current* price? 
            # Snippet logic: if new_trail > self.trailing_stop_price: update
            # Yes, standard trailing.
            trail_gap_pts = TRAILING_STOP / self.lot_size
            new_trail = current_premium - trail_gap_pts
            
            if new_trail > self.trailing_stop_price:
                self.trailing_stop_price = new_trail
                logger.info(f"  📈 Trailing stop updated: ₹{self.trailing_stop_price:.2f}")

        return False, None, pnl, diff

# ==================== SIGNAL LOGIC ====================
def check_signal_conditions(spot, day_open, vwap, rsi, oi_trend, oi_confirmation):
    conditions = {
        'CE': {
            'price_above_open': spot > day_open,
            'price_above_vwap': spot > vwap,
            'rsi_bullish': rsi > 60,  # V4.1 Threshold
            'oi_bullish': oi_trend == 'Bullish' and oi_confirmation.startswith('Strong')
        },
        'PE': {
            'price_below_open': spot < day_open,
            'price_below_vwap': spot < vwap,
            'rsi_bearish': rsi < 40,  # V4.1 Threshold
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
    trading_mode = "🔴 LIVE TRADING" if LIVE_TRADING else "📝 PAPER TRADING"
    sandbox_status = "🧪 SANDBOX MODE ENABLED" if SANDBOX_MODE else ""
    if SANDBOX_MODE:
        logger.info(f"\n{'=' * 85}")
        logger.info("🧪 SANDBOX MODE ENABLED - Using test environment (V3 API)")
        logger.info(f"{'=' * 85}\n")
    banner = f"""
{'=' * 85}
🚀 NIFTY 50 OPTIONS INTRADAY TRADING BOT - V5.0 LIVE EXECUTION
{'=' * 85}
Mode:        {trading_mode}{' | ' + sandbox_status if sandbox_status else ''}
Strategy:    Day's Open + VWAP + RSI + OI + OI-Delta Confirmation
Timeframe:   5-Minute Candles (1-min resampled)
Data Source: Live from NSE via Upstox API
Session:     {session_time}
Trade Log:   {CSV_FILE}
Terminal Log: {TERMINAL_LOG_FILE}
Expiry:      {current_expiry_date}
Lot Size:    {LOT_SIZE} quantity
Desc:        Fixed Risk Management
Stop Loss:   ₹{STOP_LOSS} (Exchange Order)
Target:      ₹{TAKE_PROFIT} (Activates Trailing)
Trailing:    ₹{TRAILING_STOP} Step
Polling:     Signal Check: {SIGNAL_CHECK_INTERVAL}s | Position Monitor: {POSITION_MONITOR_INTERVAL}s
Protection:  Exchange SL Order

✅ NEW: Live Order Execution via Upstox API
✅ NEW: Exchange Stop Loss Protection
✅ NEW: One trade per day limit
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

# ==================== FETCH LIVE SPOT CANDLES (Historical + Intraday) ====================
def fetch_live_spot_candles(symbol):
    encoded_symbol = encode_symbol(symbol)
    headers = {"accept": "application/json", "Authorization": f"Bearer {ACCESS_TOKEN}"}
    
    # 1. Fetch 5 Days History (Excluding Today)
    now = get_now_kolkata()
    today_date = now.strftime('%Y-%m-%d')
    yesterday_date = (now - dt.timedelta(days=1)).strftime('%Y-%m-%d')
    from_date = (now - dt.timedelta(days=6)).strftime('%Y-%m-%d') # 5 days back from yesterday
    
    history_url = f"https://api.upstox.com/v2/historical-candle/{encoded_symbol}/1minute/{yesterday_date}/{from_date}"
    
    # 2. Fetch Today's Live Intraday Data
    intraday_url = f"https://api.upstox.com/v2/historical-candle/intraday/{encoded_symbol}/1minute"
    
    all_candles = []
    
    try:
        # Fetch History
        r_hist = session.get(history_url, headers=headers, timeout=10)
        if r_hist.status_code == 200:
            hist_data = r_hist.json()
            hist_candles = hist_data.get('data', {}).get('candles', [])
            if hist_candles:
                all_candles.extend(hist_candles)
        else:
             logger.warning(f"  ⚠️ Historical data fetch failed: {r_hist.status_code}")

        # Fetch Intraday (Live)
        r_intra = session.get(intraday_url, headers=headers, timeout=10)
        if r_intra.status_code == 200:
            intra_data = r_intra.json()
            intra_candles = intra_data.get('data', {}).get('candles', [])
            if intra_candles:
                all_candles.extend(intra_candles)
        else:
             logger.error(f"  ❌ Intraday data fetch failed: {r_intra.status_code}")
             
        if not all_candles:
            return None
            
        # Parse and DataFrame
        df = pd.DataFrame(all_candles, columns=["time","open","high","low","close","volume","oi"])
        df["time"] = pd.to_datetime(df["time"])
        df["volume"] = df["volume"].replace(0, 1)
        
        # Remove duplicates (if any overlap) and Sort
        df = df.drop_duplicates(subset=['time'])
        df = df.sort_values("time").reset_index(drop=True)
        df.set_index("time", inplace=True)
        
        # Resample to 5-min
        df_5min = df.resample('5min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        df_5min.reset_index(inplace=True)
        
        latest_time = df_5min.iloc[-1]['time'] if not df_5min.empty else "N/A"
        # logger.info(f"  ✅ Fetched {len(all_candles)} TOTAL 1-min → {len(df_5min)} 5-min candles (Latest: {latest_time})")
        return df_5min
        
    except Exception as e:
        logger.error(f"  ❌ Candle fetch error: {e}")
        return None

# ==================== MAIN LOOP ====================
def main():
    global last_signal_time, open_position, days_open_cache, current_expiry_date, contracts_cache, trade_completed_today

    with open(CSV_FILE, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Time", "Signal", "Strike", "Premium",
            "Spot", "RSI", "VWAP", "Day_Open", "OI_Trend",
            "Exit_Reason", "PnL", "Premium_Diff"
        ])

    logger.info("CSV file initialized: " + CSV_FILE)
    logger.info("\n📥 Initializing...")
    
    # Instruments already loaded in global scope (from __main__ block)
    logger.info(f"✅ Loaded {len(contracts_cache)} instruments (from cache)")
    current_expiry_date = current_expiry_date or get_next_weekly_expiry()
    print_startup_banner()

    iteration = 0
    try:
        while True:
            iteration += 1
            now = get_now_kolkata()

            # If not in an open position, log the Iteration banner to terminal & file
            if not open_position:
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
                        
                        # ===== CANCEL SL ORDER FIRST =====
                        if open_position.sl_order_id:
                            cancel_success, cancel_msg = cancel_order(open_position.sl_order_id)
                            if cancel_success:
                                logger.info(f"  🚫 SL ORDER CANCELLED (market close)")
                        # =================================
                        
                        # ===== LIVE ORDER: EXIT AT MARKET CLOSE =====
                        exit_success, exit_order_id, exit_msg = exit_position(open_position)
                        if exit_success:
                            open_position.exit_order_id = exit_order_id
                            logger.info(f"  ✅ EXIT ORDER PLACED: {exit_order_id}")
                        else:
                            logger.error(f"  ❌ EXIT ORDER FAILED: {exit_msg}")
                            
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
                                {"name": "P&L", "value": f"₹{pnl:.2f}", "inline": False},
                                {"name": "Order ID", "value": str(exit_order_id or "N/A"), "inline": True}
                            ]
                        )
                        open_position = None

                days_open_cache = None
                trade_completed_today = False  # Reset for next day
                time.sleep(60)
                continue

            # If position open - manage with fast 1-second polling
            if open_position:
                current_premium = get_current_premium(open_position.instrument_key)
                if current_premium:
                    pnl, premium_diff = open_position.calculate_pnl(current_premium)

                    # Compact single-line status for 1-second updates
                    trail_info = f" | Trail: ₹{open_position.trailing_stop_price:.2f}" if open_position.trailing_stop_active else ""
                    current_time_str = now.strftime('%I:%M:%S %p')
                    
                    # Print to terminal with carriage return (updates in place)
                    print(f"\r   ⏰ [{current_time_str}] ⚡ LTP: ₹{current_premium:.2f} | P&L: ₹{pnl:+.2f}{trail_info}    ", end="", flush=True)
                    
                    # Log to file normally line-by-line (bypasses terminal stream handler)
                    logger.info(f"   ⏰ [{current_time_str}] ⚡ LTP: ₹{current_premium:.2f} | P&L: ₹{pnl:+.2f}{trail_info}", extra={'file_only': True})

                    should_exit, exit_reason, final_pnl, final_premium_diff = open_position.check_exit(current_premium)

                    if should_exit:
                        print()  # New line after compact updates
                        timestamp = now.strftime('%Y-%m-%d %I:%M:%S %p')

                        logger.info("\n" + "="*85)
                        logger.info(f"🔔 POSITION CLOSED: {exit_reason}")
                        logger.info("="*85)
                        
                        # ===== CANCEL SL ORDER IF EXISTS (for trailing/target exit) =====
                        if open_position.sl_order_id and "STOP LOSS" not in exit_reason:
                            # If we hit target or trailing stop, we must cancel the exchange SL
                            # If we hit SL, the order is already executed (or we are market exiting)
                            # Actually, if "STOP LOSS" is in reason, it might be SOFTWARE SL hit before exchange?
                            # If exchange SL hit, the position manager loop might not even know yet until next poll?
                            # But here we are EXPLICITLY closing via software condition.
                            # So yes, cancel SL.
                            logger.info(f"  🚫 Cancelling SL order: {open_position.sl_order_id}...")
                            cancel_success, cancel_msg = cancel_order(open_position.sl_order_id)
                            if cancel_success:
                                logger.info(f"  ✅ SL ORDER CANCELLED")
                            else:
                                logger.warning(f"  ⚠️ SL Cancel failed: {cancel_msg}")
                        # ============================================================
                        
                        logger.info(f"  Entry:       ₹{open_position.entry_premium:.2f}")
                        # ===== LIVE ORDER: EXIT AT SL/TP/TRAILING =====
                        # Only place exit order if SL didn't trigger on exchange
                        if "STOP LOSS" not in exit_reason or not open_position.sl_order_id:
                            exit_success, exit_order_id, exit_msg = exit_position(open_position)
                            if exit_success:
                                open_position.exit_order_id = exit_order_id
                                logger.info(f"  ✅ EXIT ORDER PLACED: {exit_order_id}")
                            else:
                                logger.error(f"  ❌ EXIT ORDER FAILED: {exit_msg}")
                        else:
                            # SL triggered on exchange - VERIFY IT WAS FILLED
                            # We can't just assume it filled because price touched trigger
                            sl_status = check_order_status(open_position.sl_order_id)
                            logger.info(f"  🔍 SL Triggered - Verifying Order Status: {sl_status}")
                            
                            if sl_status == 'complete':
                                logger.info(f"  ✅ SL ORDER CONFIRMED FILLED ON EXCHANGE")
                                exit_order_id = open_position.sl_order_id
                            else:
                                logger.warning(f"  ⚠️ SL HIT PRICE but Order Status is '{sl_status}' (Not Filled)")
                                logger.warning(f"  🔄 FORCE EXITING at Market to ensure protection...")
                                
                                # Cancel the pending SL order first
                                cancel_success, _ = cancel_order(open_position.sl_order_id)
                                if cancel_success:
                                    logger.info(f"  🚫 Pending SL Order Cancelled")
                                    
                                # Place Market Exit
                                exit_success, exit_order_id, exit_msg = exit_position(open_position)
                                if exit_success:
                                    open_position.exit_order_id = exit_order_id
                                    logger.info(f"  ✅ FORCE EXIT ORDER PLACED: {exit_order_id}")
                                    exit_reason += " (FORCE EXIT)"
                                else:
                                    logger.error(f"  ❌ FORCE EXIT ORDER FAILED: {exit_msg}")
                        # ==============================================
                        
                        logger.info(f"  Entry:       ₹{open_position.entry_premium:.2f}")
                        logger.info(f"  Exit:        ₹{current_premium:.2f}")
                        logger.info(f"  Premium Diff: ₹{final_premium_diff:.2f}")
                        logger.info(f"  Total P&L:   ₹{final_pnl:.2f} (₹{final_premium_diff:.2f} × {LOT_SIZE})")
                        logger.info("=" * 85)

                        log_trade_to_csv(timestamp, f"EXIT {open_position.signal_type}", open_position.strike,
                                       current_premium, 0, 0, 0, 0, "", exit_reason, final_pnl, final_premium_diff)

                        discord_thread = send_discord_alert(
                            f"🔔 {exit_reason}",
                            f"**{open_position.signal_type}** | Strike: {open_position.strike}\nReason: {exit_reason}",
                            0xff0000 if final_pnl < 0 else 0x00ff00,
                            [
                                {"name": "Entry", "value": f"₹{open_position.entry_premium:.2f}", "inline": True},
                                {"name": "Exit", "value": f"₹{current_premium:.2f}", "inline": True},
                                {"name": "P&L", "value": f"₹{final_pnl:.2f}", "inline": False},
                                {"name": "Order ID", "value": str(exit_order_id or "N/A"), "inline": True}
                            ]
                        )

                        open_position = None
                        last_signal_time = now
                        trade_completed_today = True  # One trade per day - done for today
                        logger.info("  📅 Daily trade limit reached - No more trades today")
                        logger.info("  ⏹️  MISSION ACCOMPLISHED: Trade Completed. Exiting Bot.")
                        
                        # Ensure the final discord alert has time to send before exiting
                        if discord_thread:
                            discord_thread.join(timeout=5.0)
                            
                        sys.exit(0)

                time.sleep(POSITION_MONITOR_INTERVAL)
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

            # Check if daily trade limit reached
            if trade_completed_today:
                logger.info("\n📅 DAILY TRADE COMPLETED - Monitoring only, no new trades today")
                # Sync to next minute
                sleep_sec = 60 - get_now_kolkata().second
                time.sleep(sleep_sec)
                continue

            if last_signal_time:
                elapsed = (now - last_signal_time).total_seconds()
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
                    # ===== PREMIUM FILTER REMOVED (User Request: Pure ATR) =====
                    # Old logic: if premium < MIN_PREMIUM: continue (REMOVED)
                    # ========================================

                    timestamp = now.strftime('%Y-%m-%d %I:%M:%S %p')

                    print_trade_alert(timestamp, signal, strike, premium, spot)
                    
                    # ===== LIVE ORDER: ENTRY BUY =====
                    order_success, order_id, order_msg = place_order(
                        instrument_key=instrument_key,
                        transaction_type="BUY",
                        quantity=LOT_SIZE,
                        order_tag=f"ENTRY_{signal.replace(' ', '_')}"
                    )
                    
                    if not order_success:
                        logger.error(f"  ❌ ENTRY ORDER FAILED: {order_msg}")
                        logger.error("  🛑 CRITICAL: Order placement failed")
                        logger.error("  🚫 BOT STOPPED - Manual intervention required")
                        logger.error("  💡 Please check: 1) Sufficient funds 2) Valid token 3) Market hours")
                        send_discord_alert(
                            "🛑 Bot Stopped - Order Failed",
                            f"Entry order failed: {order_msg}\n\nBot has been stopped. Please fix the issue and restart manually.",
                            0xff0000
                        )
                        logger.info("\n" + "="*85)
                        logger.info("⏹️  BOT STOPPED DUE TO ORDER FAILURE")
                        logger.info("="*85)
                        sys.exit(1)  # Exit with error code
                    
                    logger.info(f"  ✅ ENTRY ORDER PLACED: {order_id}")
                    # ==================================
                    
                    # Wait for BUY order to fill before placing SL
                    logger.info("  ⏳ Waiting for BUY order to COMPLETE...")
                    
                    entry_status = None
                    order_confirmed = False # Track if we proceed or stop
                    
                    # Wait loop (Extended to 60s for EXTREME Sandbox Lag)
                    for i in range(1, 61):
                        entry_status = check_order_status(order_id)
                        
                        # STRICT: Only ACCEPT 'complete' as success (Except Sandbox)
                        if entry_status == 'complete':
                            logger.info(f"  ✅ Entry Confirmed ({entry_status})")
                            order_confirmed = True
                            break
                        
                        # SANDBOX EXCEPTION: Accept 'put order req received' after 5s to allow testing
                        if SANDBOX_MODE and entry_status in ['open', 'put order req received'] and i > 5:
                             logger.info(f"  ✅ Entry Confirmed ({entry_status}) [Sandbox Mode]")
                             order_confirmed = True
                             break
                             
                        elif entry_status in ['rejected', 'cancelled']:
                            logger.error(f"  ❌ Entry Failed ({entry_status})")
                            break
                        elif entry_status in ['open', 'put order req received', 'validation pending']:
                            if i % 5 == 0: # Log every 5 seconds
                                logger.info(f"  ⏳ Entry status: {entry_status} ... waiting ({i}/60s)")
                        else:
                            logger.warning(f"  ⚠️ Unknown Status: {entry_status}")
                        
                        time.sleep(1)
                    
                    # CRITICAL SAFETY: Proceed ONLY if order is COMPLETE (or Accepted in Sandbox)
                    if not order_confirmed:
                        logger.error(f"  ❌ Entry order status is '{entry_status}' (NOT complete) after wait.")
                        logger.error(f"  🚨 CRITICAL: Buy order NOT filled. ABORTING SL PLACEMENT.")
                        logger.error(f"  💡 If order is 'open'/'pending', it might fill later. Please check App manually!")
                        
                        # Auto-cancel if still pending/open to avoid late fill without SL
                        if entry_status in ['open', 'put order req received', 'validation pending']:
                            logger.info(f"  🚫 Attempting to cancel STUCK '{entry_status}' entry order {order_id}...")
                            cancel_result, cancel_msg_out = cancel_order(order_id)
                            if cancel_result:
                                logger.info(f"  ✅ Stuck Entry Order Cancelled Successfully.")
                            else:
                                logger.error(f"  ❌ Failed to cancel stuck entry order: {cancel_msg_out}")

                        send_discord_alert(
                            "🚨 Bot Stopped - Entry Stuck",
                            f"Order {order_id} status: {entry_status}\n\nTerminating to prevent naked position.",
                            0xff0000
                        )
                        logger.info("\n" + "="*85)
                        logger.info("⏹️  BOT STOPPED FOR SAFETY")
                        logger.info("="*85)
                        sys.exit(1)

                            
                    # Create Position object (Fixed Risk)
                    open_position = Position(signal, strike, premium, instrument_key, timestamp, entry_order_id=order_id)
                    
                    logger.info(f"  ✅ BUY position confirmed. Entry Price: {premium}")

                    # ===== LIVE ORDER: PLACE SL ORDER ON EXCHANGE =====
                    # Calculate SL trigger based on FIXED RISK (STOP_LOSS = 1500)
                    # SL Amount per unit = 1500 / 65 = ~23.07 pts
                    sl_points = STOP_LOSS / LOT_SIZE
                    sl_trigger = premium - sl_points
                    
                    logger.info(f"  🎯 SL Strategy: Fixed Risk ₹{STOP_LOSS} -> {sl_points:.2f} pts")

                    # CRITICAL: Ensure SL Trigger is valid (> 0.05) and positive
                    if sl_trigger <= 0:
                        logger.warning(f"  ⚠️ SL trigger calculated as ₹{sl_trigger:.2f} (<= 0). Adjusting to minimum ₹0.05")
                        sl_trigger = 0.05
                        
                    # Place the SL order
                    sl_success, sl_order_id, sl_msg = place_sl_order(
                        instrument_key=instrument_key,
                        quantity=LOT_SIZE,
                        trigger_price=sl_trigger,
                        order_tag=f"SL_{signal.replace(' ', '_')}"
                    )
                        
                    if sl_success:
                        open_position.sl_order_id = sl_order_id
                        logger.info(f"  🛡️ SL ORDER ACTIVE on Exchange: Trigger @ ₹{sl_trigger:.2f}")
                    else:
                        logger.warning(f"  ⚠️ SL ORDER FAILED: {sl_msg} - Bot will monitor manually")
                        # Fallback to software monitoring if exchange SL fails
                    # ==================================================


                    log_trade_to_csv(timestamp, signal, strike, premium, spot, rsi, vwap, day_open, oi_trend)

                    send_discord_alert(
                        f"🚀 NEW SIGNAL - {signal}",
                        f"Strike: {strike} | Lot: {LOT_SIZE}",
                        0x00ff00,
                        [
                            {"name": "Premium", "value": f"₹{premium:.2f}", "inline": True},
                            {"name": "Spot", "value": f"{spot:.2f}", "inline": True},
                            {"name": "Investment", "value": f"₹{premium * LOT_SIZE:.2f}", "inline": True},
                            {"name": "Order ID", "value": str(order_id), "inline": True},
                            {"name": "SL Trigger", "value": f"₹{sl_trigger:.2f}", "inline": True}
                        ]
                    )

                    last_signal_time = now
                else:
                    logger.warning("\n⚠️  Signal generated but strike/premium unavailable")
            else:
                logger.info("\n⏸  NO SIGNAL - Waiting for all conditions to align...")

            # logger.info(f"\n⏱  Next check in {SIGNAL_CHECK_INTERVAL} seconds...")
            # Skip long sleep if we just entered a position
            if open_position:
                continue
                
            # Sync to next minute (00 seconds) for clean candle alignment
            sleep_sec = 60 - get_now_kolkata().second
            # logger.info(f"⏳ Waiting {sleep_sec}s for next minute candle...")
            time.sleep(sleep_sec)

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
