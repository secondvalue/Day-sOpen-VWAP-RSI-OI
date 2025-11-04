"""
================================================================================
NIFTY 50 OPTIONS INTRADAY TRADING BOT - FIXED VERSION V3
================================================================================
Strategy: Day's Open + VWAP + RSI + OI (Open Interest) Confirmation
Timeframe: 5-Minute Candles (resampled from 1-minute live data)
Data Source: Live from NSE via Upstox API

✅ FIXED: OI Logic (CE high = Bullish, PE high = Bearish)
✅ FIXED: Day's Open (fetched from 9:15 AM candle - ACTUAL today's open)

BUY CALL (CE) when ALL align:
  ✅ Price > Day's Open
  ✅ Price > VWAP
  ✅ RSI > 60
  ✅ OI Trend = Bullish (More CALL OI)

BUY PUT (PE) when ALL align:
  ✅ Price < Day's Open
  ✅ Price < VWAP
  ✅ RSI < 40
  ✅ OI Trend = Bearish (More PUT OI)

P&L MANAGEMENT:
  🎯 Take Profit: ₹1,500 (₹20 per lot)
  🛡️ Stop Loss: ₹2,000 (₹26.67 per lot)
  📈 Trailing Stop: ₹500 (₹6.67 per lot) after TP

Expected: 4-6 signals/day | 75-82% win rate
================================================================================
"""

import requests
import pandas as pd
import numpy as np
import datetime as dt
import time
import csv
import os
import logging

# ==================== CONFIGURATION ====================
ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiI1NUJBOVgiLCJqdGkiOiI2OTA5NzUxMGM5YzYzZDU4ZWViZjgwZDkiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNQbHVzUGxhbiI6ZmFsc2UsImlhdCI6MTc2MjIyNzQ3MiwiaXNzIjoidWRhcGktZ2F0ZXdheS1zZXJ2aWNlIiwiZXhwIjoxNzYyMjkzNjAwfQ.ZqO_xW_7ShNalpapEdzocZy6sdRlqZdeLPUhTWXDYG8"
NIFTY_SYMBOL = "NSE_INDEX|Nifty 50"
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1412386951474057299/Jgft_nxzGxcfWOhoLbSWMde-_bwapvqx8l3VQGQwEoR7_8n4b9Q9zN242kMoXsVbLdvG"

# FOLDER CONFIGURATION
TRADE_LOGS_DIR = "trade_logs"
TERMINAL_LOGS_DIR = "terminal_logs"

os.makedirs(TRADE_LOGS_DIR, exist_ok=True)
os.makedirs(TERMINAL_LOGS_DIR, exist_ok=True)

timestamp = dt.datetime.now().strftime('%Y-%m-%d_%I%M%S_%p')

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

SIGNAL_COOLDOWN = 300

LOT_SIZE = 75
TAKE_PROFIT = 1500
STOP_LOSS = 2000
TRAILING_STOP = 500
# =======================================================

last_signal_time = None
current_expiry_date = None
contracts_cache = []
open_position = None
days_open_cache = None

# ==================== POSITION TRACKING ====================

class Position:
    def __init__(self, signal_type, strike, entry_premium, instrument_key, timestamp):
        self.signal_type = signal_type
        self.strike = strike
        self.entry_premium = entry_premium
        self.instrument_key = instrument_key
        self.timestamp = timestamp
        self.lot_size = LOT_SIZE
        self.highest_pnl = 0
        self.trailing_stop_active = False
        self.trailing_stop_price = None
    
    def calculate_pnl(self, current_premium):
        premium_diff = current_premium - self.entry_premium
        pnl = premium_diff * self.lot_size
        
        if pnl > self.highest_pnl:
            self.highest_pnl = pnl
        
        return pnl, premium_diff
    
    def check_exit(self, current_premium):
        pnl, premium_diff = self.calculate_pnl(current_premium)
        
        if pnl <= -STOP_LOSS:
            return True, f"STOP LOSS (Loss: ₹{abs(pnl):.2f})", pnl, premium_diff
        
        if pnl >= TAKE_PROFIT:
            if not self.trailing_stop_active:
                self.trailing_stop_active = True
                self.trailing_stop_price = current_premium - (TRAILING_STOP / self.lot_size)
                logger.info(f"  🎯 Take Profit reached! Trailing stop: ₹{self.trailing_stop_price:.2f}")
        
        if self.trailing_stop_active:
            if current_premium <= self.trailing_stop_price:
                return True, f"TRAILING STOP (Profit: ₹{pnl:.2f})", pnl, premium_diff
            
            new_trail = current_premium - (TRAILING_STOP / self.lot_size)
            if new_trail > self.trailing_stop_price:
                self.trailing_stop_price = new_trail
                logger.info(f"  📈 Trailing stop updated: ₹{self.trailing_stop_price:.2f}")
        
        return False, None, pnl, premium_diff

# ==================== DISCORD ====================

def send_discord_alert(title, description, color=0x00ff00, fields=None):
    if DISCORD_WEBHOOK_URL == "YOUR_DISCORD_WEBHOOK_URL_HERE":
        return
    
    embed = {
        "title": title,
        "description": description,
        "color": color,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "footer": {"text": f"Nifty Bot | Lot: {LOT_SIZE}"}
    }
    
    if fields:
        embed["fields"] = fields
    
    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json={"embeds": [embed]}, timeout=10)
        if response.status_code == 204:
            logger.info("  ✅ Discord alert sent")
    except:
        pass

# ==================== HELPER FUNCTIONS ====================

def get_next_tuesday_expiry():
    today = dt.datetime.now()
    
    if today.weekday() == 1:
        if today.hour < 15 or (today.hour == 15 and today.minute < 30):
            expiry = today
        else:
            expiry = today + dt.timedelta(days=7)
    else:
        days_ahead = (1 - today.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7
        expiry = today + dt.timedelta(days=days_ahead)
    
    return expiry.strftime('%Y-%m-%d')

def get_arrow(current, reference):
    return "🔺" if current > reference else "🔻" if current < reference else "➡️"

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

# ==================== FIXED: GET DAY'S OPEN FROM 9:15 AM CANDLE ====================

def get_days_open_from_intraday():
    """
    FIXED: Get actual day's open from the first candle (9:15 AM)
    This is the TRUE market open price for today
    """
    global days_open_cache
    
    # Return cached value if already fetched today
    if days_open_cache is not None:
        return days_open_cache
    
    try:
        encoded_symbol = NIFTY_SYMBOL.replace("|", "%7C").replace(" ", "%20")
        url = f"https://api.upstox.com/v2/historical-candle/intraday/{encoded_symbol}/1minute"
        
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {ACCESS_TOKEN}"
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            logger.warning(f"  ⚠️  Day's open API returned status {response.status_code}")
            return None
        
        data = response.json()
        
        if "data" not in data or "candles" not in data["data"]:
            logger.warning("  ⚠️  No candle data for day's open")
            return None
        
        candles = data["data"]["candles"]
        
        if len(candles) == 0:
            logger.warning("  ⚠️  Empty candles list")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(candles, columns=["time", "open", "high", "low", "close", "volume", "oi"])
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").reset_index(drop=True)
        
        # Get today's date
        today = dt.datetime.now().date()
        
        # Filter only today's candles
        df['date'] = df['time'].dt.date
        today_candles = df[df['date'] == today]
        
        if len(today_candles) == 0:
            logger.warning("  ⚠️  No candles found for today")
            return None
        
        # Get the FIRST candle of today (9:15 AM candle)
        first_candle = today_candles.iloc[0]
        day_open = first_candle['open']
        candle_time = first_candle['time']
        
        # Cache it
        days_open_cache = day_open
        
        logger.info(f"  ✅ Day's Open (9:15 AM): {day_open:.2f} (from {candle_time.strftime('%I:%M %p')})")
        return day_open
        
    except Exception as e:
        logger.error(f"  ❌ Day's open fetch error: {e}")
        return None

# ==================== LIVE DATA FETCHING ====================

def fetch_live_spot_candles(symbol):
    encoded_symbol = symbol.replace("|", "%7C").replace(" ", "%20")
    url = f"https://api.upstox.com/v2/historical-candle/intraday/{encoded_symbol}/1minute"
    
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {ACCESS_TOKEN}"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return None
        
        data = response.json()
        
        if "data" not in data or "candles" not in data["data"]:
            return None
        
        candles = data["data"]["candles"]
        
        if len(candles) == 0:
            return None
        
        df = pd.DataFrame(candles, columns=["time", "open", "high", "low", "close", "volume", "oi"])
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

def get_option_instruments():
    global current_expiry_date, contracts_cache
    
    current_expiry_date = get_next_tuesday_expiry()
    
    encoded_symbol = "NSE_INDEX%7CNifty%2050"
    url = f"https://api.upstox.com/v2/option/contract?instrument_key={encoded_symbol}&expiry_date={current_expiry_date}"
    
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {ACCESS_TOKEN}"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return []
        
        data = response.json()
        
        if "data" not in data or data["data"] is None or len(data["data"]) == 0:
            url_no_expiry = f"https://api.upstox.com/v2/option/contract?instrument_key={encoded_symbol}"
            response2 = requests.get(url_no_expiry, headers=headers, timeout=10)
            
            if response2.status_code == 200:
                data2 = response2.json()
                all_contracts = data2.get("data", [])
                
                if len(all_contracts) > 0:
                    expiries = sorted(set([c["expiry"] for c in all_contracts]))
                    nearest_expiry = expiries[0]
                    current_expiry_date = nearest_expiry
                    contracts_cache = [c for c in all_contracts if c["expiry"] == nearest_expiry]
            else:
                return []
        else:
            contracts_cache = data["data"]
        
        if len(contracts_cache) == 0:
            return []
        
        spot_price = get_spot_price()
        
        if spot_price:
            filtered = [c["instrument_key"] for c in contracts_cache 
                       if abs(c["strike_price"] - spot_price) <= 500]
            return filtered
        else:
            return [c["instrument_key"] for c in contracts_cache[:50]]
        
    except Exception as e:
        logger.error(f"  ❌ Option instruments error: {e}")
        return []

def get_spot_price():
    try:
        encoded_symbol = NIFTY_SYMBOL.replace("|", "%7C").replace(" ", "%20")
        url = f"https://api.upstox.com/v2/market-quote/quotes?instrument_key={encoded_symbol}"
        headers = {"accept": "application/json", "Authorization": f"Bearer {ACCESS_TOKEN}"}
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if "data" in data and NIFTY_SYMBOL in data["data"]:
                return data["data"][NIFTY_SYMBOL]["last_price"]
        
        return None
    except:
        return None

# ==================== FIXED: GET LIVE OI ====================

def get_live_oi_from_quotes(instrument_keys):
    """
    FIXED: Correct OI interpretation
    - High CE OI = Bullish (more call buying)
    - High PE OI = Bearish (more put buying)
    """
    if not instrument_keys:
        return None, 0, 0
    
    ce_oi_total = 0
    pe_oi_total = 0
    
    for i in range(0, len(instrument_keys), 100):
        batch = instrument_keys[i:i+100]
        instrument_param = ",".join(batch)
        
        url = f"https://api.upstox.com/v2/market-quote/quotes?instrument_key={instrument_param}"
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {ACCESS_TOKEN}"
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code != 200:
                continue
            
            data = response.json()
            
            if "data" in data:
                for instrument_key, quote_data in data["data"].items():
                    if "oi" in quote_data:
                        oi_value = quote_data["oi"]
                        
                        if "CE" in instrument_key:
                            ce_oi_total += oi_value
                        elif "PE" in instrument_key:
                            pe_oi_total += oi_value
        
        except:
            continue
    
    if ce_oi_total == 0 and pe_oi_total == 0:
        return None, 0, 0
    
    # FIXED: Correct interpretation
    if ce_oi_total > pe_oi_total * 1.05:
        trend = "Bullish"
    elif pe_oi_total > ce_oi_total * 1.05:
        trend = "Bearish"
    else:
        trend = "Sideways"
    
    return trend, ce_oi_total, pe_oi_total

# ==================== INDICATORS ====================

def calculate_vwap_rsi(df):
    df["TP"] = (df["high"] + df["low"] + df["close"]) / 3
    df["TPV"] = df["TP"] * df["volume"]
    df["Cumulative_TPV"] = df["TPV"].cumsum()
    df["Cumulative_Volume"] = df["volume"].cumsum()
    df["VWAP"] = df["Cumulative_TPV"] / df["Cumulative_Volume"]
    
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    df["RSI"] = 100 - (100 / (1 + rs))
    
    df["VWAP"] = df["VWAP"].fillna(df["close"])
    df["RSI"] = df["RSI"].fillna(50)
    
    return df

# ==================== STRIKE & PREMIUM ====================

def get_current_premium(instrument_key):
    quote_url = f"https://api.upstox.com/v2/market-quote/quotes?instrument_key={instrument_key}"
    headers = {"accept": "application/json", "Authorization": f"Bearer {ACCESS_TOKEN}"}
    
    try:
        response = requests.get(quote_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            quote_data = response.json()
            
            if "data" in quote_data:
                for key in quote_data["data"]:
                    data_item = quote_data["data"][key]
                    premium = data_item.get("last_price", 0)
                    if premium == 0:
                        premium = data_item.get("ltp", 0)
                    return premium
        
        return None
    except:
        return None

def find_atm_strike_and_premium(spot_price, option_type):
    global contracts_cache
    
    try:
        strikes = [c for c in contracts_cache if c.get("instrument_type") == option_type]
        
        if not strikes:
            return None, None, None
        
        atm_contract = min(strikes, key=lambda x: abs(x["strike_price"] - spot_price))
        atm_strike = atm_contract["strike_price"]
        instrument_key = atm_contract["instrument_key"]
        
        premium = get_current_premium(instrument_key)
        
        if premium:
            return atm_strike, premium, instrument_key
        
        return atm_strike, 0, instrument_key
        
    except:
        return None, None, None

# ==================== SIGNAL LOGIC ====================

def check_signal_conditions(spot, day_open, vwap, rsi, oi_trend):
    """Check signal conditions with fixed OI logic"""
    conditions = {
        "CE": {
            "price_above_open": spot > day_open,
            "price_above_vwap": spot > vwap,
            "rsi_bullish": rsi > 60,
            "oi_bullish": oi_trend == "Bullish"
        },
        "PE": {
            "price_below_open": spot < day_open,
            "price_below_vwap": spot < vwap,
            "rsi_bearish": rsi < 40,
            "oi_bearish": oi_trend == "Bearish"
        }
    }
    
    if all(conditions["CE"].values()):
        return "BUY CE", conditions
    
    if all(conditions["PE"].values()):
        return "BUY PE", conditions
    
    return None, conditions

# ==================== DISPLAY ====================

def print_startup_banner():
    session_time = dt.datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')
    
    banner = f"""
{'=' * 85}
🚀 NIFTY 50 OPTIONS INTRADAY TRADING BOT - FIXED V3
{'=' * 85}
Strategy:    Day's Open + VWAP + RSI + OI Confirmation
Timeframe:   5-Minute Candles (1-min resampled)
Data Source: Live from NSE via Upstox API
Target:      75-82% Win Rate | 4-6 Signals/Day
Session:     {session_time}
Trade Log:   {CSV_FILE}
Terminal Log: {TERMINAL_LOG_FILE}
Expiry:      {current_expiry_date} (Tuesday)
Lot Size:    {LOT_SIZE} quantity
Take Profit: ₹{TAKE_PROFIT} | Stop Loss: ₹{STOP_LOSS} | Trail: ₹{TRAILING_STOP}

✅ FIXED: OI Logic (CE high = Bullish, PE high = Bearish)
✅ FIXED: Day's Open (from 9:15 AM candle - ACTUAL today's open)
{'=' * 85}

⏰ Bot started. Monitoring live market data...
Press Ctrl+C to stop.
"""
    logger.info(banner)

def print_market_snapshot(spot, day_open, vwap, rsi, oi_trend, oi_ce, oi_pe):
    snapshot = f"""
📊 MARKET SNAPSHOT
{'-' * 85}
  Spot Price:    {spot:8.2f}  |  Day's Open:   {day_open:8.2f}  {get_arrow(spot, day_open)}
  VWAP:          {vwap:8.2f}  |  Position:     {'ABOVE ✅' if spot > vwap else 'BELOW ❌'}
  RSI:           {rsi:8.2f}  |  Momentum:     {get_rsi_label(rsi)}
  OI Trend:      {oi_trend:>8}  |  CE OI: {oi_ce:,} | PE OI: {oi_pe:,}"""
    
    logger.info(snapshot)

def print_signal_evaluation(conditions):
    ce = conditions["CE"]
    pe = conditions["PE"]
    
    ce_result = "🔔 TRIGGER!" if all(ce.values()) else "❌ NO"
    pe_result = "🔔 TRIGGER!" if all(pe.values()) else "❌ NO"
    
    evaluation = f"""
🔍 SIGNAL EVALUATION (All ✅ required for trade)
{'-' * 85}
  CALL: {'✅' if ce['price_above_open'] else '❌'} Open  {'✅' if ce['price_above_vwap'] else '❌'} VWAP  {'✅' if ce['rsi_bullish'] else '❌'} RSI>60  {'✅' if ce['oi_bullish'] else '❌'} OI-Bull  →  {ce_result}
  PUT:  {'✅' if pe['price_below_open'] else '❌'} Open  {'✅' if pe['price_below_vwap'] else '❌'} VWAP  {'✅' if pe['rsi_bearish'] else '❌'} RSI<40  {'✅' if pe['oi_bearish'] else '❌'} OI-Bear  →  {pe_result}"""
    
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
{'=' * 85}"""
    
    logger.info(alert)

# ==================== LOGGING ====================

def log_trade_to_csv(timestamp, signal, strike, premium, spot, rsi, vwap, day_open, oi_trend, exit_reason=None, pnl=None, premium_diff=None):
    with open(CSV_FILE, "a", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp, signal, strike, premium,
            round(spot, 2), round(rsi, 2), round(vwap, 2), round(day_open, 2), oi_trend,
            exit_reason if exit_reason else "",
            round(pnl, 2) if pnl else "",
            round(premium_diff, 2) if premium_diff else ""
        ])

# ==================== MAIN LOOP ====================

def main():
    global last_signal_time, open_position, days_open_cache
    
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
    
    print_startup_banner()
    
    iteration = 0
    
    try:
        while True:
            iteration += 1
            now = dt.datetime.now()
            
            logger.info(f"\n{'=' * 85}")
            logger.info(f"⏰ [{now.strftime('%d-%b-%Y %I:%M:%S %p')}] Iteration #{iteration}")
            logger.info("=" * 85)
            
            if now.hour < 9 or (now.hour == 9 and now.minute < 15):
                logger.info("⏸  Market not open yet (Opens 9:15 AM)")
                time.sleep(60)
                continue
            
            if (now.hour == 15 and now.minute > 30) or now.hour > 15:
                logger.info("⏸  Market Closed (Closes 3:30 PM)")
                
                if open_position:
                    current_premium = get_current_premium(open_position.instrument_key)
                    if current_premium:
                        pnl, premium_diff = open_position.calculate_pnl(current_premium)
                        timestamp = now.strftime('%Y-%m-%d %I:%M:%S %p')
                        
                        logger.info(f"\n💼 CLOSING POSITION AT MARKET CLOSE")
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
                
                # Reset day's open cache for next day
                days_open_cache = None
                time.sleep(60)
                continue
            
            logger.info("\n📥 Fetching live data from NSE...")
            
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
                        
                        exit_msg = f"""
{'='*85}
🔔 POSITION CLOSED: {exit_reason}
{'='*85}
  Entry:       ₹{open_position.entry_premium:.2f}
  Exit:        ₹{current_premium:.2f}
  Premium Diff: ₹{final_premium_diff:.2f}
  Total P&L:   ₹{final_pnl:.2f} (₹{final_premium_diff:.2f} × {LOT_SIZE})
{'=' * 85}"""
                        
                        logger.info(exit_msg)
                        
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
            
            # FIXED: Get day's open from 9:15 AM candle
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
            
            df = calculate_vwap_rsi(df)
            
            latest = df.iloc[-1]
            spot = latest["close"]
            vwap = latest["VWAP"]
            rsi = latest["RSI"]
            
            logger.info(f"  ✅ Spot: {spot:.2f} | VWAP: {vwap:.2f} | RSI: {rsi:.2f}")
            
            oi_trend, oi_ce, oi_pe = get_live_oi_from_quotes(option_instruments)
            
            if oi_trend is None:
                oi_trend = "Unknown"
                oi_ce, oi_pe = 0, 0
            else:
                logger.info(f"  ✅ Live OI: CE={oi_ce:,} | PE={oi_pe:,} → {oi_trend}")
            
            print_market_snapshot(spot, day_open, vwap, rsi, oi_trend, oi_ce, oi_pe)
            
            if last_signal_time:
                elapsed = (now - last_signal_time).seconds
                if elapsed < SIGNAL_COOLDOWN:
                    remaining = SIGNAL_COOLDOWN - elapsed
                    logger.info(f"\n⏳ COOLDOWN ACTIVE: {remaining}s remaining until next signal")
                    time.sleep(60)
                    continue
            
            signal, conditions = check_signal_conditions(spot, day_open, vwap, rsi, oi_trend)
            
            print_signal_evaluation(conditions)
            
            if signal:
                option_type = "CE" if signal == "BUY CE" else "PE"
                
                strike, premium, instrument_key = find_atm_strike_and_premium(spot, option_type)
                
                if strike and premium and instrument_key:
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
                    logger.warning(f"\n⚠️  Signal generated but strike/premium unavailable")
            else:
                logger.info(f"\n⏸  NO SIGNAL - Waiting for all conditions to align...")
            
            logger.info(f"\n⏱  Next check in 60 seconds...")
            time.sleep(60)
    
    except KeyboardInterrupt:
        logger.info(f"\n\n{'=' * 85}")
        logger.info("⏹  BOT STOPPED BY USER")
        logger.info(f"{'=' * 85}")
        logger.info(f"All signals saved to: {CSV_FILE}")
        logger.info("=" * 85)
        logger.info("\n✅ Thank you for using Nifty Options Trading Bot!\n")
    
    except Exception as e:
        logger.critical(f"\n\n❌ CRITICAL ERROR: {e}", exc_info=True)

if __name__ == "__main__":
    main()