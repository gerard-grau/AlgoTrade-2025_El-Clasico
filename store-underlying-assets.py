# save_underlyings.py

import asyncio
import json
import os
from pickle import TRUE
import signal
import sys
from datetime import datetime
from zoneinfo import ZoneInfo   # requires Python ≥3.9

import websockets      # pip install websockets
import websockets.exceptions

# ── CONFIG ────────────────────────────────────────────────────────────────
TEAM_SECRET = "e9e36d8c-9fc2-4047-9e49-bcd19c658470"
EXCHANGE_URI = "ws://192.168.100.10:9001/trade"
WS_URL       = f"{EXCHANGE_URI}?team_secret={TEAM_SECRET}"
UNDERLYINGS  = ["$CARD","$GARR","$HEST","$JUMP","$LOGN","$SIMP"]
BASE_DIR     = "market-data"
# ───────────────────────────────────────────────────────────────────────────

# storage for incoming price points: now a dict of dicts
storage     = {}
round_start: str | None = None
last_time   = None
ws          = None
DEBUG_VALUES = True

def init_dirs():
    os.makedirs(BASE_DIR, exist_ok=True)
    for u in UNDERLYINGS:
        os.makedirs(os.path.join(BASE_DIR, u.strip("$")), exist_ok=True)

def flush_round(start_ts: str):
    for u in UNDERLYINGS:
        recs = storage.get(u, {})
        if not recs:
            continue
        asset = u.strip("$")
        # include asset name in the filename
        fn = os.path.join(BASE_DIR, asset, f"{asset}_{start_ts}.json")
        # recs is now { "time1": {open:…,high:…}, "time2": {...}, … }
        with open(fn, "w") as f:
            json.dump(recs, f)
        # reset for next round
        storage[u] = {}

async def consumer():
    global last_time, round_start
    try:
        async for raw in ws:
            msg = json.loads(raw)
            if msg.get("type") != "market_data_update":
                continue

            t = msg["time"]
            # on first message of a new round, stamp the round
            if last_time is None or (last_time is not None and t < last_time):
                if round_start is not None:
                    flush_round(round_start)
                # use local timezone – e.g. Europe/Madrid (Spain) or Europe/Zagreb (Croatia)
                local_tz = ZoneInfo("Europe/Madrid")
                round_start = datetime.now(local_tz).strftime("%Y%m%d_%H%M%S")

            last_time = t

            untr = msg["candles"]["untradeable"]
            for u in UNDERLYINGS:
                arr = untr.get(u)
                if arr and len(arr):
                    # capture full candle info plus timestamp
                    record = arr[0].copy()
                    record["time"] = t
                    if DEBUG_VALUES:
                        print(f"{u} at time={t/1000} → {record}")
                        if u == "$SIMP": print()
                    # remove time from the stored value
                    ts = str(record.pop("time"))
                    storage[u][ts] = record

    except websockets.exceptions.ConnectionClosedOK:
        # lost connection (e.g. Wi-Fi drop): just exit consumer,
        # but keep round_start & last_time intact so we don't flush
        return
    except websockets.exceptions.ConnectionClosedError as e:
        # also just exit on error, preserve state
        print(f"INFO: connection closed: {e}")
        return

async def main():
    global ws, round_start, last_time
    init_dirs()
    # initialize storage and state
    for u in UNDERLYINGS:
        storage[u] = {}
    round_start = None
    last_time    = None

    # loop forever across rounds
    while True:
        try:
            # disable client pings to avoid timeouts
            async with websockets.connect(WS_URL, ping_interval=None) as socket:
                ws = socket
                await consumer()
        except Exception as e:
            # connection dropped or other error: log and reconnect
            print(f"INFO: connection dropped: {e}")
        # do NOT flush or reset here; only flush when time wraps
        # small pause before re-connecting
        await asyncio.sleep(1)

def on_sigint(sig, frame):
    print("Interrupted, flushing…")
    try:
        if round_start:
            flush_round(round_start)
    finally:
        sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, on_sigint)
    asyncio.run(main())