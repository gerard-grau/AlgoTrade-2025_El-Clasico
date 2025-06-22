import asyncio
import json
import websockets

TEAM_SECRET = "e9e36d8c-9fc2-4047-9e49-bcd19c658470"
WS_URL = f"ws://192.168.100.10:9001/trade?team_secret={TEAM_SECRET}"
OUTPUT_FILE = "output-messages.txt"

async def main():
    # open file once in append (or "w" to overwrite each run)
    with open(OUTPUT_FILE, "w") as f:
        async with websockets.connect(WS_URL) as ws:
            f.write("Connected – dumping all incoming messages:\n")
            f.flush()
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                    formatted = json.dumps(msg, indent=2, sort_keys=True)
                except json.JSONDecodeError:
                    formatted = f"<<< non-JSON message >>> {raw}"
                f.write(formatted + "\n")
                f.flush()

if __name__ == "__main__":
    asyncio.run(main())