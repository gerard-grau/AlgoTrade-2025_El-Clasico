import asyncio
import json
from time import sleep

import websockets
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Tuple, Any
from collections import defaultdict


InstrumentID_t = str
Price_t = int
Time_t = int
Quantity_t = int
OrderID_t = str
TeamID_t = str

@dataclass
class BaseMessage:
    type: str

@dataclass
class AddOrderRequest(BaseMessage):
    type: str = field(default="add_order", init=False)
    user_request_id: str
    instrument_id: InstrumentID_t
    price: Price_t
    expiry: Time_t
    side: str
    quantity: Quantity_t

@dataclass
class CancelOrderRequest(BaseMessage):
    type: str = field(default="cancel_order", init=False)
    user_request_id: str
    order_id: OrderID_t
    instrument_id: InstrumentID_t

@dataclass
class GetInventoryRequest(BaseMessage):
    type: str = field(default="get_inventory", init=False)
    user_request_id: str

@dataclass
class GetPendingOrdersRequest(BaseMessage):
    type: str = field(default="get_pending_orders", init=False)
    user_request_id: str

@dataclass
class WelcomeMessage(BaseMessage):
    type: str
    message: str

@dataclass
class AddOrderResponseData:
    order_id: Optional[OrderID_t] = None
    message: Optional[str] = None
    immediate_inventory_change: Optional[Quantity_t] = None
    immediate_balance_change: Optional[Quantity_t] = None

@dataclass
class AddOrderResponse(BaseMessage):
    type: str
    user_request_id: str
    success: bool
    data: AddOrderResponseData

@dataclass
class CancelOrderResponse(BaseMessage):
    type: str
    user_request_id: str
    success: bool
    message: Optional[str] = None

@dataclass
class ErrorResponse(BaseMessage):
    type: str
    user_request_id: str
    message: str

@dataclass
class GetInventoryResponse(BaseMessage):
    type: str
    user_request_id: str
    data: Dict[InstrumentID_t, Tuple[Quantity_t, Quantity_t]]

@dataclass
class OrderJSON:
    orderID: OrderID_t
    teamID: TeamID_t
    price: Price_t
    time: Time_t
    expiry: Time_t
    side: str
    unfilled_quantity: Quantity_t
    total_quantity: Quantity_t
    live: bool

@dataclass
class GetPendingOrdersResponse:
    type: str
    user_request_id: str
    data: Dict[InstrumentID_t, Tuple[List[OrderJSON], List[OrderJSON]]]

@dataclass
class OrderbookDepth:
    bids: Dict[Price_t, Quantity_t]
    asks: Dict[Price_t, Quantity_t]

@dataclass
class CandleDataResponse:
    tradeable: Dict[InstrumentID_t, List[Dict[str, Any]]]
    untradeable: Dict[InstrumentID_t, List[Dict[str, Any]]]

Trade_t = Dict[str, Any]
Settlement_t = Dict[str, Any]
Cancel_t = Dict[str, Any]

@dataclass
class MarketDataResponse(BaseMessage):
    type: str
    time: Time_t
    candles: CandleDataResponse
    orderbook_depths: Dict[InstrumentID_t, OrderbookDepth]
    events: List[Dict[str, Any]]
    user_request_id: Optional[str] = None

global_user_request_id = 0

class DemoTradingBot:
    def __init__(self, uri: str, team_secret: str, print_market_data: bool = True):
        self.uri = f"{uri}?team_secret={team_secret}"
        self.ws = None
        self._pending: Dict[str, asyncio.Future] = {}
        self._trade_sequence_triggered = False
        self.print_market_data = print_market_data
        self._instrument_ids = {"$CARD", "$LOGN", "$HEST", "$JUMP", "$GARR", "$SIMP"}
        self.assets = {
            "$CARD": {
                "future": {},  # here time and depthbook
                "call": {},  # here time and strike price and depthbook
                "put": {},  # same as call
            },
            "$LOGN": {
                "future": {}, "call": {}, "put": {},
            },
            "$HEST": {
                "future": {}, "call": {}, "put": {},
            },
            "$JUMP": {
                "future": {}, "call": {}, "put": {},
            },
            "$GARR": {
                "future": {}, "call": {}, "put": {},
            },
            "$SIMP": {
                "future": {}, "call": {}, "put": {},
            }
        }
        self.candles = {
            "$CARD": {
                "asset": {}, #same as future
                "future": {},  # here time and price types
                "call": {},  # here time and strike price candles
                "put": {},  # same as call
            },
            "$LOGN": {
                "asset": {}, "future": {}, "call": {}, "put": {},
            },
            "$HEST": {
                "asset": {}, "future": {}, "call": {}, "put": {},
            },
            "$JUMP": {
                "asset": {}, "future": {}, "call": {}, "put": {},
            },
            "$GARR": {
                "asset": {}, "future": {}, "call": {}, "put": {},
            },
            "$SIMP": {
                "asset": {}, "future": {}, "call": {}, "put": {},
            }
        }

    async def connect(self):
        self.ws = await websockets.connect(self.uri)
        welcome_data = json.loads(await self.ws.recv())
        welcome_message = WelcomeMessage(**welcome_data)
        
        #(json.dumps({"welcome": asdict(welcome_message)}, indent=2))
        asyncio.create_task(self._receive_loop())

    async def _receive_loop(self):
        assert self.ws, "Websocket connection not established."
        async for msg in self.ws:
            data = json.loads(msg)

            #if not (data.get("type") == "market_data_update" and not self.print_market_data):
                #(json.dumps({"message": data}, indent=2))

            rid = data.get("user_request_id")
            if rid and rid in self._pending:
                self._pending[rid].set_result(data)
                del self._pending[rid]

            msg_type = data.get("type")
            if msg_type == "market_data_update":
                try:
                    parsed_orderbook_depths = {
                        instr_id: OrderbookDepth(**depth_data)
                        for instr_id, depth_data in data.get("orderbook_depths", {}).items()
                    }
                
                    candles_data = data.get("candles", {})
                    parsed_candles = CandleDataResponse(
                        tradeable=candles_data.get("tradeable", {}),
                        untradeable=candles_data.get("untradeable", {})
                    )

                    market_data = MarketDataResponse(
                        type=data["type"],
                        time=data["time"],
                        candles=parsed_candles,
                        orderbook_depths=parsed_orderbook_depths,
                        events=data.get("events", []),
                        user_request_id=data.get("user_request_id")
                    )
                    self._handle_market_data_update(market_data)
                except KeyError as e:
                    print(f"Error: Missing expected key in MarketDataResponse: {e}. Data: {data}")
                except Exception as e:
                    print(f"Error deserializing MarketDataResponse: {e}. Data: {data}")
            

    def _handle_market_data_update(self, data: MarketDataResponse):
        if self._trade_sequence_triggered:
            return

        # Save orderbook_depths into self.assets
        for instr_id, depth in data.orderbook_depths.items():
            # Parse instrument type, e.g. $CARD_call_100000_60
            parts = instr_id.split('_')
            base_asset = None
            option_type = None
            strike = None

            # For example instr_id = "$CARD_call_100000_60"
            # base_asset = $CARD, option_type=call, strike=100000

            # Determine base_asset and option_type:
            # It can be tricky if instr_id doesn't follow this pattern, so be defensive
            if parts[0].startswith('$'):
                base_asset = parts[0]
                if len(parts) > 1:
                    # Check if second part is "call" or "put" or "future"
                    if parts[1] in ('call', 'put', 'future'):
                        option_type = parts[1]
                    else:
                        option_type = "future"  # fallback if unknown, you can customize
                else:
                    option_type = "future"
            else:
                # fallback for unknown format
                base_asset = instr_id
                option_type = "future"

            if base_asset not in self.assets:
                # Initialize if not exists (optional)
                self.assets[base_asset] = {"future": {}, "call": {}, "put": {}}

            # Store bids and asks for this instrument in assets dictionary
            # We keep them indexed by instrument id string (you can customize indexing)
            if option_type not in self.assets[base_asset]:
                self.assets[base_asset][option_type] = {}

            self.assets[base_asset][option_type][instr_id] = {
                "bids": depth.bids,
                "asks": depth.asks
            }

        # Save candles into self.candles
        # Tradeable candles: usually options or instruments with strike prices
        for instr_id, candle_list in data.candles.tradeable.items():
            parts = instr_id.split('_')
            base_asset = None
            option_type = None

            if parts[0].startswith('$'):
                base_asset = parts[0]
                if len(parts) > 1:
                    if parts[1] in ('call', 'put', 'future'):
                        option_type = parts[1]
                    else:
                        option_type = "future"
                else:
                    option_type = "future"
            else:
                base_asset = instr_id
                option_type = "future"

            if base_asset not in self.candles:
                self.candles[base_asset] = {"asset": {}, "future": {}, "call": {}, "put": {}}

            if option_type not in self.candles[base_asset]:
                self.candles[base_asset][option_type] = {}

            # Store the list of candle dicts for this instrument
            self.candles[base_asset][option_type][instr_id] = candle_list

        # Untradeable candles: usually underlying assets, futures without strike
        for instr_id, candle_list in data.candles.untradeable.items():
            base_asset = instr_id  # usually base asset symbol like "$CARD"
            if base_asset not in self.candles:
                self.candles[base_asset] = {"asset": {}, "future": {}, "call": {}, "put": {}}
            # For untradeable, store in "asset" key (underlying prices)
            self.candles[base_asset]["asset"][instr_id] = candle_list

        # Optional: You can print or log this update if print_market_data is True
        if self.print_market_data:
            print(f"Updated assets and candles at time {data.time}")

        # --- Put–Call Parity Arbitrage Logic ---
        threshold = 10  # minimum price edge
        futures_mid: Dict[Tuple[str,int], float] = {}
        calls_mid: defaultdict = defaultdict(dict)
        puts_mid: defaultdict = defaultdict(dict)

        # 1) collect futures mid-prices per (underlying, expiry)
        for base in self._instrument_ids:
            for fut_id, book in self.assets[base]['future'].items():
                parts = fut_id.split('_')
                exp = int(parts[-1])
                bids, asks = book['bids'], book['asks']
                if not bids or not asks:
                    continue
                best_bid, best_ask = max(bids), min(asks)
                futures_mid[(base, exp)] = (best_bid + best_ask) / 2

        # 2) collect call/put mid-prices grouped by (underlying, expiry, strike)
        for base in self._instrument_ids:
            for call_id, book in self.assets[base]['call'].items():
                parts = call_id.split('_')
                strike, exp = int(parts[2]), int(parts[3])
                bids, asks = book['bids'], book['asks']
                if bids and asks:
                    calls_mid[(base, exp)][strike] = (max(bids) + min(asks)) / 2
            for put_id, book in self.assets[base]['put'].items():
                parts = put_id.split('_')
                strike, exp = int(parts[2]), int(parts[3])
                bids, asks = book['bids'], book['asks']
                if bids and asks:
                    puts_mid[(base, exp)][strike] = (max(bids) + min(asks)) / 2

        # 3) detect parity breaks and trigger trade()
        for (base, exp), fut_mid in futures_mid.items():
            for strike, c_mid in calls_mid.get((base, exp), {}).items():
                p_mid = puts_mid.get((base, exp), {}).get(strike)
                if p_mid is None:
                    continue
                synthetic = c_mid - p_mid + strike
                # synthetic cheap → buy synthetic, sell future
                if synthetic < fut_mid - threshold and not self._trade_sequence_triggered:
                    self._trade_sequence_triggered = True
                    asyncio.create_task(self.trade(base, strike, exp, fut_mid, c_mid, p_mid))
                # synthetic expensive → sell synthetic, buy future
                elif synthetic > fut_mid + threshold and not self._trade_sequence_triggered:
                    self._trade_sequence_triggered = True
                    asyncio.create_task(self.trade(base, strike, exp, fut_mid, c_mid, p_mid))
        # --- end parity logic ---

    def _is_success(self, response) -> bool:
        return isinstance(response, AddOrderResponse) and response.success

    async def trade(self,
                    instr: InstrumentID_t,
                    strike: Price_t,
                    exp: Price_t,
                    fut_mid: float,
                    call_mid: float,
                    put_mid: float):
        self._trade_sequence_triggered = True
        # Recompute synthetic
        synthetic = call_mid - put_mid + strike
        call_id = f"{instr}_call_{strike}_{exp}"
        put_id  = f"{instr}_put_{strike}_{exp}"
        fut_id  = f"{instr}_future_{exp}"
        threshold = 10
        print(f"Parity arb for {instr} K={strike} T={exp}: synth={synthetic:.1f}, fut={fut_mid:.1f}")

        if synthetic < fut_mid - threshold:
            # synthetic cheap → buy call, sell put, sell future
            print("→ Synthetic cheap: BUY call, SELL put, SELL future")
            legs = [
                ("buy",  call_id, call_mid),
                ("sell", put_id,  put_mid),
                ("sell", fut_id,  fut_mid),
            ]
        else:
            # synthetic expensive → sell call, buy put, buy future
            print("→ Synthetic expensive: SELL call, BUY put, BUY future")
            legs = [
                ("sell", call_id, call_mid),
                ("buy",  put_id,  put_mid),
                ("buy",  fut_id,  fut_mid),
            ]

        # Execute each leg sequentially
        for side, iid, price in legs:
            print(f"{side.upper()} {iid} @ {price}")
            resp = await (self.buy(iid, price) if side == "buy" else self.sell(iid, price))
            print("  ➔", resp)
            if not self._is_success(resp):
                print(f"  ✖ {side.upper()} {iid} failed, aborting arb.")
                break

        print("Parity arb sequence done.")
        self._trade_sequence_triggered = False

    async def send(self, payload: BaseMessage, timeout: int = 3):
        global global_user_request_id
        rid = str(global_user_request_id).zfill(10)
        global_user_request_id += 1

        payload.user_request_id = rid
        payload_dict = asdict(payload)

        fut = asyncio.get_event_loop().create_future()
        self._pending[rid] = fut

        await self.ws.send(json.dumps(payload_dict))
        #print(json.dumps({"sent": payload_dict}, indent=2))

        try:
            resp = await asyncio.wait_for(fut, timeout)
            if resp.get("type") == "add_order_response":
                resp['data'] = AddOrderResponseData(**resp.get('data', {}))
                return AddOrderResponse(**resp)
            elif resp.get("type") == "cancel_order_response":
                return CancelOrderResponse(**resp)
            elif resp.get("type") == "get_inventory_response":
                return GetInventoryResponse(**resp)
            elif resp.get("type") == "get_pending_orders_response":
                parsed_data = {}
                for instr_id, (bids_raw, asks_raw) in resp.get('data', {}).items():
                    parsed_bids = [OrderJSON(**order_data) for order_data in bids_raw]
                    parsed_asks = [OrderJSON(**order_data) for order_data in asks_raw]
                    parsed_data[instr_id] = (parsed_bids, parsed_asks)
                resp['data'] = parsed_data
                return GetPendingOrdersResponse(**resp)
            elif resp.get("type") == "error":
                return ErrorResponse(**resp)
            else:
                return resp
        except asyncio.TimeoutError:
            if rid in self._pending:
                del self._pending[rid]
            #print(json.dumps({"error": "timeout", "user_request_id": rid}, indent=2))
            return {"success": False, "user_request_id": rid, "message": "Request timed out"}

    async def buy(self, instr: InstrumentID_t, price: Price_t):
        expiry = int(instr.split("_")[-1])
        buy_request = AddOrderRequest(
            user_request_id="",
            instrument_id=instr,
            price=int(price*1.01),
            quantity=1,
            side="bid",
            expiry=expiry * 1000
        )
        print(buy_request)
        return await self.send(buy_request)

    async def get_pending_orders(self):
        get_pending_request = GetPendingOrdersRequest(user_request_id="")
        return await self.send(get_pending_request)

    async def get_inventory(self):
        get_inventory_request = GetInventoryRequest(user_request_id="")
        return await self.send(get_inventory_request)

    async def sell(self, instr: InstrumentID_t, price: Price_t):
        expiry = int(instr.split("_")[-1])
        sell_request = AddOrderRequest(
            user_request_id="",
            instrument_id=instr,
            price=int(price*0.99),
            quantity=1,
            side="ask",
            expiry=expiry * 1000
        )
        print(sell_request)
        return await self.send(sell_request)

    async def cancel(self, instr: InstrumentID_t, oid: OrderID_t):
        cancel_request = CancelOrderRequest(
            user_request_id="",
            order_id=oid,
            instrument_id=instr
        )
        return await self.send(cancel_request)

    async def run_sequence(self, instr: InstrumentID_t, price: Price_t):
        try:
            print(f"\n--- Running trading sequence for {instr} at price {price} ---")

            sell_order_id_to_check: Optional[OrderID_t] = None
            sell_order_successfully_submitted = False

            # --- Buy Order ---
            print("1) Sending Buy Order...")
            buy_resp = await self.buy(instr, price)
            
            if isinstance(buy_resp, AddOrderResponse) and buy_resp.success:
                print(f"   Buy Order SUBMISSION SUCCESS. OrderID: {buy_resp.data.order_id}")
            else:
                error_message = ""
                if isinstance(buy_resp, AddOrderResponse):
                    error_message = f" (Server message: '{buy_resp.data.message}')"
                elif isinstance(buy_resp, ErrorResponse):
                    error_message = f" (Error: '{buy_resp.message}')"
                else:
                    error_message = f" (Timeout or Unexpected response type: {type(buy_resp)})"
                print(f"   Buy Order SUBMISSION FAILED{error_message}. Aborting sequence.")
                return 

            # --- Get Inventory ---
            print("2) Getting Inventory after buy...")
            inventory_resp_after_buy = await self.get_inventory()
            initial_instrument_owned_quantity = 0
            initial_instrument_reserved_quantity = 0

            if isinstance(inventory_resp_after_buy, GetInventoryResponse):
                # Corrected: first element is reserved, second is owned
                reserved, owned = inventory_resp_after_buy.data.get(instr, (0, 0))
                initial_instrument_reserved_quantity = reserved
                initial_instrument_owned_quantity = owned
                print(f"   Current Inventory for {instr}: {initial_instrument_owned_quantity} owned, {initial_instrument_reserved_quantity} reserved.")
            else:
                error_message = ""
                if isinstance(inventory_resp_after_buy, ErrorResponse):
                    error_message = f" (Error: '{inventory_resp_after_buy.message}')"
                else:
                    error_message = f" (Timeout or Unexpected response type: {type(inventory_resp_after_buy)})"
                print(f"   Get Inventory failed{error_message}. Aborting sequence.")
                return

            if initial_instrument_owned_quantity <= 0:
                print(f"   Warning: Did not acquire {instr} after buy order (owned quantity is 0). Skipping sell attempts.")
                return

            # --- Sell Order ---
            sell_price = int(price * 1.01)
            print(f"3) Sending Sell Order at {sell_price}...")
            sell_resp = await self.sell(instr, sell_price)

            if isinstance(sell_resp, AddOrderResponse) and sell_resp.success:
                sell_order_id_to_check = sell_resp.data.order_id
                sell_order_successfully_submitted = True
                print(f"   Sell Order SUBMISSION SUCCESS. OrderID: {sell_order_id_to_check}.")
            else:
                error_message = ""
                if isinstance(sell_resp, AddOrderResponse):
                    error_message = f" (Server message: '{sell_resp.data.message}')"
                elif isinstance(sell_resp, ErrorResponse):
                    error_message = f" (Error: '{sell_resp.message}')"
                else:
                    error_message = f" (Timeout or Unexpected response type: {type(sell_resp)})"
                print(f"   Sell Order SUBMISSION FAILED{error_message}. No order was placed to cancel.")
            
            # --- Wait for Sell Order / Cancel if needed ---
            if sell_order_successfully_submitted and sell_order_id_to_check:
                print(f"4) Sell order was successfully submitted. Waiting for it to FILL or TIMEOUT ({5} seconds max), checking pending orders...")
                max_wait_time_for_fill = 5
                poll_interval = 0.5
                waited_time = 0
                order_still_pending = True

                while waited_time < max_wait_time_for_fill and order_still_pending:
                    await asyncio.sleep(poll_interval)
                    waited_time += poll_interval

                    pending_orders_resp = await self.get_pending_orders()
                    if isinstance(pending_orders_resp, GetPendingOrdersResponse):
                        order_found_in_pending = False
                        if instr in pending_orders_resp.data:
                            _, asks = pending_orders_resp.data[instr]
                            for order_json in asks:
                                if order_json.orderID == sell_order_id_to_check:
                                    order_found_in_pending = True
                                    print(f"   [{waited_time:.1f}s] Sell order {sell_order_id_to_check} still LIVE.")
                                    break
                        
                        order_still_pending = order_found_in_pending

                        if not order_still_pending:
                            print(f"   Sell order was successful.")
                            break 
                    else:
                        print(f"   [{waited_time:.1f}s] Failed to get pending orders during poll: {pending_orders_resp}. Cannot confirm if order is pending.")
                
                print(f"5) Final check after {max_wait_time_for_fill}s wait for sell order {sell_order_id_to_check}:")
                print(f"   - Order still pending (based on last poll): {order_still_pending}.")

                if sell_order_successfully_submitted and sell_order_id_to_check and order_still_pending:
                    print(f"   Sell order (ID: {sell_order_id_to_check}) still PENDING after TIMEOUT. Initiating cancellation...")
                    cancel_response = await self.cancel(instr, sell_order_id_to_check)
                    if isinstance(cancel_response, CancelOrderResponse) and cancel_response.success:
                        print(f"      Cancellation SUCCESS. Message: '{cancel_response.message}'.")
                    else:
                        error_message = ""
                        if isinstance(cancel_response, CancelOrderResponse):
                            error_message = f" (Server message: '{cancel_response.message}')"
                        elif isinstance(cancel_response, ErrorResponse):
                            error_message = f" (Error: '{cancel_response.message}')"
                        else:
                            error_message = f" (Timeout or Unexpected response type: {type(cancel_response)})"
                        print(f"      Cancellation FAILED{error_message}.")
                elif sell_order_successfully_submitted and sell_order_id_to_check and not order_still_pending:
                    pass # Message 'Sell order was successful.' already printed within the loop.
                else:
                    print("   Sell order was NOT SUCCESSFULLY SUBMITTED, or no order ID. No cancellation possible or needed.")

            else:
                print("4) Sell order was not submitted successfully. No waiting or cancellation needed.")

            print("--- Trading sequence complete ---")
        except Exception as e:
            print(f"An unexpected error occurred during trading sequence: {e}")
        finally:
            self._trade_sequence_triggered = False

async def main():
    EXCHANGE_URI = "ws://192.168.100.10:9001/trade"
    TEAM_SECRET = "e9e36d8c-9fc2-4047-9e49-bcd19c658470"

    bot = DemoTradingBot(
        EXCHANGE_URI,
        TEAM_SECRET,
        print_market_data=True
    )

    await bot.connect()
    await asyncio.Future()

if __name__ == '__main__':
    asyncio.run(main())
