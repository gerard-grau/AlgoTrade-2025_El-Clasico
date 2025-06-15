import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import os
import asyncio
import json
import websockets
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Tuple, Any
from collections import defaultdict

# ==============================================================================
#  Configuration
# ==============================================================================
CURRENCIES = ["CARD", "GARR", "HEST", "JUMP", "LOGN", "SIMP"]
MODELS_PATH = './Prediction-model/models-4-predictions/' # Make sure this path is correct relative to where you run the script
HISTORY_SIZE = 500  # Number of recent candles to keep for feature calculation
PREDICTION_INTERVAL = 1000 # Run prediction cycle every 1000ms (1 second)

MODEL_CONFIGS = {
    'lower':  {'objective': 'quantile', 'alpha': 0.05, 'metric': 'quantile'},
    'median': {'objective': 'quantile', 'alpha': 0.5,  'metric': 'quantile'},
    'upper':  {'objective': 'quantile', 'alpha': 0.95, 'metric': 'quantile'},
    'mean':   {'objective': 'regression_l2', 'metric': 'l2'}
}

ONLINE_UPDATE_PARAMS = {
    'learning_rate': 0.01,
    'n_estimators': 10,
    'verbose': -1,
}

global_user_request_id = 0

# ==============================================================================
#  Message Dataclasses
# ==============================================================================
InstrumentID_t, Price_t, Time_t, Quantity_t, OrderID_t = str, int, int, int, str

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

# ... (All other dataclasses from your original file should be here) ...
@dataclass
class WelcomeMessage(BaseMessage):
    type: str
    message: str
    
@dataclass
class OrderbookDepth:
    bids: Dict[Price_t, Quantity_t]
    asks: Dict[Price_t, Quantity_t]

# ==============================================================================
#  Feature Engineering & Prediction Logic
# ==============================================================================

def create_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates a comprehensive set of features from a raw candle DataFrame."""
    # The raw DataFrame from the exchange has the timestamp as its index.
    # We save this as the 'time' column for later splitting.
    df['time'] = df.index
    
    # We must use the integer 'index' column from the JSON for all time-step calculations.
    # We reset the index to make 'index' a regular column, then set it as the new index.
    df_featured = df.reset_index(drop=True).set_index('index', drop=False)
    
    # --- The rest of your feature engineering code ---
    df_featured['body'] = df_featured['close'] - df_featured['open']
    df_featured['range'] = df_featured['high'] - df_featured['low']
    df_featured['body_pct'] = df_featured['body'] / (df_featured['range'] + 1e-6)
    df_featured['upper_wick'] = df_featured['high'] - df_featured[['open', 'close']].max(axis=1)
    df_featured['lower_wick'] = df_featured[['open', 'close']].min(axis=1) - df_featured['low']

    lags_to_create = [1, 2, 3, 5, 7, 10]
    columns_to_lag = ['mid', 'range', 'body_pct']
    for col in columns_to_lag:
        for lag in lags_to_create:
            df_featured[f'{col}_lag_{lag}'] = df_featured[col].shift(lag)

    windows = [6, 12, 25, 50, 100, 200]
    features_to_average = ['close', 'range', 'body_pct']
    for col in features_to_average:
        for w in windows:
            df_featured[f'{col}_sma_{w}'] = df_featured[col].rolling(window=w).mean()
            df_featured[f'{col}_ema_{w}'] = df_featured[col].ewm(span=w, adjust=False).mean()

    for w in windows:
        df_featured[f'close_div_ema_{w}'] = df_featured['close'] / (df_featured[f'close_ema_{w}'] + 1e-6)
    
    vol_windows = [12, 25, 50]
    for w in vol_windows:
        df_featured[f'range_std_{w}'] = df_featured['range'].rolling(window=w).std()

    ema_fast = df_featured['close'].ewm(span=12, adjust=False).mean()
    ema_slow = df_featured['close'].ewm(span=26, adjust=False).mean()
    df_featured['macd'] = ema_fast - ema_slow
    df_featured['macd_signal'] = df_featured['macd'].ewm(span=9, adjust=False).mean()
    df_featured['macd_hist'] = df_featured['macd'] - df_featured['macd_signal']
    
    return df_featured

def get_dynamic_target_timestamps(t_now: int, n_targets: int = 4, interval: int = 15) -> List[int]:
    """Calculates the nearest future n_targets that are multiples of the interval."""
    first_target = ((t_now // interval) + 1) * interval
    return [first_target + i * interval for i in range(n_targets)]

class LiveModelPredictor:
    """Manages loading, prediction, and online updates for all models."""
    def __init__(self, models_path: str, history_size: int):
        self.models_path = models_path
        self.history_size = history_size
        self.feature_list = None
        self.models: Dict[str, Dict[str, lgb.Booster]] = {}
        self.data_history: Dict[str, pd.DataFrame] = {curr: pd.DataFrame() for curr in CURRENCIES}
        self._load_all_artifacts()

    def _load_all_artifacts(self):
        print("[Predictor] Loading initial models and feature list...")
        try:
            self.feature_list = joblib.load(f'{self.models_path}feature_list.joblib')
            for currency in CURRENCIES:
                self.models[currency] = {}
                for name in ['lower', 'median', 'upper', 'mean']:
                    model_file = f'{self.models_path}{currency}_{name}_model.txt'
                    if os.path.exists(model_file):
                        self.models[currency][name] = lgb.Booster(model_file=model_file)
            print("[Predictor] All models loaded successfully.")
        except Exception as e:
            print(f"🔴 CRITICAL ERROR: Could not load initial models. Error: {e}")
            raise

    def add_candle_data(self, currency: str, candle: Dict[str, Any]):
        """Adds a new candle to the historical data, using the raw timestamp as index."""
        raw_timestamp = list(candle.keys())[0]
        candle_data = list(candle.values())[0]
        candle_data['timestamp'] = raw_timestamp
        
        new_candle_df = pd.DataFrame([candle_data]).set_index('timestamp')
        
        self.data_history[currency] = pd.concat([self.data_history[currency], new_candle_df])
        if len(self.data_history[currency]) > self.history_size:
            self.data_history[currency] = self.data_history[currency].iloc[-self.history_size:]

    def generate_predictions(self, currency: str) -> Optional[Dict[str, Any]]:
        history_df = self.data_history[currency]
        if len(history_df) < 250: return None

        features_df = create_all_features(history_df.copy())
        latest_features = features_df.iloc[[-1]]
        
        if latest_features.isnull().values.any(): return None

        t_now = latest_features.index[0]
        target_timestamps = get_dynamic_target_timestamps(t_now)
        if not target_timestamps: return None
        
        prediction_inputs = []
        for T_target in target_timestamps:
            horizon = T_target - t_now
            if horizon <= 0: continue
            input_row = latest_features.copy()
            input_row['forecast_horizon'] = horizon
            prediction_inputs.append(input_row)

        if not prediction_inputs: return None
        
        X_live = pd.concat(prediction_inputs)[self.feature_list]

        try:
            predictions = {name: model.predict(X_live) for name, model in self.models[currency].items()}
            predictions['target_timestamps'] = [t for t in target_timestamps if t > t_now]
            return predictions
        except Exception as e:
            print(f"Error during prediction for {currency}: {e}")
            return None

    def update_model(self, currency: str, X_new: pd.DataFrame, y_new: pd.Series):
        print(f"[{currency}] Performing online update with {len(X_new)} new sample(s)...")
        X_new_ordered = X_new[self.feature_list]
        for name, existing_booster in self.models[currency].items():
            model_path = f'{self.models_path}{currency}_{name}_model.txt'
            config = MODEL_CONFIGS[name]
            lgbm_update_params = ONLINE_UPDATE_PARAMS.copy()
            lgbm_update_params['objective'] = config['objective']
            if 'alpha' in config: lgbm_update_params['alpha'] = config['alpha']

            updated_booster = lgb.train(
                params=lgbm_update_params,
                train_set=lgb.Dataset(X_new_ordered, label=y_new),
                init_model=existing_booster,
                num_boost_round=lgbm_update_params['n_estimators'],
                verbose_eval=False
            )
            updated_booster.save_model(model_path)
            self.models[currency][name] = updated_booster
        print(f"[{currency}] Models updated successfully.")

# ==============================================================================
#  The Trading Bot Class
# ==============================================================================

class PredictiveTradingBot:
    def __init__(self, uri: str, team_secret: str):
        self.uri = f"{uri}?team_secret={team_secret}"
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self._pending: Dict[str, asyncio.Future] = {}
        self.predictor = LiveModelPredictor(models_path=MODELS_PATH, history_size=HISTORY_SIZE)
        self.orderbook_data: Dict[str, OrderbookDepth] = {}
        self.current_time = 0
        self.last_prediction_time = 0
        self.pending_training_data: Dict[Tuple[int, str], pd.DataFrame] = {}
        print("Predictive Trading Bot initialized.")

    async def connect(self):
        self.ws = await websockets.connect(self.uri)
        welcome_data = json.loads(await self.ws.recv())
        print(f"Welcome Message: {welcome_data}")
        print("Connection established. Waiting for messages...")
        await self._receive_loop()

    async def _receive_loop(self):
        assert self.ws
        async for msg in self.ws:
            data = json.loads(msg)
            msg_type = data.get("type")
            if msg_type == "market_data_update":
                await self._handle_market_data_update(data)
            # ... Handle other response message types ...

    async def _handle_market_data_update(self, data: Dict[str, Any]):
        self.current_time = data['time']
        self.orderbook_data = {k: OrderbookDepth(**v) for k, v in data.get("orderbook_depths", {}).items()}
        
        new_labeled_samples = defaultdict(list)
        for currency in CURRENCIES:
            asset_key = f"${currency}"
            if asset_key in data.get('candles', {}).get('untradeable', {}):
                for candle_data in data['candles']['untradeable'][asset_key]:
                    # The candle data is a dict with timestamp as key
                    timestamp, candle = list(candle_data.items())[0]
                    self.predictor.add_candle_data(currency, {timestamp: candle})
                    
                    candle_time_step = candle['index']
                    pending_key = (candle_time_step, currency)
                    
                    if pending_key in self.pending_training_data:
                        X_new_row = self.pending_training_data[pending_key]
                        y_new_value = candle['close']
                        new_labeled_samples[currency].append((X_new_row, y_new_value))
                        del self.pending_training_data[pending_key]

        for currency, samples in new_labeled_samples.items():
            if samples:
                X_batch = pd.concat([s[0] for s in samples])
                y_batch = pd.Series([s[1] for s in samples])
                self.predictor.update_model(currency, X_batch, y_batch)

        if self.current_time >= self.last_prediction_time + PREDICTION_INTERVAL:
            self.last_prediction_time = self.current_time
            asyncio.create_task(self._run_prediction_and_trade_cycle())

    async def _run_prediction_and_trade_cycle(self):
        print(f"\n--- Running Prediction Cycle at time {self.current_time} ---")
        for currency in CURRENCIES:
            predictions = self.predictor.generate_predictions(currency)
            if predictions:
                latest_feature_row = self.predictor.data_history[currency].iloc[[-1]]
                features_for_storage = create_all_features(latest_feature_row.copy())

                for T_target in predictions['target_timestamps']:
                    self.pending_training_data[(T_target, currency)] = features_for_storage
                
                print(f"  - {currency} Prediction for T={predictions['target_timestamps'][0]}: Mean={predictions['mean'][0]:.2f}")
                await self._execute_trading_logic(currency, predictions)

    async def _execute_trading_logic(self, currency: str, predictions: Dict[str, Any]):
        target_timestamp = predictions['target_timestamps'][0]
        predicted_mean = predictions['mean'][0]
        
        future_instrument_id, expiry = None, 0
        for instr_id in self.orderbook_data.keys():
            if instr_id.startswith(f"${currency}_future_"):
                try:
                    exp = int(instr_id.split('_')[2])
                    if exp > target_timestamp:
                        future_instrument_id, expiry = instr_id, exp
                        break
                except (IndexError, ValueError): continue
        
        if not future_instrument_id: return
        book = self.orderbook_data.get(future_instrument_id)
        if not book or not book.bids or not book.asks: return

        best_ask, best_bid = min(book.asks.keys()), max(book.bids.keys())
        mid_price = (best_ask + best_bid) / 2
        profit_threshold = 150

        if predicted_mean > mid_price + profit_threshold:
            print(f"  - TRADE SIGNAL: BUY {currency} future at {best_ask}")
            # await self.place_order(future_instrument_id, best_ask, "bid", expiry)
        elif predicted_mean < mid_price - profit_threshold:
            print(f"  - TRADE SIGNAL: SELL {currency} future at {best_bid}")
            # await self.place_order(future_instrument_id, best_bid, "ask", expiry)

    async def send(self, payload: BaseMessage, timeout: int = 3):
        global global_user_request_id
        rid = str(global_user_request_id).zfill(10)
        global_user_request_id += 1
        payload.user_request_id = rid
        payload_dict = asdict(payload)
        fut = asyncio.get_event_loop().create_future()
        self._pending[rid] = fut
        await self.ws.send(json.dumps(payload_dict))
        try:
            return await asyncio.wait_for(fut, timeout)
        except asyncio.TimeoutError:
            del self._pending[rid]
            return {"success": False, "message": "Request timed out"}

# ==============================================================================
#  Main Execution Block
# ==============================================================================

async def main():
    EXCHANGE_URI = "ws://192.168.100.10:9001/trade"
    TEAM_SECRET = "e9e36d8c-9fc2-4047-9e49-bcd19c658470"

    bot = PredictiveTradingBot(EXCHANGE_URI, TEAM_SECRET)
    await bot.connect()
    
    try:
        await asyncio.Future()
    except KeyboardInterrupt:
        print("\nBot stopped by user.")

if __name__ == '__main__':
    asyncio.run(main())