# AlgoTrade 2025 - El Clasico Team

A sophisticated algorithmic trading system that uses **Put-Call Parity** signals to drive its core trading strategy, supplemented by **Predictive Market Models** for financial derivatives trading.

## 🚀 Key Features

- **Hybrid Trading Strategy**: Implements a strategy where Put-Call Parity violations are used as a signal to execute a two-legged trade (buying a future and a put option).
- **Predictive Modeling**: Utilizes a LightGBM-based pipeline to forecast asset prices with uncertainty quantification.
- **Real-Time Trading**: Features a robust WebSocket-based engine for live data processing and order management.
- **Comprehensive Tooling**: Includes scripts for data collection, feature engineering, model training, and evaluation.

## 📊 Trading Strategy: Put-Call Parity Signal

The primary trading strategy, implemented in `fixedDemoTradingBot.py`, uses the Put-Call Parity principle as a signal generator for placing trades. It is a form of statistical arbitrage rather than a pure, risk-free arbitrage.

**Core Logic**:
1.  **Real-Time Pricing**: The bot continuously calculates weighted average prices for futures, calls, and puts from the live order book.
2.  **Synthetic Future Calculation**: It uses the Put-Call Parity formula (`Call Price - Put Price + Strike Price`) to calculate a "synthetic" future price for each options series.
3.  **Signal Detection**: A trading signal is generated when the synthetic future price is significantly more expensive (by a configurable threshold, e.g., >2.25%) than the actual traded future's price.
4.  **Trade Execution**: Upon detecting a signal, the bot executes a specific two-leg trade:
    *   It **BUYS** the actual future contract.
    *   It **BUYS** the corresponding put option.

## 🤖 Predictive Analytics Pipeline

We use a LightGBM-based machine learning pipeline to forecast future asset prices, complete with uncertainty estimates.

### Pipeline Execution
To run the full pipeline from data processing to model training and evaluation:
```bash
# 1. Preprocess data and create features
python Prediction-model/preprocessing-with-correlation.py

# 2. Train the prediction models
python Prediction-model/train-model.py

# 3. Evaluate model performance
python Prediction-model/evaluate-and-plot.py
```

### Feature Engineering
The pipeline automatically engineers a rich feature set from raw time-series data, including:
- **Price-Action Features**: Candlestick properties like body size and wick length.
- **Temporal Features**: Lags and moving averages (SMA/EMA) over various periods.
- **Momentum Indicators**: MACD and other standard momentum metrics.
- **Cross-Asset Features**: Relative strength against the market average.

### Model Architecture & Evaluation
- **Multi-Model Approach**: For each asset, we train four LightGBM models to predict the mean, median (50th percentile), and a 90% prediction interval (using the 5th and 95th percentile models). This interval provides a crucial measure of market uncertainty, which is vital for risk management.
- **Evaluation**: Models are rigorously evaluated for accuracy (MAE, RMSE) and the reliability of their prediction intervals.

## 🏗️ System Architecture

The project's architecture is designed to separate data handling, model training, and trading execution.

**Data Flow:**

1.  **Data Collection**: `store-underlying-assets.py` captures live data from the `Exchange WebSocket` and saves it into the `market-data/` directory.
2.  **ML Preprocessing**: `preprocessing-with-correlation.py` reads the raw data, engineers features, and stores the result in `processed-data/`.
3.  **Model Training**: `train-model.py` uses the processed data to train the LightGBM models, which are saved in `models-4-predictions/`.
4.  **Trading**:
    *   `bot-with-lightgbm-predictor.py` loads the trained models to inform its trading decisions.
    *   `fixedDemoTradingBot-put-call-parity.py` trades based on the put-call parity strategy without relying on the ML models.

This can be visualized as:
```
[Exchange] -> Data Collection -> [Raw Data] -> ML Pipeline -> [Trained Models] -> ML Bot -> [Exchange]
                                                                                Arbitrage Bot -> [Exchange]
```

<br/>

## ⚙️ Getting Started

### Prerequisites

Before running the bots, make sure you have all the necessary Python packages installed. You can install them using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Configuration

Before running a bot, you may need to configure its parameters.

- **Team Secret**: Set your `TEAM_SECRET` in the corresponding Python file:
  ```python
  TEAM_SECRET = "your-team-secret-here"
  ```
- **Arbitrage Strategy**: In `fixedDemoTradingBot-put-call-parity.py`, you can adjust the `threshold` to define the minimum profit edge required to trigger an arbitrage trade.
- **ML-Based Strategy**: In `bot-with-lightgbm-predictor.py`, you can modify parameters related to trade logic, although the core settings are tied to the trained models.

### Running a Bot
To run the primary arbitrage bot:
```bash
python fixedDemoTradingBot-put-call-parity.py
```

## 📁 Project Structure

The repository is organized into the following key components:

- **`Trading Bots`**: Contains the main trading algorithms.
    - `fixedDemoTradingBot-put-call-parity.py`: **(Primary)** The main bot implementing the Put-Call Parity arbitrage strategy.
    - `bot-with-lightgbm-predictor.py`: An experimental bot that integrates the ML prediction pipeline.
    - `fixedDemoTradingBot.py` & `demoTradingBot.py`: Earlier versions of the trading bot, kept for reference.

- **`Prediction-model/`**: The complete machine learning pipeline for price forecasting.
    - `preprocessing-with-correlation.py`: Cleans data and engineers features.
    - `train-model.py`: Trains the LightGBM prediction models.
    - `evaluate-and-plot.py`: Validates model performance.
    - `Preprocessing.ipynb` & `train-notebook.ipynb`: Notebooks for interactive development.
    - `models-4-predictions/`: Stores trained model artifacts.
    - `processed_data/`: Contains processed data for model training.

- **`Data & Utilities`**:
    - `store-underlying-assets.py`: Captures and saves live market data.
    - `see-messages.py`: A debugging tool to inspect raw WebSocket messages.
    - `market-data/`: Contains raw market data (JSON files with OHLCV candles).
    - `processed-data/`: Stores the master datasets (Parquet files) with engineered features like moving averages and MACD.
    - `BAD-market-data/`: Contains data with known anomalies for robust testing.

- `requirements.txt`: Lists all Python package dependencies.

## 🤝 Contributing

This project was developed for the AlgoTrade 2025 hackathon. Future work could include refining strategies, enhancing ML models, and improving risk management.

## 📄 License

Developed by Team El Clasico for the AlgoTrade 2025 competition.

---

*Built with ❤️ for algorithmic trading excellence*