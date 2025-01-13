# NIFTY Stock Price Prediction using LSTM Neural Networks

A sophisticated deep learning implementation for predicting NIFTY stock prices using Long Short-Term Memory (LSTM) neural networks. This project leverages the power of TradingView data through tvDatafeed and implements a sequence-based prediction approach with comprehensive data preprocessing and model architecture.

## Table of Contents
- [Overview](#overview)
- [Technical Architecture](#technical-architecture)
- [Installation](#installation)
- [Data Pipeline](#data-pipeline)
- [Model Implementation](#model-implementation)
- [Training Process](#training-process)
- [Evaluation](#evaluation)
- [Usage Guide](#usage-guide)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Overview

This project implements a time-series prediction model for NIFTY stock prices with the following key features:
- Real-time data acquisition from TradingView
- Advanced sequence-based prediction using LSTM networks
- Comprehensive data preprocessing and normalization
- Customizable prediction intervals
- Visual performance analysis
- Model persistence and reusability

## Technical Architecture

### Dependencies
```python
from tvDatafeed import TvDatafeed, Interval
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import talib as ta
from datetime import datetime
```

### Key Components
1. **Data Fetching**: TvDatafeed integration for real-time market data
2. **Data Processing**: Custom sequence creation and normalization
3. **Model Architecture**: Multi-layer LSTM network
4. **Visualization**: Matplotlib-based performance plotting
5. **Persistence**: Model and data storage capabilities

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/nifty-price-prediction.git

# Install required packages
pip install -r requirements.txt
```

### Requirements.txt
```
tensorflow>=2.7.0
pandas>=1.3.0
numpy>=1.19.5
matplotlib>=3.4.3
scikit-learn>=0.24.2
tvDatafeed>=2.0.0
talib-binary>=0.4.19
```

## Data Pipeline

### Data Collection
```python
# Initialize TVDatafeed
tv = TvDatafeed()

# Configuration parameters
symbol = "NIFTY"
exchange = "NSE"
interval = Interval.in_5_minute

# Fetch historical data
data = tv.get_hist(
    symbol=symbol, 
    exchange=exchange, 
    interval=interval,
    n_bars=3000
)
```

### Sequence Creation
```python
def create_sequences(data, seq_length):
    """
    Creates sequences for LSTM input with specified length
    
    Parameters:
        data (np.array): Input time series data
        seq_length (int): Length of each sequence
    
    Returns:
        X (np.array): Input sequences
        y (np.array): Target values
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)
```

### Data Preprocessing
- Normalization using MinMaxScaler
- Sequence creation with 30-interval windows
- 80/20 train-test split
- Data reshaping for LSTM input

## Model Implementation

### LSTM Architecture
```python
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(SEQ_LENGTH, 1)),
    LSTM(50),
    Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mean_squared_error'
)
```

### Model Parameters
- Input Shape: (30, 1) - 30 time steps with 1 feature
- First LSTM Layer: 50 units with return sequences
- Second LSTM Layer: 50 units
- Dense Output Layer: 1 unit (price prediction)
- Optimizer: Adam with default learning rate
- Loss Function: Mean Squared Error

## Training Process

### Training Configuration
```python
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)
```

### Hyperparameters
- Epochs: 20
- Batch Size: 32
- Training Data: 2000 sequences
- Validation Data: 500 sequences
- Early Stopping: Not implemented (can be added)

## Evaluation

### Performance Metrics
- Mean Squared Error (MSE) on test set
- Visual comparison of predicted vs actual prices

### Visualization
```python
plt.figure(figsize=(10, 6))
plt.plot(y_test_original, label='Actual Prices', color='blue')
plt.plot(predictions, label='Predicted Prices', color='red')
plt.legend()
plt.savefig('initial.png')
```

## Usage Guide

### Basic Usage
1. Configure data parameters:
```python
symbol = "NIFTY"
exchange = "NSE"
interval = Interval.in_5_minute
```

2. Run prediction:
```python
# Load data
data = tv.get_hist(symbol=symbol, exchange=exchange, interval=interval, n_bars=3000)

# Preprocess
prices = data['close'].values
prices_scaled = scaler.fit_transform(prices.reshape(-1, 1))

# Create sequences
X, y = create_sequences(prices_scaled, SEQ_LENGTH)

# Make predictions
predictions = model.predict(X_test)
```

### Model Persistence
```python
# Save model
model.save('stock_price_lstm_model.h5')

# Load model
from tensorflow.keras.models import load_model
model = load_model('stock_price_lstm_model.h5')
```

## Advanced Features

### Potential Enhancements
1. Technical Indicators Integration
```python
def add_technical_indicators(df):
    df['RSI'] = ta.RSI(df['close'].values)
    df['MACD'], df['MACD_signal'], _ = ta.MACD(df['close'].values)
    return df
```

2. Multiple Timeframe Analysis
3. Ensemble Methods
4. Advanced Feature Engineering
5. Real-time Trading Integration

## Troubleshooting

Common issues and solutions:
1. Data Fetching Errors:
   - Check internet connection
   - Verify exchange trading hours
   - Ensure symbol validity

2. Model Performance Issues:
   - Adjust sequence length
   - Modify LSTM architecture
   - Fine-tune hyperparameters

## Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## License

MIT License

## Disclaimer

This project is for educational and research purposes only. Trading financial instruments carries risk, and this model should not be used as the sole basis for trading decisions.

---

## Citation

If you use this project in your research or work, please cite:

```
@misc{nifty-price-prediction,
  author = {Your Name},
  title = {NIFTY Stock Price Prediction using LSTM Neural Networks},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/nifty-price-prediction}
}
```
