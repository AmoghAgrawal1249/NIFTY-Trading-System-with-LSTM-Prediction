## Stock Price Prediction Using ML

from tvDatafeed import TvDatafeed, Interval
# Initialize TVDatafeed without logging in
tv = TvDatafeed()
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM
from sklearn.preprocessing import MinMaxScaler

import talib as ta
from datetime import datetime

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)



symbol = "NIFTY"
exchange = "NSE"
interval = Interval.in_5_minute #Using 5 min intervals
data = tv.get_hist(symbol=symbol, exchange=exchange, interval=interval , n_bars = 3000)  # Using last 3000 candles Nifty50 for data source (2500+500)

data.to_csv('stock_data.csv', index=True)
print("Data saved to stock_data.csv")  #Saving to csv for convinience



data = pd.read_csv('stock_data.csv', parse_dates=True) #Loading data 

n = 2500  # Skipping latest 500 candles
data = data.iloc[:n]  # Retain rows starting from the nth row onward

prices = data['close'].values

scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(prices.reshape(-1, 1))     #changing data in proportion to (0,1) [Max = 1 , Min = 0] val => X - Xmin / Xmax-Xmin

SEQ_LENGTH = 30  # Use 30 intervals of data to predict the next interval(may be changed)
X, y = create_sequences(prices_scaled, SEQ_LENGTH)



X = X.reshape(X.shape[0], X.shape[1], 1)  #Reshaping in the format (number of arrays , number of arrays in each array , no. of elememts in each of the inner array)

train = int(len(X)*0.8)

X_train = X[:train]
X_test = X[train:]
                        # 2000 intervals training data and 500 intervals validation
y_train = y[:train]
y_test = y[train:]

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(SEQ_LENGTH, 1)),       #Building the model 
    LSTM(50),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')  #Compiling the model
model.summary()

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")

# Make predictions
predictions = model.predict(X_test)

# Inverse transform the predictions to original scale
predictions = scaler.inverse_transform(predictions)
y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(y_test_original, label='Actual Prices', color='blue')
plt.plot(predictions, label='Predicted Prices', color='red')
plt.legend()
plt.savefig('initial.png')

model.save('stock_price_lstm_model.h5')
print("Model saved to stock_price_lstm_model.h5")  #Saving the model