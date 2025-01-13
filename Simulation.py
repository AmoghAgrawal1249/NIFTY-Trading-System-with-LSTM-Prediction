import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from datetime import time
from MockBroker import MockBroker
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))


# Load stock data
data = pd.read_csv('stock_data.csv', parse_dates=True)

# Ensure data is sorted correctly
data['datetime'] = pd.to_datetime(data['datetime'])
data['Date'] = data['datetime'].dt.date
data = data.sort_values(by=['Date'])

# Prepare price data (adjust columns as per your CSV)
price_data = data.tail(500)
price_data_unscaled = data.tail(500)
price_data['close'] = scaler.fit_transform(price_data['close'].values.reshape(-1, 1))
stock_prices = {'nifty50': price_data['close'].values}



# Initialize the broker
broker = MockBroker(
    balance=100000, 
    price=stock_prices, 
    trading_start=time(9, 15),  # Market start time: 9:15 AM
    trading_end=time(15, 30)   # Market end time: 3:30 PM
)

# Load the trained LSTM model
model = load_model('stock_price_lstm_model.h5')

# Function to predict the next interval's price
def predict_next_day_price(model, recent_prices):
    recent_prices = np.array(recent_prices).reshape(1, 30, 1)  # Shape as (1, 30, 1)
    predicted_price = model.predict(recent_prices)
    predicted_price = scaler.inverse_transform(predicted_price)
    return predicted_price

# Simulate trading
last_date = None
for i in range(30, len(price_data)):
    # Extract current time and date
    current_time = price_data['datetime'].iloc[i].time()  # Extract current time
    current_date = price_data['Date'].iloc[i]  # Extract current date
    day_opening_price = price_data_unscaled['open'].iloc[i] #Extract openening price of the day (or roughly the closing price of the previous day)
    
    # Check if trading is allowed
    if not broker.trade_intraday(i, current_time):
        continue

    # End-of-day square off
    if last_date is not None and current_date != last_date:
        broker.square_off(i,day_opening_price)  # Square off all positions at the end of the previous day
        print(broker.holdings)
    
    # Get the last 30 candles of prices for 'nifty50' (or other stocks)
    recent_prices = price_data['close'].iloc[i-29:i+1].values
    current_price = price_data_unscaled['close'].iloc[i]
    
    
    
    # Predict the next day's price
    predicted_price = predict_next_day_price(model, recent_prices)

    # Trading logic
    if predicted_price > current_price*1.001:  # Predicted price is 0.1% higher
        broker.buy('nifty50', current_price, qty=10)
    elif predicted_price < current_price * 0.999:  # Predicted price is 0.1% lower
        broker.sell('nifty50', current_price, qty=10)

    # Update net worth
    broker.manage_networth(i)

    # Update the last date
    last_date = current_date

current_price = price_data_unscaled['close'].iloc[499]

# Final square off at the end of the simulation
broker.square_off(len(price_data) - 1,current_price)
print(broker.holdings) #just to verify

# Calculate percentage returns
initial_balance = 100000
final_net_worth = broker.networth[-1]
returns = ((final_net_worth - initial_balance) / initial_balance) * 100
print(f"Total returns: {returns:.2f}%")

# Plot net worth over time
plt.plot(broker.networth, label="Net Worth")
plt.title("Net Worth Over Time")
plt.xlabel("Days")
plt.ylabel("Net Worth")
plt.legend()
plt.savefig('returns.png')
plt.show()
