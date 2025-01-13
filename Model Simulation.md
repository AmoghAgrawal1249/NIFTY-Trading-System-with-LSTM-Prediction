# NIFTY Stock Price Prediction using LSTM Neural Networks

[Previous sections remain the same...]

## Trading Simulation

### Mock Trading Implementation
The project includes a sophisticated mock trading system that simulates real market conditions with the following features:

- Virtual trading with ₹100,000 initial balance
- Intraday trading simulation between 9:15 AM and 3:30 PM
- Automatic square-off at market close
- Position tracking and portfolio management
- Real-time profit/loss calculation

### Trading Components

#### MockBroker Class
```python
class MockBroker:
    """
    Simulates a trading environment with:
    - Balance tracking
    - Position management
    - Trading hour restrictions
    - Automatic square-off
    """
```

#### Trading Strategy
```python
def predict_next_day_price(model, recent_prices):
    """
    Predicts the next interval's price using the LSTM model
    
    Parameters:
        model: Trained LSTM model
        recent_prices: Last 30 intervals of price data
    
    Returns:
        predicted_price: Next interval's predicted price
    """
    recent_prices = np.array(recent_prices).reshape(1, 30, 1)
    predicted_price = model.predict(recent_prices)
    predicted_price = scaler.inverse_transform(predicted_price)
    return predicted_price
```

### Trading Rules
1. Entry Conditions:
   - Buy Signal: Predicted price > Current price * 1.001 (0.1% higher)
   - Sell Signal: Predicted price < Current price * 0.999 (0.1% lower)
   - Fixed position size: 10 units per trade

2. Risk Management:
   - Automatic end-of-day square-off
   - Trading hour restrictions (9:15 AM - 3:30 PM)
   - Real-time portfolio tracking

### Performance Analysis
- Net worth tracking over time
- Percentage return calculation
- Visual representation of returns
- Position and holding monitoring

### Sample Usage
```python
# Initialize broker
broker = MockBroker(
    balance=100000,
    price=stock_prices,
    trading_start=time(9, 15),
    trading_end=time(15, 30)
)

# Trading loop
for i in range(30, len(price_data)):
    current_time = price_data['datetime'].iloc[i].time()
    current_date = price_data['Date'].iloc[i]
    
    if not broker.trade_intraday(i, current_time):
        continue
        
    # Trading logic implementation
    recent_prices = price_data['close'].iloc[i-29:i+1].values
    current_price = price_data_unscaled['close'].iloc[i]
    predicted_price = predict_next_day_price(model, recent_prices)
    
    # Execute trades based on predictions
    if predicted_price > current_price * 1.001:
        broker.buy('nifty50', current_price, qty=10)
    elif predicted_price < current_price * 0.999:
        broker.sell('nifty50', current_price, qty=10)
```

### Visualization
```python
# Plot returns
plt.plot(broker.networth, label="Net Worth")
plt.title("Net Worth Over Time")
plt.xlabel("Days")
plt.ylabel("Net Worth")
plt.legend()
plt.savefig('returns.png')
```

### Performance Metrics
```python
# Calculate returns
initial_balance = 100000
final_net_worth = broker.networth[-1]
returns = ((final_net_worth - initial_balance) / initial_balance) * 100
print(f"Total returns: {returns:.2f}%")
```

### Future Enhancements for Trading Simulation
1. Advanced Position Sizing:
   - Dynamic position sizing based on volatility
   - Risk-adjusted position calculations
   - Portfolio optimization

2. Risk Management Features:
   - Stop-loss implementation
   - Take-profit orders
   - Trailing stops

3. Additional Analysis Tools:
   - Drawdown calculation
   - Sharpe ratio
   - Maximum adverse excursion

4. Market Condition Handling:
   - Gap opening management
   - Volatility-based trading restrictions
   - Circuit breaker simulation

[Previous sections continue...]
