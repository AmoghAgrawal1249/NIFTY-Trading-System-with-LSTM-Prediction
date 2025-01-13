import numpy as np
import pandas as pd
import talib as ta

class MockBroker:
    def __init__(self, balance, price, trading_start, trading_end):
        self.holdings = {}
        self.holdingsprofitat = {}
        self.balance = balance
        self.highest_networth = 0
        self.price = price
        self.equity = 0
        self.networth = []
        self.stocks_traded = set()
        self.blacklisted_stocks = {}
        self.trading_start = trading_start  # Start of trading session (e.g., 9:15 AM)
        self.trading_end = trading_end      # End of trading session (e.g., 3:30 PM)

    def __str__(self):
        return (
            f"Equity: {self.equity}\n"
            f"Holdings: {self.holdings}\n"
            f"Balance: {self.balance}\n"
            f"Total Net Worth: {self.networth[-1]}\n"
            f"Highest Net Worth: {self.highest_networth}\n"
            f"Profit/Loss on Stocks: {self.holdingsprofitat}"
        )

    def manage_networth(self, i):
        """Calculate the net worth for the current time step."""
        self.equity = 0
        for stock, qty in self.holdings.items():
            self.equity += self.price.get(stock, [0])[i] * qty
        current_networth = self.equity + self.balance
        self.networth.append(current_networth)

        if current_networth > self.highest_networth:
            self.highest_networth = current_networth

        print(f"Net Worth: {current_networth}")

    def buy(self, stock, price, qty):
        """Execute a buy order."""
        total_cost = price * qty
        #if self.balance >= total_cost:
        self.holdings[stock] = self.holdings.get(stock, 0) + qty
        self.holdingsprofitat[stock] = self.holdingsprofitat.get(stock, 0) - total_cost
        self.balance -= total_cost
        print(f"Bought {qty} of {stock} at {price} each")
        print(f"Balance: {self.balance}")
        self.stocks_traded.add(stock)
        '''else:
            print("Insufficient balance to execute the trade")'''

    def sell(self, stock, price, qty):
        """Execute a sell order."""
        #if stock in self.holdings and self.holdings[stock] >= qty:
        self.holdings[stock] = self.holdings.get(stock, 0) - qty
        self.balance += price * qty
        self.holdingsprofitat[stock] = self.holdingsprofitat.get(stock, 0) + price * qty
        print(f"Sold {qty} of {stock} at {price} each")
        print(f"Balance: {self.balance}")
        '''else:
            print(f"Insufficient holdings to sell {qty} of {stock}")'''

    def square_off(self, i , price):
        """Square off all positions at the end of the trading day."""
        for stock, qty in list(self.holdings.items()):
            if qty > 0:
                self.sell(stock, price, qty)
            else:
                self.buy(stock, price, -qty)
        self.manage_networth(i)

    def sell_off_if(self, i):
        """Sell profitable holdings."""
        for stock, qty in list(self.holdings.items()):
            if self.holdingsprofitat.get(stock, 0) + (self.price[stock][i] * qty) > 0:
                self.sell(stock, self.price[stock][i], qty)
        self.manage_networth(i)

    def sell_off_if_thres(self, i, threshold):
        """Sell holdings if the profit/loss exceeds a threshold."""
        for stock, qty in list(self.holdings.items()):
            profit_loss = self.holdingsprofitat.get(stock, 0) + (self.price[stock][i] * qty)
            if profit_loss < -threshold or profit_loss > 0:
                self.sell(stock, self.price[stock][i], qty)
        self.manage_networth(i)

    def sell_off_if_thres_blacklist(self, i, threshold, blacklist_threshold):
        """Sell holdings and add to blacklist if loss exceeds a threshold."""
        for stock, qty in list(self.holdings.items()):
            profit_loss = self.holdingsprofitat.get(stock, 0) + (self.price[stock][i] * qty)
            if profit_loss < -threshold:
                self.sell(stock, self.price[stock][i], qty)
                self.blacklisted_stocks[stock] = self.blacklisted_stocks.get(stock, 0) + 1
            elif profit_loss > 0:
                self.sell(stock, self.price[stock][i], qty)
        self.manage_networth(i)

    def trade_intraday(self, i, current_time):
        """Enforce intraday trading rules."""
        if current_time < self.trading_start or current_time > self.trading_end:
            print("Trading is only allowed during market hours.")
            return False

        # Ensure square-off by end of day
        if current_time == self.trading_end:
            self.square_off(i)

        return True

    def get_holdings(self):
        """Return the current holdings."""
        return self.holdings
 
