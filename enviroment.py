import numpy as np


class PortfolioEnv:
    def __init__(self, df):
        self.df = df
        self.reset()

    def reset(self):
        self.current_step = 0
        self.balance = 10000  # starting money
        self.shares_held = 0
        self.net_worth = 10000
        return self._get_observation()

    def _get_observation(self):
        # closing price, collect data of RSI and SMA
        obs = np.array([
            self.df.iloc[self.current_step]['Close'],
            self.df.iloc[self.current_step]['RSI'],
            self.df.iloc[self.current_step]['SMA_20']
        ])
        return obs

    def step(self, action):
        # 0: Hold, 1: Buy, 2: Sell
        current_price = self.df.iloc[self.current_step]['Close']

        if action == 1:  # Buy
            if self.balance > current_price:
                self.shares_held += self.balance / current_price
                self.balance = 0
        elif action == 2:  # Sell
            self.balance += self.shares_held * current_price
            self.shares_held = 0

        self.current_step += 1
        self.net_worth = self.balance + (self.shares_held * current_price)

        reward = self.net_worth
        done = self.current_step >= len(self.df) - 1
        return self._get_observation(), reward, done