import math
import time
import warnings
from datetime import datetime as dt

import ccxt
import numpy as np
import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

pd.set_option("display.max_rows", None)
warnings.filterwarnings("ignore")
# Bot setting
BOT_NAME = "VXMA"
# Change Symbol Here
SYMBOL_NAME = "BTC"
# Change Time Frame Here
TF = "1d"
# API CONNECT
exchange = ccxt.binance()
symboli = SYMBOL_NAME + "/USDT"


print(f"Fetching new bars for {dt.now().isoformat()}")
bars = exchange.fetch_ohlcv(symboli, timeframe=TF, since=None, limit=2002)
df = pd.DataFrame(
    bars[:-1], columns=["timestamp", "Open", "High", "Low", "Close", "Volume"]
)
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")


def indicator(emafast, emaslow):
    global df
    df["emafast"] = ta.ema(df["Close"], int(emafast))
    df["emaslow"] = ta.ema(df["Close"], int(emaslow))
    return df["emafast"], df["emaslow"]


def signalbuy():
    trade = crossover(pd.Series(df["emafast"]), pd.Series(df["emaslow"]))
    return trade


def signalsell():
    trade = crossover(pd.Series(df["emaslow"]), pd.Series(df["emafast"]))
    return trade


class run_bot(Strategy):
    ema_fast = 12
    ema_slow = 26

    def init(self):
        super().init()
        self.A2 = self.I(
            indicator,
            self.ema_fast,
            self.ema_slow,
        )
        self.A0 = signalbuy
        self.A1 = signalsell

    def next(self):
        if self.A0:
            self.position.close()
            self.buy()
        elif self.A1:
            self.position.close()
            self.sell()


bt = Backtest(df, run_bot, cash=100000)
stat = bt.run()
# stat = bt.optimize(
# ema_fast = range(1,200,2),
# ema_slow = range(1,200,2),
# maximize = 'Win Rate [%]')
print(stat)
bt.plot()
