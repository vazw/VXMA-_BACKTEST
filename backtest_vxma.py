import math
import time
import warnings
from datetime import datetime as dt

import ccxt
import numpy as np
import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.lib import barssince, crossover

pd.set_option("display.max_rows", None)
warnings.filterwarnings("ignore")
# Bot setting
BOT_NAME = "VXMA"
# Change Symbol Here
SYMBOL_NAME = "BTC"
# Change Time Frame Here
TF = "15m"
# API CONNECT
exchange = ccxt.binance()
symboli = SYMBOL_NAME + "/USDT"
exchange.load_markets()
market = exchange.markets[symboli]
print(f"Fetching new bars for {dt.now().isoformat()}")
bars = exchange.fetch_ohlcv(symboli, timeframe=TF, since=None, limit=2002)
df = pd.DataFrame(
    bars[:-1], columns=["timestamp", "Open", "High", "Low", "Close", "Volume"]
)
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

# Alphatrend
def alphatrend(df, atr_p, atr_m, rsi):
    df["atr"] = ta.sma(
        ta.true_range(df["High"], df["Low"], df["Close"]), atr_p
    )
    df["rsi"] = ta.rsi(df["Close"], rsi)
    df["downT"] = 0.0
    df["upT"] = 0.0
    df["alphatrend"] = 0.0
    # AlphaTrend rsibb >= 50 ? upT < nz(AlphaTrend[1]) ? nz(AlphaTrend[1]) : upT : downT > nz(AlphaTrend[1]) ? nz(AlphaTrend[1]) : downT
    for current in range(1, len(df.index)):
        previous = current - 1
        df["downT"][current] = df["High"][current] + df["atr"][current] * atr_m
        df["upT"][current] = df["Low"][current] - df["atr"][current] * atr_m
        if df["rsi"][current] >= 50:
            if df["upT"][current] < (
                df["alphatrend"][previous]
                if df["alphatrend"][previous] != None
                else 0
            ):
                df["alphatrend"][current] = (
                    df["alphatrend"][previous]
                    if df["alphatrend"][previous] != None
                    else 0
                )
            else:
                df["alphatrend"][current] = df["upT"][current]
        else:
            if df["downT"][current] > (
                df["alphatrend"][previous]
                if df["alphatrend"][previous] != None
                else 0
            ):
                df["alphatrend"][current] = (
                    df["alphatrend"][previous]
                    if df["alphatrend"][previous] != None
                    else 0
                )
            else:
                df["alphatrend"][current] = df["downT"][current]
    return df


# Andean_Oscillator
def andean(df, AOL):
    df["up1"] = 0.0
    df["up2"] = 0.0
    df["dn1"] = 0.0
    df["dn2"] = 0.0
    df["cmpbull"] = 0.0
    df["cmpbear"] = 0.0
    alpha = 2 / (AOL + 1)
    for current in range(1, len(df.index)):
        previous = current - 1
        CloseP = df["Close"][current]
        OpenP = df["Open"][current]
        up1 = df["up1"][previous]
        up2 = df["up2"][previous]
        dn1 = df["dn1"][previous]
        dn2 = df["dn2"][previous]
        # up1 := nz(math.max(C, O, up1[1] - (up1[1] - C) * alpha), C)
        df["up1"][current] = (
            max(CloseP, OpenP, up1 - (up1 - CloseP) * alpha)
            if max(CloseP, OpenP, up1 - (up1 - CloseP) * alpha) != None
            else df["Close"][current]
        )
        # up2 := nz(math.max(C * C, O * O, up2[1] - (up2[1] - C * C) * alpha), C * C)
        df["up2"][current] = (
            max(
                CloseP * CloseP,
                OpenP * OpenP,
                up2 - (up2 - CloseP * CloseP) * alpha,
            )
            if max(
                CloseP * CloseP,
                OpenP * OpenP,
                up2 - (up2 - CloseP * CloseP) * alpha,
            )
            != None
            else df["Close"][current] * df["Close"][current]
        )
        # dn1 := nz(math.min(C, O, dn1[1] + (C - dn1[1]) * alpha), C)
        df["dn1"][current] = (
            min(CloseP, OpenP, dn1 + (CloseP - dn1) * alpha)
            if min(CloseP, OpenP, dn1 + (CloseP - dn1) * alpha) != None
            else df["Close"][current]
        )
        # dn2 := nz(math.min(C * C, O * O, dn2[1] + (C * C - dn2[1]) * alpha), C * C)
        df["dn2"][current] = (
            min(
                CloseP * CloseP,
                OpenP * OpenP,
                dn2 + (CloseP * CloseP - dn2) * alpha,
            )
            if min(
                CloseP * CloseP,
                OpenP * OpenP,
                dn2 + (CloseP * CloseP - dn2) * alpha,
            )
            != None
            else df["Close"][current] * df["Close"][current]
        )
        up1n = df["up1"][current]
        up2n = df["up2"][current]
        dn1n = df["dn1"][current]
        dn2n = df["dn2"][current]
        df["cmpbull"][current] = math.sqrt(dn2n - (dn1n * dn1n))
        df["cmpbear"][current] = math.sqrt(up2n - (up1n * up1n))
    return df


# VXMA Indicator
def vxma(df):
    df["vxma"] = 0.0
    df["trend"] = False
    df["buy"] = False
    df["sell"] = False
    for current in range(2, len(df.index)):
        previous = current - 1
        before = current - 2
        EMAFAST = df["ema"][current]
        LINREG = df["subhag"][current]
        ALPHATREND = df["alphatrend"][before]
        clohi = max(EMAFAST, LINREG, ALPHATREND)
        clolo = min(EMAFAST, LINREG, ALPHATREND)
        # CloudMA := (bull > bear) ? clolo < nz(CloudMA[1]) ? nz(CloudMA[1]) : clolo :
        if df["cmpbull"][current] > df["cmpbear"][current]:
            if clolo < (
                df["vxma"][previous] if df["vxma"][previous] != None else 0
            ):
                df["vxma"][current] = (
                    df["vxma"][previous] if df["vxma"][previous] != None else 0
                )
            else:
                df["vxma"][current] = clolo
        #  (bear > bull) ? clohi > nz(CloudMA[1]) ? nz(CloudMA[1]) : clohi : nz(CloudMA[1])
        elif df["cmpbull"][current] < df["cmpbear"][current]:
            if clohi > (
                df["vxma"][previous] if df["vxma"][previous] != None else 0
            ):
                df["vxma"][current] = (
                    df["vxma"][previous] if df["vxma"][previous] != None else 0
                )
            else:
                df["vxma"][current] = clohi
        else:
            df["vxma"][current] = (
                df["vxma"][previous] if df["vxma"][previous] != None else 0
            )
        # Get trend True = Bull False = Bear
        if (
            df["vxma"][current] > df["vxma"][previous]
            and df["vxma"][previous] > df["vxma"][before]
        ):
            df["trend"][current] = True
        elif (
            df["vxma"][current] < df["vxma"][previous]
            and df["vxma"][previous] < df["vxma"][before]
        ):
            df["trend"][current] = False
        else:
            df["trend"][current] = df["trend"][previous]
        # get zone
        if df["trend"][current] and not df["trend"][previous]:
            df["buy"][current] = True
            df["sell"][current] = False
        elif not df["trend"][current] and df["trend"][previous]:
            df["buy"][current] = False
            df["sell"][current] = True
        else:
            df["buy"][current] = False
            df["sell"][current] = False
    return df


def indicator(ema, linear, smooth, atr_p, atr_m, AOL, rsi):
    global df
    df["ema"] = ta.ema(df["Close"], ema)
    df["subhag"] = ta.ema(ta.linreg(df["Close"], linear, 0), smooth)
    alphatrend(df, atr_p, atr_m, rsi)
    andean(df, AOL)
    vxma(df)
    print(df.tail(10))
    df.drop(columns=["ema", "subhag", "atr", "up2"], axis=1, inplace=True)
    df.drop(columns=["alphatrend", "dn1"], axis=1, inplace=True)
    df.drop(columns=["cmpbull", "cmpbear", "up1", "dn2"], axis=1, inplace=True)
    return df.vxma


def signalbuy():
    trade = df.buy
    return trade


def signalsell():
    trade = df.sell
    return trade


class run_bot(Strategy):
    ema = 30
    linear = 30
    smooth = 30
    atr_p = 12
    atr_m = 1.6
    AOL = 30
    rsi = 25

    def init(self):
        super().init()
        self.A2 = self.I(
            indicator,
            self.ema,
            self.linear,
            self.smooth,
            self.atr_p,
            self.atr_m,
            self.AOL,
            self.rsi,
        )
        self.A0 = self.I(signalbuy)
        self.A1 = self.I(signalsell)

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
# ema = range(1,200,2),
# linear = range(1,200,2),
# smooth = range(1,200,2),
# atr_p = range(1,50,1),
# AOL =   range(1,200,2),
# rsi =   range(1,50,1),
# maximize = 'Win Rate [%]')
print(stat)
bt.plot()
