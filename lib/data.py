import os
import csv
import glob
import numpy as np
import collections
import pandas as pd

# Hold the prices for the small data frame
Prices = collections.namedtuple(
    "Prices", field_names=["open", "high", "low", "close", "volume"]
)

# Hold the prices for the large data frame
PricesL = collections.namedtuple(
    "Prices",
    field_names=[
        "open",
        "high",
        "low",
        "close",
        "volume",
        "vwap",
        "histogram",
        "macd",
        "signal",
        "rsi",
        "bbands",
        "ma10",
        "ma20",
        "ma50",
    ],
)

# Helper function for creating the Prices-tuple
def get_tuple_from_df(df, large=False):
    if large:

        for i in range(10, 20, 50):
            df[f"ma{i}"] = df.rolling(window=i)["close"].mean()

        df = df.dropna()

        return PricesL(
            open=np.array(df["open"]),
            high=np.array(df["high"]),
            low=np.array(df["low"]),
            close=np.array(df["close"]),
            volume=np.array(df["Volume"]),
            vwap=np.array(df["VWAP"]),
            histogram=np.array(df["Histogram"]),
            macd=np.array(df["MACD"]),
            signal=np.array(df["Signal"]),
            rsi=np.array(df["RSI"]),
            bbands=np.array(df["Bollinger Bands %B"]),
            ma10=np.array(df["ma10"]),
            ma20=np.array(df["ma10"]),
            ma50=np.array(df["ma10"]),
        )

    return Prices(
        open=np.array(df["open"]),
        high=np.array(df["high"]),
        low=np.array(df["low"]),
        close=np.array(df["close"]),
        volume=np.array(df["Volume"]),
    )


def prices_to_relative(prices, normalize_volume=False, large=False):
    """
    Convert prices to relative in respect to open price
    :param ochl: tuple with open, close, high, low
    :return: tuple with open, rel_close, rel_high, rel_low
    """
    if large:
        assert isinstance(prices, PricesL)
    else:
        assert isinstance(prices, Prices)

    rh = (prices.high - prices.open) / prices.open
    rl = (prices.low - prices.open) / prices.open
    rc = (prices.close - prices.open) / prices.open

    volume = prices.volume
    if normalize_volume:
        vol_cpy = prices.volume.copy()
        vol_cpy[0] = 0

        for i, num in enumerate(volume[:-1], start=1):
            d = volume[i] - num
            if d == 0:
                vol_cpy[i] = 0
            elif num == 0:
                vol_cpy[i] = 1
            else:
                vol_cpy[i] = d / num

        volume = vol_cpy
    if not large:
        return Prices(open=prices.open, high=rh, low=rl, close=rc, volume=volume)

    macd = prices.macd / prices.open

    # Normalize VWAP
    vwap = prices.vwap
    vwap_cpy = prices.vwap.copy()
    vwap_cpy[0] = 0

    for i, num in enumerate(vwap[:-1], start=1):
        d = vwap[i] - num
        if d == 0:
            vwap_cpy[i] = 0
        elif num == 0:
            vwap_cpy[i] = 1
        else:
            vwap_cpy[i] = d / num

    vwap = vwap_cpy

    ma10 = (prices.ma10 - prices.open) / prices.open
    ma20 = (prices.ma20 - prices.open) / prices.open
    ma50 = (prices.ma50 - prices.open) / prices.open

    return PricesL(
        open=prices.open,
        high=rh,
        low=rl,
        close=rc,
        volume=volume,
        macd=macd,
        vwap=vwap,
        rsi=prices.rsi / 100.0,
        bbands=prices.bbands,
        signal=prices.signal / 100.0,
        histogram=prices.histogram,
        ma10=ma10,
        ma20=ma20,
        ma50=ma50,
    )


def get_data_as_dict(paths: dict, normalize_volume=True, large=False, get_dfs=False):
    data_dict = {}
    dfs_dict = {}

    for pair, path in paths.items():
        df = pd.read_csv(path)
        if get_dfs:
            dfs_dict[pair] = df

        data_tuple = get_tuple_from_df(df, large=large)
        data_dict[pair] = prices_to_relative(
            data_tuple, normalize_volume=normalize_volume, large=large
        )
    if get_dfs:
        return data_dict, dfs_dict
    return data_dict
