"""
Functions to analyze ticker data with indicators and produce buy and sell signals
"""
import logging
from datetime import timedelta
from enum import Enum
from typing import List, Dict

import arrow
import talib.abstract as ta
from pandas import DataFrame, to_datetime
import numpy

from freqtrade.exchange import get_ticker_history
from freqtrade.vendor.qtpylib.indicators import awesome_oscillator, crossed_above, crossed_below

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """ Enum to distinguish between buy and sell signals """
    BUY = "buy"
    SELL = "sell"


def parse_ticker_dataframe(ticker: list) -> DataFrame:
    """
    Analyses the trend for the given ticker history
    :param ticker: See exchange.get_ticker_history
    :return: DataFrame
    """
    columns = {'C': 'close', 'V': 'volume', 'O': 'open', 'H': 'high', 'L': 'low', 'T': 'date'}
    frame = DataFrame(ticker) \
        .drop('BV', 1) \
        .rename(columns=columns)
    frame['date'] = to_datetime(frame['date'], utc=True, infer_datetime_format=True)
    frame.sort_values('date', inplace=True)
    return frame


def populate_indicators(dataframe: DataFrame) -> DataFrame:
    """
    Adds several different TA indicators to the given DataFrame
    """
    dataframe['sar'] = ta.SAR(dataframe)
    dataframe['adx'] = ta.ADX(dataframe)
    stoch = ta.STOCHF(dataframe)
    dataframe['fastd'] = stoch['fastd']
    dataframe['fastk'] = stoch['fastk']
    dataframe['blower'] = ta.BBANDS(dataframe, nbdevup=2, nbdevdn=2)['lowerband']
    dataframe['sma'] = ta.SMA(dataframe, timeperiod=40)

    dataframe['sma200'] = ta.SMA(dataframe, timeperiod=200)
    dataframe['sma5'] = ta.SMA(dataframe, timeperiod=5)
    dataframe['sma'] = ta.SMA(dataframe, timeperiod=40)
    dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)
    dataframe['mfi'] = ta.MFI(dataframe)
    dataframe['rsi'] = ta.RSI(dataframe)
    dataframe['plus_dm'] = ta.PLUS_DM(dataframe)
    dataframe['plus_di'] = ta.PLUS_DI(dataframe)
    dataframe['minus_dm'] = ta.MINUS_DM(dataframe)
    dataframe['minus_di'] = ta.MINUS_DI(dataframe)
    dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
    dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)
    dataframe['ema21'] = ta.EMA(dataframe, timeperiod=21)
    dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
    dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)
    dataframe['ema200'] = ta.EMA(dataframe, timeperiod=200)
    dataframe['ao'] = awesome_oscillator(dataframe)
    macd = ta.MACD(dataframe)
    dataframe['macd'] = macd['macd']
    dataframe['macdsignal'] = macd['macdsignal']
    dataframe['macdhist'] = macd['macdhist']
    hilbert = ta.HT_SINE(dataframe)
    dataframe['htsine'] = hilbert['sine']
    dataframe['htleadsine'] = hilbert['leadsine']
    dataframe['plus_dm'] = ta.PLUS_DM(dataframe)
    dataframe['plus_di'] = ta.PLUS_DI(dataframe)
    dataframe['minus_dm'] = ta.MINUS_DM(dataframe)
    dataframe['minus_di'] = ta.MINUS_DI(dataframe)
    dataframe['cci'] = ta.CCI(dataframe, timeperiod=20)

    rsi = .1 * (dataframe['rsi'] - 50)
    dataframe['fishrsi'] = (numpy.exp(2 * rsi) - 1) / (numpy.exp(2 * rsi) + 1)
    return dataframe


def populate_buy_trend(dataframe: DataFrame) -> DataFrame:
    """
    Based on TA indicators, populates the buy signal for the given dataframe
    :param dataframe: DataFrame
    :return: DataFrame with buy column
    """
    dataframe['fastk-previous'] = dataframe.fastk.shift(10)
    dataframe.loc[
#   v1
#         (dataframe['rsi'] < 31) &
#         (dataframe['fastd'] < 28) &
#         (dataframe['ao'] < 40) &
#         (dataframe['adx'] > 28) &
        # (dataframe['ema50'] > dataframe['ema200']) &
        # (dataframe['plus_di'] > 0.5) &

#   v2
#         (
#             (dataframe['rsi'] < 32) &
#             (dataframe['fastd'] < 28) &
#             (dataframe['adx'] > 28) &
#             (dataframe['plus_di'] > 0.5)
#         ) |
#         (
#             (dataframe['adx'] > 65) &
#             (dataframe['plus_di'] > 0.5)
#         ),
#         'buy'] = 1

        # (dataframe['close'] < dataframe['sma']) &
        # (dataframe['fastd'] > dataframe['fastk']) &
        # (dataframe['rsi'] > 0) &
        # (dataframe['fastd'] > 0) &
        # (dataframe['fishrsi'] < -0.54),
        # v3
        # (dataframe['rsi'] < 35) &
        # (dataframe['close'] < dataframe['sma']) &
        # (dataframe['fishrsi'] < -0.54) &
        # (dataframe['mfi'] < 17.0) &
        # (dataframe['ema50'] > dataframe['ema100']) &
        # (dataframe['volume'] > 2500) &

        # v3
        # (dataframe['rsi'] < 35) &
        # (dataframe['close'] < dataframe['sma']) &

        # (dataframe['mfi'] < 17.0) &
        # # (dataframe['fastk-previous'] > 10) &
        # (dataframe['adx'] > 65) &
        # (dataframe['plus_di'] > 0.5),

        # v4
        # (crossed_below(dataframe['rsi'], 30)) &
        # (dataframe['fishrsi'] < -0.54) &
        # (crossed_above(dataframe['cci'], -100)) &
        # (dataframe['ema10'] >= dataframe['ema21']),
        # (dataframe['mfi'] < 17.0) &
        # (dataframe['macd'] < 0),
        # (dataframe['fastk-previous'] > 10) &
        # (dataframe['adx'] > 65),

        # v5
        # (dataframe['close'] >= dataframe['blower']),
        # (
        #     # (crossed_above(dataframe['cci'], -100)) &
        #     (dataframe['fastd'] < 44) &
        #     (dataframe['rsi'] < 34) &
        # (crossed_above(dataframe['ema50'], dataframe['ema100'])),
        # ),

        # (dataframe['close'] > dataframe['sma200']) &
        # (dataframe['close'] < dataframe['sma5']) &
        # (dataframe['rsi'] < 10) &
        # (dataframe['close'] >= dataframe['blower']) &
        # v6
        # (
        #     (dataframe['adx'] > 20) &
        #     (dataframe['minus_dm'] > 0)
        # ) &
        # (
        #     (crossed_above(dataframe['ema10'], dataframe['ema21'])) |
        #     (crossed_above(dataframe['macd'], dataframe['macdsignal']))
        # ),

        # v7
        #     {
        #         "adx": 0,
        #         "fastd": 1,
        #         "fastd-value": 48.0,
        #         "green_candle": 1,
        #         "mfi": 1,
        #         "mfi-value": 23.0,
        #         "over_sar": 1,
        #         "rsi": 1,
        #         "rsi-value": 21.0,
        #         "trigger": 4,
        #         "uptrend_long_ema": 1,
        #         "uptrend_short_ema": 1,
        #         "uptrend_sma": 0
        #     }
        # 2017 - 12 - 24
        # Result: Made 830 buys.  Average profit 0.09 %.  Total profit was 0.00711045 BTC.Average duration 34.5 mins.

        (dataframe['fastd'] < 48) &
        (dataframe['close'] > dataframe['open']) &
        (dataframe['mfi'] < 23) &
        (dataframe['rsi'] < 21) &
        (dataframe['close'] > dataframe['sar']) &
        (dataframe['ema50'] > dataframe['ema100']) &
        # (crossed_above(dataframe['macd'], dataframe['macdsignal'])) &
        (dataframe['ema5'] > dataframe['ema10']),
    'buy'] = 1
    return dataframe


def populate_sell_trend(dataframe: DataFrame) -> DataFrame:
    """
    Based on TA indicators, populates the sell signal for the given dataframe
    :param dataframe: DataFrame
    :return: DataFrame with buy column
    """
    dataframe.loc[
        # v1
        # (
        #   (crossed_above(dataframe['rsi'], 85)) |
        #   (crossed_above(dataframe['fastd'], 80)) |
        #   (crossed_above(dataframe['ao'], 82))
        # ) &
        # (dataframe['macd'] < 0) &
        # (dataframe['minus_dm'] > 0) &
        # (dataframe['plus_di'] > 0) &
        # (dataframe['adx'] < 1),

        # v2
        # (
        #     (dataframe['sar'] > dataframe['close']) &
        #     # (dataframe['fishrsi'] > 0.2) &
        #     (crossed_above(dataframe['rsi'], 70))
        # ) |
        # (
        #     (dataframe['adx'] > 71) &
        #     (dataframe['minus_di'] > 0.5)
        # ),

        #  v3
        # (crossed_above(dataframe['rsi'], 85)) &
        # (crossed_above(dataframe['macd'],dataframe['macdsignal'])) &
        # (dataframe['sar'] > dataframe['close']) &
        # (dataframe['fishrsi'] > 0.3),
        # v4
        (
            (
                (crossed_above(dataframe['rsi'], 85)) |
                (crossed_above(dataframe['fastd'], 79))
            ) &
            (dataframe['adx'] > 15) &
            (dataframe['minus_di'] > 0)
        ) |
        (
            (crossed_above(dataframe['rsi'], 50)) &
            (dataframe['macd'] < 10) &
            (dataframe['minus_di'] > 0)
        ),
        # v5
        # (
        #   (crossed_above(dataframe['rsi'], 73)) |
          # (crossed_above(dataframe['fastd'], 80))
        # ) &
        # (crossed_below(dataframe['ema10'], dataframe['ema21'])),

        # v6
        # (
        #         (
        #                 (crossed_above(dataframe['rsi'], 70)) |
        #                 (crossed_above(dataframe['fastd'], 70))
        #         ) &
        #         (dataframe['adx'] > 10) &
        #         (dataframe['minus_di'] > 0)
        # ) |
        # (
        #         (dataframe['adx'] > 70) &
        #         (dataframe['minus_di'] > 0.5)
        # ),

        'sell'] = 1
    return dataframe


def analyze_ticker(ticker_history: List[Dict]) -> DataFrame:
    """
    Parses the given ticker history and returns a populated DataFrame
    add several TA indicators and buy signal to it
    :return DataFrame with ticker data and indicator data
    """
    dataframe = parse_ticker_dataframe(ticker_history)
    dataframe = populate_indicators(dataframe)
    dataframe = populate_buy_trend(dataframe)
    dataframe = populate_sell_trend(dataframe)
    return dataframe


def get_signal(pair: str, signal: SignalType) -> bool:
    """
    Calculates current signal based several technical analysis indicators
    :param pair: pair in format BTC_ANT or BTC-ANT
    :return: True if pair is good for buying, False otherwise
    """
    ticker_hist = get_ticker_history(pair)
    if not ticker_hist:
        logger.warning('Empty ticker history for pair %s', pair)
        return False

    try:
        dataframe = analyze_ticker(ticker_hist)
    except ValueError as ex:
        logger.warning('Unable to analyze ticker for pair %s: %s', pair, str(ex))
        return False
    except Exception:
        logger.exception('Unexpected error when analyzing ticker for pair %s.', pair)
        return False

    if dataframe.empty:
        return False

    latest = dataframe.iloc[-1]

    # Check if dataframe is out of date
    signal_date = arrow.get(latest['date'])
    if signal_date < arrow.now() - timedelta(minutes=10):
        return False

    result = latest[signal.value] == 1
    logger.debug('%s_trigger: %s (pair=%s, signal=%s)', signal.value, latest['date'], pair, result)
    return result
