#!/usr/bin/env python3

import sys
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from pandas import DataFrame
from freqtrade import exchange, analyze
from freqtrade.misc import common_args_parser
from freqtrade.strategy.strategy import Strategy
import matplotlib.pyplot as plt
import matplotlib  # Install PYQT5 manually if you want to test this helper function
from matplotlib.finance import candlestick2_ohlc
from matplotlib.lines import Line2D

matplotlib.use("Qt5Agg")


def plot_parse_args(args):
    parser = common_args_parser(description='Graph utility')
    parser.add_argument(
        '-p', '--pair',
        help='What currency pair',
        dest='pair',
        default='BTC_ETH',
        type=str,
    )
    parser.add_argument(
        '-i', '--interval',
        help='what interval to use',
        dest='interval',
        default=5,
        type=int,
    )
    return parser.parse_args(args)


def plot_analyzed_dataframe(args) -> None:
    """
    Calls analyze() and plots the returned dataframe
    :param pair: pair as str
    :return: None
    """
    # Init strategy
    strategy = Strategy()
    strategy.init({'strategy': args.strategy})

    # Init Bittrex to use public API
    exchange._API = exchange.Bittrex({'key': '', 'secret': ''})
    ticker = exchange.get_ticker_history(args.pair, args.interval)
    dataframe = analyze.analyze_ticker(ticker)

    dataframe = populate_indicator(dataframe)

    # Two subplots sharing x axis
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((4, 1), (2, 0), sharex=ax1)
    ax3 = plt.subplot2grid((4, 1), (3, 0), sharex=ax1)
    fig.suptitle(args.pair + " " + str(args.interval), fontsize=14, fontweight='bold')
    # ax1.plot(dataframe.index.values, dataframe['close'], label='close')
    # setup marker styles
    buy_marker = dict(linestyle=":", color='green', markersize=8)
    sell_marker = dict(linestyle=":", color='red', markersize=8)
    ax1.plot(dataframe.index.values, dataframe['sell_price'], 'v', **sell_marker, label='sell')
    ax1.plot(dataframe.index.values, dataframe['buy_price'], '^', **buy_marker, label='buy')
    ax1.plot(dataframe.index.values, dataframe['bb_lowerband'], ':', label='blower')

    candlestick2_ohlc(ax1, dataframe['open'], dataframe['high'], dataframe['low'], dataframe['close'], width=0.4,
                      colorup='g', colordown='r', alpha=1)
    ax1.legend()

    ax2.plot(dataframe.index.values, dataframe['adx'], label='ADX')
    ax2.plot(dataframe.index.values, dataframe['mfi'], label='MFI')
    ax2.plot(dataframe.index.values, dataframe['rsi'], label='RSI')
    # ax2.plot(dataframe.index.values, [25] * len(dataframe.index.values))
    ax2.legend()

    # ax3.plot(dataframe.index.values, [20] * len(dataframe.index.values))
    ax3.plot(dataframe.index.values, dataframe['direction'], '--', label="direction")
    # ax3.plot(dataframe.index.values, dataframe['emarsi'], label="emarsi")
    ax3.legend()

    # Fine-tune figure; make subplots close to each other and hide x ticks for
    # all but bottom plot.
    fig.subplots_adjust(left=0.13, bottom=0.07, right=0.98, top=0.90, wspace=0.2, hspace=0)
    ax1.autoscale_view()
    ax2.autoscale_view()
    ax3.autoscale_view()
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False, rotation=45)

    plt.show()


def populate_indicator(dataframe: DataFrame) -> DataFrame:
    dataframe.loc[dataframe['buy'] == 1, 'buy_price'] = dataframe['close']
    dataframe.loc[dataframe['sell'] == 1, 'sell_price'] = dataframe['close']

    # ADX
    if 'adx' not in dataframe:
        dataframe['adx'] = ta.ADX(dataframe)

    # Bollinger bands
    if 'bb_lowerband' not in dataframe:
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']

    # Stoch fast
    if 'fastd' not in dataframe or 'fastk' not in dataframe:
        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']

    # MFI
    if 'mfi' not in dataframe:
        dataframe['mfi'] = ta.MFI(dataframe)

    # SMA - Simple Moving Average
    if 'sma' not in dataframe:
        dataframe['sma'] = ta.SMA(dataframe, timeperiod=40)

    # TEMA - Triple Exponential Moving Average
    if 'tema' not in dataframe:
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)

    return dataframe


if __name__ == '__main__':
    args = plot_parse_args(sys.argv[1:])
    plot_analyzed_dataframe(args)
