# pragma pylint: disable=missing-docstring
import json
import os


def load_backtesting_data(ticker_interval: int = 1):
    path = os.path.abspath(os.path.dirname(__file__))
    result = {}
    pairs = [
        "BTC_VTC", "BTC_ARK",
        "BTC_FTC", "BTC_STORJ", "BTC_NXT", "BTC_OK",
        "BTC_TIX", "BTC_ADA", "BTC_GNT", "BTC_XLM"
    ]
    for pair in pairs:
        with open('{abspath}/testdata/{pair}-{ticker_interval}.json'.format(
            abspath=path,
            pair=pair,
            ticker_interval=ticker_interval,
        )) as tickerdata:
            result[pair] = json.load(tickerdata)
    return result
