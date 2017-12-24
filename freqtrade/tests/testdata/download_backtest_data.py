#!/usr/bin/env python3

"""This script generate json data from bittrex"""
import json
from os import path

from freqtrade import exchange
from freqtrade.exchange import Bittrex

PAIRS = [
    "BTC_ETH",
    "BTC_LTC",
    "BTC_POWR",
    "BTC_FTC",
    "BTC_STORJ",
    "BTC_OK",
    "BTC_TIX",
    "BTC_XVG",
    "BTC_PIVX",
    "BTC_ADA",
    "BTC_XLM",
    "BTC_NXT",
    "BTC_OMG",
    "BTC_WAVES",
    "BTC_XRP",
    "BTC_QTUM",
    "BTC_MCO",
    "BTC_NEO",
    "BTC_FLO",
    "BTC_VOX",
    "BTC_ZEC",
    "BTC_SALT",
    "BTC_XZC",
    "BTC_XMR",
"BTC_ETH",
                "BTC_LTC",
                "BTC_ETC",
                "BTC_DASH",
                "BTC_ZEC",
                "BTC_XLM",
                "BTC_NXT",
                "BTC_POWR",
                "BTC_ADA",
                "BTC_XMR"
]
TICKER_INTERVAL = 1  # ticker interval in minutes (currently implemented: 1 and 5)
OUTPUT_DIR = path.dirname(path.realpath(__file__))

# Init Bittrex exchange
exchange._API = Bittrex({"key": "6a4d4a295cc440649303c9c923c94237", "secret": "60740f0e82694e6895bf7df5e3928bce"})

for pair in PAIRS:
    data = exchange.get_ticker_history(pair, TICKER_INTERVAL)
    filename = path.join(OUTPUT_DIR, '{}-{}.json'.format(
        pair,
        TICKER_INTERVAL,
    ))
    with open(filename, 'w') as fp:
        json.dump(data, fp)
