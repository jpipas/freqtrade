"""
File that contains the configuration for Hyperopt
"""


def hyperopt_optimize_conf() -> dict:
    """
    This function is used to define which parameters Hyperopt must used.
    The "pair_whitelist" is only used is your are using Hyperopt with MongoDB,
    without MongoDB, Hyperopt will use the pair your have set in your config file.
    :return:
    """
    return {
        'max_open_trades': 5,
        'stake_currency': 'BTC',
        'stake_amount': 0.001,
        "minimal_roi": {
            "20": 0.000,
            "15": 0.025,
            "10": 0.03,
            "0": 0.045
        },
        'stoploss': -0.15,
        "bid_strategy": {
            "ask_last_balance": 0.0
        },
        "exchange": {
            "pair_whitelist": [
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
                "BTC_XMR"
            ]
        }
    }
