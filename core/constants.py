import os
from dotenv import load_dotenv

load_dotenv()

ALPACA_KEY = os.environ["ALPACA_KEY"]
ALPACA_SECRET = os.environ["ALPACA_SECRET"]
HIDDEN_SIZE = 64
NUM_LAYERS = 3
DROPOUT = 0.15
CASH_FRACTION = 0.2
THRESHOLD = 0.6

FEATURES = [
    "Close",
    "High",
    "Low",
    "Open",
    "Volume",
    "DayOpen",
    "LogDayOpen",
    "EODClose",
    "LogEODClose",
    "PrevEODClose",
    "LogPrevEODClose",
    "SimpleReturn",
    "SMA_20",
    "SMA_50",
    "RSI",
    "Volatility",
    "LogClose",
    "LogReturn",
    "RangePct",
    "CloseOpenPct",
    "RetLag1",
    "RetLag2",
    "RetLag3",
    "RetLag5",
    "RetLag10",
    "RetLag21",
    "RollMeanRet5",
    "RollStdRet5",
    "PriceMom5",
    "RollMeanRet10",
    "RollStdRet10",
    "PriceMom10",
    "RollMeanRet21",
    "RollStdRet21",
    "PriceMom21",
    "RollMeanRet63",
    "RollStdRet63",
    "PriceMom63",
    "EMA12",
    "EMA26",
    "MACD",
    "MACDSignal",
    "RSI14",
    "ATR14",
    "ATR14Pct",
    "DayOfWeek",
    "Month",
    "HourSin",
    "HourCos",
    "DoWSin",
    "DoWCos",
    "MonthSin",
    "MonthCos",
    "SentimentMean",
    "SentimentWeighted",
    "SentimentRolling5",
    "SentimentDelta",
    "ArticleCount",
]


class AAPLConstants:
    HIDDEN_SIZE = 64
    NUM_LAYERS = 3
    DROPOUT = 0.15


class MSFTConstants:
    HIDDEN_SIZE = 64
    NUM_LAYERS = 3
    DROPOUT = 0.15
