import yfinance as yf
import pandas as pd
import numpy as np


class DataHandler:
    def __init__(self, ticker, period="1y", interval="1h"):
        self.ticker = ticker
        self.period = period
        self.interval = interval
        self.df = self.fetch_data()
        df = self.add_features(self.df)
        self.df = df

    def get_latest_data(self):
        self.df = self.fetch_data()
        self.add_features(self.df)
        return self.df

    def fetch_data(self):
        """
        Downloads information from yfinance
        """
        df = yf.download(self.ticker, period=self.period, interval=self.interval)
        df.dropna(inplace=True)
        df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df.columns = ["Datetime", "Close", "High", "Low", "Open", "Volume"]
        df["Date"] = df["Datetime"].dt.date
        df["Hour"] = df["Datetime"].dt.hour
        return pd.DataFrame(df)

    def add_features(self, df):
        """
        Adds technical indicators as features
        """

        day_open = df.groupby("Date")["Open"].first().rename("DayOpen")
        df = df.merge(day_open, on="Date", how="left")
        df["LogDayOpen"] = np.log(df["DayOpen"])

        eod_close = df.groupby("Date")["Close"].last().rename("EODClose")
        df = df.merge(eod_close, on="Date", how="left")
        df["LogEODClose"] = np.log(df["EODClose"])

        daily_close = eod_close.shift(1).rename("PrevEODClose")
        df = df.merge(daily_close, on="Date", how="left")
        df["LogPrevEODClose"] = np.log(df["PrevEODClose"])

        df["Label"] = (df["EODClose"] > df["PrevEODClose"]).astype(int)

        df["SimpleReturn"] = df["Close"].pct_change()
        df["SMA_20"] = df["Close"].rolling(20).mean()
        df["SMA_50"] = df["Close"].rolling(50).mean()
        df["RSI"] = self._compute_rsi(df["Close"])
        df["Volatility"] = df["SimpleReturn"].rolling(20).std()

        df["LogClose"] = np.log(df["Close"])
        df["LogReturn"] = df["Close"].diff()

        df["RangePct"] = (df["High"] - df["Low"]) / df["Close"]
        df["CloseOpenPct"] = (df["Close"] - df["Open"]) / df["Open"]

        for lag in [1, 2, 3, 5, 10, 21]:
            df[f"RetLag{lag}"] = df["LogReturn"].shift(lag)

        for win in [5, 10, 21, 63]:
            df[f"RollMeanRet{win}"] = df["LogReturn"].rolling(win).mean()
            df[f"RollStdRet{win}"] = df["LogReturn"].rolling(win).std()
            df[f"PriceMom{win}"] = np.log(df["Close"] / df["Close"].shift(win))

        # Moving-average structure
        df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
        df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = df["EMA12"] - df["EMA26"]
        df["MACDSignal"] = df["MACD"].ewm(span=9, adjust=False).mean()

        # RSI(14)
        delta = df["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["RSI14"] = 100 - (100 / (1 + rs))

        # ATR(14)
        hl = df["High"] - df["Low"]
        hpc = (df["High"] - df["Close"].shift(1)).abs()
        lpc = (df["Low"] - df["Close"].shift(1)).abs()
        tr = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
        df["ATR14"] = tr.rolling(14).mean()
        df["ATR14Pct"] = df["ATR14"] / df["Close"]

        # Calendar features (cyclical)
        df["DayOfWeek"] = df["Datetime"].dt.dayofweek
        df["Month"] = df["Datetime"].dt.month
        df["HourSin"] = np.sin(2 * np.pi * df["Hour"] / 24)
        df["HourCos"] = np.cos(2 * np.pi * df["Hour"] / 24)
        df["DoWSin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 5)
        df["DoWCos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 5)
        df["MonthSin"] = np.sin(2 * np.pi * df["Month"] / 12)
        df["MonthCos"] = np.cos(2 * np.pi * df["Month"] / 12)

        #df.dropna(inplace=True)
        return df

    def _compute_rsi(self, series, period=14):
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = -delta.clip(upper=0).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
