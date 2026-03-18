import torch
import torch.nn as nn
import numpy as np
import pytz
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from core.data_handler import DataHandler


class TradingLTSM(nn.Module):
    def __init__(
        self,
        hidden_size=64,
        num_layers=2,
        output_size=1,
        dropout=0.2,
        feature_cols=[],
    ):
        super().__init__()
        self.feature_cols = feature_cols
        input_size = len(self.feature_cols)
        print(f"Input size: {input_size}")
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)  # Last timestep
        return self.sigmoid(out)  # Output: probability of price going up

    def prepare_daily_sequences(self, df):
        """
        Uses the ticker data and features and scales them
        Labels data if next day close > todays close
        Returns two tensor objects
        """
        X, y, dates = [], [], []

        all_dates = sorted(df["Date"].unique())

        for i in range(len(all_dates) - 1):

            yesterday = all_dates[i]
            today = all_dates[i + 1]

            yesterday_df = df[df["Date"] == yesterday].sort_values("Hour")

            if len(yesterday_df) < 6:
                continue

            features = yesterday_df[self.feature_cols].values
            # Normalize
            self.scaler = StandardScaler()
            features = self.scaler.fit_transform(features)
            label = df[df["Date"] == today]["Label"].iloc[-1]
            X.append(features)
            y.append(label)
            dates.append(today)

        # Pad sequences to same length (in case some days have missing bars)
        max_len = max(len(x) for x in X)
        X_padded = np.zeros((len(X), max_len, len(self.feature_cols)))
        for i, x in enumerate(X):
            X_padded[i, : len(x), :] = x

        X_tensor = torch.tensor(X_padded, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        return X_tensor, y_tensor, dates, self.scaler

    def train_model(self, X_train, y_train, epochs=50, lr=0.001):
        """Trains model using an Adam optimizer, with BCE loss criterion"""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.BCELoss()

        self.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            preds = self(X_train).squeeze()
            loss = criterion(preds, y_train.squeeze())
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

        return

    def get_yesterdays_prediction(self, yf_ticker, scaler):
        """
        Fetch yesterdays hourly ticker bars from yfinance and make a prediction

        """

        data_handler = DataHandler(yf_ticker, period="14d", interval="1h")
        df = data_handler.df

        available_dates = sorted(df["Date"].unique())

        ET = pytz.timezone("America/New_York")
        now_et = datetime.now(ET)
        today_et = now_et.date()

        # Sunday
        if today_et.weekday() == 6:
            yesterday_et = today_et - timedelta(days=2)
        # Monday
        elif today_et.weekday() == 0: 
            yesterday_et = today_et - timedelta(days=3)
        else:
            yesterday_et = today_et - timedelta(days=1)

        if len(available_dates) < 2:
            raise ValueError("Not enough data — need at least 2 days")
        
        if yesterday_et not in available_dates:
            print(f"Warning — {yesterday_et} not in data, falling back to {available_dates[-1]}")
            yesterday_et = available_dates[-1]

        yesterday_df = df[df["Date"] == yesterday_et].sort_values("Datetime")

        if len(yesterday_df) < 6:
            raise ValueError(f"Too few bars for {yesterday_et} ({len(yesterday_df)}) — skipping")

        # Scale using the SAME scaler fitted during training
        features = yesterday_df[self.feature_cols].values
        features = scaler.transform(features)

        # Build tensor: (1, timesteps, features)
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        self.eval()
        with torch.no_grad():
            pred = self(x).squeeze().item()

        return pred, yesterday_et
