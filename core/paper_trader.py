from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from data_handler import DataHandler
from trading_lstm import TradingLTSM


class PaperTrader:
    def __init__(self, model: TradingLTSM, data_handler: DataHandler, ticker="AAPL"):

        self.model = model
        self.data_handler = data_handler
        self.ticker = ticker
        self.trading_client = TradingClient(
            "PKSAFILHAEV7QE23XVUGLI4DU3",
            "FPJtCknW6dzYPjShCGijomw7n6Ggaj6CxizfWeKKEMza",
            paper=True,
        )

        account = self.trading_client.get_account
        print(f"Cash: ${account.cash}")
        print(f"Portfolio Value: ${account.portfolio_value}")

    def get_position(self):
        try:
            pos = self.api.get_position(self.ticker)
            return int(pos.qty)
        except:
            return 0  # No position

    def open_position(self):
        """Runs at market open, predict and place order"""
        clock = self.api.get_clock()
        if not clock.is_open:
            print("Market is closed — skipping")
            return

        # Get prediction from yesterday's data
        try:
            pred, data_date = self.model.get_yesterdays_prediction(
                self.ticker, self.model, self.scaler, self.feature_cols
            )
        except ValueError as e:
            print(f"Prediction error: {e}")
            return

    def execute_trade(self):
        clock = self.trading_client.get_clock()
        if not clock.is_open:
            print
