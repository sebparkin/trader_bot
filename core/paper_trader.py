from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
import pandas as pd
import schedule, time
from datetime import datetime
import pytz
from core.trading_lstm import TradingLTSM

ET = pytz.timezone("America/New_York")


class PaperTrader:
    def __init__(self, model: TradingLTSM, ticker="AAPL", cash_fraction=0.2, threshold=0.6):

        self.model = model
        self.ticker = ticker
        self.cash_fraction = cash_fraction
        self.threshold = threshold
        self.trading_client = TradingClient(
            "PKSAFILHAEV7QE23XVUGLI4DU3",
            "FPJtCknW6dzYPjShCGijomw7n6Ggaj6CxizfWeKKEMza",
            paper=True,
        )
        self.stock_client = StockHistoricalDataClient(
            "PKSAFILHAEV7QE23XVUGLI4DU3", "FPJtCknW6dzYPjShCGijomw7n6Ggaj6CxizfWeKKEMza"
        )

        account = self.trading_client.get_account()
        print(f"Cash: ${account.cash}")
        print(f"Portfolio Value: ${account.portfolio_value}")

    def get_position(self):
        try:
            pos = self.trading_client.get_open_position(self.ticker)
            return pos.qty
        except:
            return 0  # No position

    def open_position(self, scaler):
        """Runs at market open, predict and place order"""
        clock = self.trading_client.get_clock()
        if not clock.is_open:
            print("Market is closed — skipping")
            return

        # Get prediction from yesterday's data
        try:
            pred, data_date = self.model.get_yesterdays_prediction(self.ticker, scaler)
        except ValueError as e:
            print(f"Prediction error: {e}")
            return

        account = self.trading_client.get_account()

        request_param = StockLatestQuoteRequest(symbol_or_symbols=self.ticker)
        latest = self.stock_client.get_stock_latest_quote(request_param)
        price = float(latest[self.ticker].ask_price)
        cash = float(account.cash)
        portfolio = float(account.portfolio_value)

        if self.get_position() != 0:
            print("Already in a position — skipping open")
            return

        qty = round(cash * self.cash_fraction / price)  # Trading amount in dollars

        if pred > self.threshold:

            order_data = MarketOrderRequest(
                symbol=self.ticker,
                qty=qty,
                side=OrderSide.BUY,
                type="market",
                time_in_force="day",
            )
            self.trading_client.submit_order(order_data)
            self.position = "long"
            print(f"BUY ${qty} shares of {self.ticker} @ ~${price:.2f}")

        elif pred < (1 - self.threshold):

            order_data = MarketOrderRequest(
                symbol=self.ticker,
                qty=qty,
                side=OrderSide.SELL,
                type="market",
                time_in_force="day",
            )
            self.trading_client.submit_order(order_data)
            self.position = "short"
            print(f"SHORT ${qty} shares of {self.ticker} @ ~${price:.2f}")

        else:
            self.position = None
            print(f"No trade - confidence within neutral zone")

    def close_position(self):
        """Called before market close — flatten everything"""
        print(f"\\n{'='*50}")
        print(f"CLOSE — {datetime.now(ET).strftime('%Y-%m-%d %H:%M:%S ET')}")
        print(f"{'='*50}")

        current_qty = self.get_position()

        if current_qty == 0:
            print("No open position to close")
            return

        try:
            self.trading_client.close_position(self.ticker)

            # Log P&L
            account = self.trading_client.get_account()
            portfolio = float(account.portfolio_value)
            equity = float(account.equity)
            last_eq = float(account.last_equity)
            day_pl = equity - last_eq

            print(f"All positions closed")
            print(f"Portfolio value: ${portfolio:.2f}")
            print(f"Day P&L:         ${day_pl:+.2f}")
            self.position = None

        except Exception as e:
            print(f"Error closing positions: {e}")

    def get_recent_orders(self):

        order_request = GetOrdersRequest(status="all", limit=20, symbols=[self.ticker])

        orders = self.trading_client.get_orders(filter=order_request)

        if not orders:
            return pd.DataFrame()
        return pd.DataFrame(
            [
                {
                    "Time": pd.to_datetime(o.submitted_at).strftime("%Y-%m-%d %H:%M"),
                    "Symbol": o.symbol,
                    "Side": o.side.upper(),
                    "Amount": (f"${float(o.notional):.2f}" if o.notional else f"{o.qty} shares"),
                    "Status": o.status,
                }
                for o in orders
            ]
        )

    def test_trade(self, direction):

        account = self.trading_client.get_account()

        request_param = StockLatestQuoteRequest(symbol_or_symbols=self.ticker)
        latest = self.stock_client.get_stock_latest_quote(request_param)
        price = float(latest[self.ticker].ask_price)
        cash = float(account.cash)
        portfolio = float(account.portfolio_value)

        if self.get_position() != 0:
            print("Already in a position — skipping open")
            return

        qty = round(cash * self.cash_fraction / price)  # Trading amount in dollars

        side = OrderSide.BUY if direction == "buy" else OrderSide.SELL
        order_data = MarketOrderRequest(
            symbol=self.ticker, qty=qty, side=side, type="market", time_in_force="day"
        )
        self.trading_client.submit_order(order_data)
        self.position = "long"
        print(f"BUY ${qty} shares of {self.ticker} @ ~${price:.2f}")

    def test_sell(self):

        account = self.trading_client.get_account()

        request_param = StockLatestQuoteRequest(symbol_or_symbols=self.ticker)
        latest = self.stock_client.get_stock_latest_quote(request_param)
        price = float(latest[self.ticker].ask_price)
        cash = float(account.cash)
        portfolio = float(account.portfolio_value)

        if self.get_position() != 0:
            print("Already in a position — skipping open")
            return

        qty = round(cash * self.cash_fraction / price)  # Trading amount in dollars

        order_data = MarketOrderRequest(
            symbol=self.ticker,
            notional=qty,
            side=OrderSide.SELL,
            type="market",
            time_in_force="day",
        )
        self.trading_client.submit_order(order_data)
        self.position = "long"
        print(f"BUY ${qty} shares of {self.ticker} @ ~${price:.2f}")
