import yfinance as yf
import pandas as pd
from data.StockDataFetcherBase import StockDataFetcherBase


class YahooFinanceStockDataFetcher(StockDataFetcherBase):
    def __init__(self, symbol, start_date, end_date):
        if not isinstance(symbol, str):
            raise TypeError("Symbol should be a string")
        if not isinstance(start_date, (str, pd.Timestamp)):
            raise TypeError("Start date should be a string or pandas Timestamp")
        if not isinstance(end_date, (str, pd.Timestamp)):
            raise TypeError("End date should be a string or pandas Timestamp")

        self.symbol = symbol + ".T"
        self.start_date = (
            pd.Timestamp(start_date) if isinstance(start_date, str) else start_date
        )
        self.end_date = (
            pd.Timestamp(end_date) if isinstance(end_date, str) else end_date
        )

    def fetch_data(self) -> pd.DataFrame:
        data = yf.download(
            self.symbol, start=self.start_date, end=self.end_date, interval="1d"
        )
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        return data

    def standardize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.reset_index()
        # シンボルから .T を除外
        data["symbol"] = self.symbol.replace(".T", "")
        data = data.rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        # `date`を文字列に変換
        data["date"] = data["date"].astype(str)
        return data[["date", "symbol", "open", "high", "low", "close", "volume"]]
