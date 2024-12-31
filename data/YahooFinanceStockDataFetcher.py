# data\YahooFinanceStockDataFetcher.py
# opti-ml-py\data\YahooFinanceStockDataFetcher.py
import yfinance as yf
import pandas as pd
from data.StockDataFetcherABC import StockDataFetcherABC
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート


class YahooFinanceStockDataFetcher(StockDataFetcherABC):
    @ArgsChecker((None, str, (str, pd.Timestamp), (str, pd.Timestamp)), None)
    def __init__(self, symbol, start_date, end_date):
        self.symbol = symbol + ".T"
        self.start_date = (
            pd.Timestamp(start_date) if isinstance(start_date, str) else start_date
        )
        self.end_date = (
            pd.Timestamp(end_date) if isinstance(end_date, str) else end_date
        )

    @ArgsChecker((None,), pd.DataFrame)
    def fetch_data(self) -> pd.DataFrame:
        data = yf.download(
            self.symbol, start=self.start_date, end=self.end_date, interval="1d"
        )
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        return data

    @ArgsChecker((None, pd.DataFrame), pd.DataFrame)
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
