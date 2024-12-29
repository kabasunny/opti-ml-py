import yfinance as yf
import pandas as pd
from data.StockData import StockData


class StockDataFetcher:
    def __init__(self, symbol, start_date, end_date):
        if not isinstance(symbol, str):
            raise TypeError("Symbol should be a string")
        if not isinstance(start_date, (str, pd.Timestamp)):
            raise TypeError("Start date should be a string or pandas Timestamp")
        if not isinstance(end_date, (str, pd.Timestamp)):
            raise TypeError("End date should be a string or pandas Timestamp")

        self.symbol = symbol
        self.start_date = (
            pd.Timestamp(start_date) if isinstance(start_date, str) else start_date
        )
        self.end_date = (
            pd.Timestamp(end_date) if isinstance(end_date, str) else end_date
        )

    def fetch_data(self):
        # 日足データの取得
        data = yf.download(
            self.symbol, start=self.start_date, end=self.end_date, interval="1d"
        )

        # マルチインデックスの場合、カラムの最初のレベルを削除
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        # StockDataのインスタンスを生成
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Fetched data should be a pandas DataFrame")

        return StockData(self.symbol, data)
