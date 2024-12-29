import yfinance as yf
import pandas as pd
from data.StockData import StockData


class StockDataFetcher:
    def __init__(self, symbol, start_date, end_date):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date

    def fetch_data(self):
        # 日足データの取得
        data = yf.download(
            self.symbol, start=self.start_date, end=self.end_date, interval="1d"
        )

        # マルチインデックスの場合、カラムの最初のレベルを削除
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        # StockDataのインスタンスを生成
        return StockData(self.symbol, data)
