# data\StockData.py
# opti-ml-py\data\StockData.py
import pandas as pd
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート


class StockData:
    @ArgsChecker((str, str, float, float, float, float, int), None)
    def __init__(
        self,
        date: str,
        symbol: str,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: int,
    ):
        self.date = date
        self.symbol = symbol
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume

    @staticmethod
    @ArgsChecker(
        (pd.DataFrame,), list
    )  # データフレームを受け取り、StockDataオブジェクトのリストを返す型チェック
    def from_dataframe(df: pd.DataFrame) -> list:
        stock_data_list = []
        for _, row in df.iterrows():
            stock_data = StockData(
                date=row["date"],
                symbol=row["symbol"],
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
            )
            stock_data_list.append(stock_data)
        return stock_data_list
