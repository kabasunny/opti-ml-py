# data\StockDataFetcherABC.py
# opti-ml-py\data\StockDataFetcherABC.py
from abc import ABC, abstractmethod
import pandas as pd
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート


class StockDataFetcherABC(ABC):
    @abstractmethod
    @ArgsChecker((None,), pd.DataFrame)  # fetch_dataメソッドの戻り値型チェック
    def fetch_data(self) -> pd.DataFrame:
        pass

    @abstractmethod
    @ArgsChecker(
        (None, pd.DataFrame), pd.DataFrame
    )  # standardize_dataメソッドの引数と戻り値の型チェック
    def format_data(self, data: pd.DataFrame) -> pd.DataFrame:
        pass
