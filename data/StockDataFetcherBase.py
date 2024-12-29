from abc import ABC, abstractmethod
import pandas as pd


class StockDataFetcherBase(ABC):
    @abstractmethod
    def fetch_data(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def standardize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        pass
