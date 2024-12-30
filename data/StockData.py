import pandas as pd
from data.decorators import ArgsChecker  # 正しくインポート


class StockData:
    @ArgsChecker((str, pd.DataFrame), ())
    def __init__(self, symbol, data):
        self.symbol = symbol
        self.data = data

    @ArgsChecker((str,), ())
    def save_to_csv(self, filepath):
        self.data.to_csv(filepath, index=True)

    @ArgsChecker((str,), ())
    def load_from_csv(self, filepath):
        self.data = pd.read_csv(filepath, index_col=0)
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("Loaded data should be a pandas DataFrame")

    def head(self):
        return self.data.head()
