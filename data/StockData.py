import pandas as pd
from data.decorators import args_check  # 正しくインポート


class StockData:
    @args_check((str, pd.DataFrame), ())
    def __init__(self, symbol, data):
        self.symbol = symbol
        self.data = data

    @args_check((str,), ())
    def save_to_csv(self, filepath):
        self.data.to_csv(filepath, index=True)

    @args_check((str,), ())
    def load_from_csv(self, filepath):
        self.data = pd.read_csv(filepath, index_col=0)
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("Loaded data should be a pandas DataFrame")

    def head(self):
        return self.data.head()
