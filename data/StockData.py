import pandas as pd


class StockData:
    def __init__(self, symbol, data):
        self.symbol = symbol
        self.data = data

    def save_to_csv(self, filepath):
        self.data.to_csv(filepath, index=True)

    def load_from_csv(self, filepath):
        self.data = pd.read_csv(filepath, index_col=0)

    def head(self):
        return self.data.head()
