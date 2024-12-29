import pandas as pd


class StockData:
    def __init__(self, symbol, data):
        self.symbol = symbol
        self.data = data

    def save_to_csv(self, filepath):
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("Data should be a pandas DataFrame")
        if not isinstance(filepath, str):
            raise TypeError("File path should be a string")
        self.data.to_csv(filepath, index=True)

    def load_from_csv(self, filepath):
        if not isinstance(filepath, str):
            raise TypeError("File path should be a string")
        self.data = pd.read_csv(filepath, index_col=0)
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("Loaded data should be a pandas DataFrame")

    def head(self):
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("Data should be a pandas DataFrame")
        return self.data.head()
