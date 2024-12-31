import os
import pandas as pd
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート


class RawDataManager:
    @ArgsChecker((None, str, str), None)
    def __init__(self, save_path: str, load_path: str):
        self.save_path = save_path
        self.load_path = load_path

    @ArgsChecker((None, pd.DataFrame), None)
    def save_raw_data(self, data: pd.DataFrame):
        """データを指定されたパスに保存するメソッド"""
        parent_dir = os.path.dirname(self.save_path)
        os.makedirs(parent_dir, exist_ok=True)
        data.to_csv(self.save_path, index=False)
        print(f"Data saved to {self.save_path}")

    @ArgsChecker((None,), pd.DataFrame)
    def load_raw_data(self) -> pd.DataFrame:
        """指定されたパスからデータを読み込むメソッド"""
        if not os.path.exists(self.load_path):
            raise FileNotFoundError(f"No such file: '{self.load_path}'")
        data = pd.read_csv(self.load_path)
        print(f"Data loaded from {self.load_path}")
        return data
