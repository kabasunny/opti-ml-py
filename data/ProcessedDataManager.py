import os
import pandas as pd
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート


class ProcessedDataManager:
    @ArgsChecker((None, str), None)
    def __init__(self, save_path: str):
        self.save_path = save_path

    @ArgsChecker((None, pd.DataFrame), None)
    def save_processed_data(self, data: pd.DataFrame):
        """データを指定されたパスに保存するメソッド"""
        parent_dir = os.path.dirname(self.save_path)
        os.makedirs(parent_dir, exist_ok=True)
        data.to_csv(self.save_path, index=False)
        print(f"Data saved to {self.save_path}")

    @ArgsChecker((None, str), pd.DataFrame)
    def load_processed_data(self, file_path: str) -> pd.DataFrame:
        """指定されたパスからデータを読み込むメソッド"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No such file: '{file_path}'")
        data = pd.read_csv(file_path)
        print(f"Data loaded from {file_path}")
        return data
