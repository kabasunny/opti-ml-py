import pandas as pd
import os
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート


class DataManager:
    @ArgsChecker((None, str, str, str, str), None)
    def __init__(self, base_path: str, file_ext: str, data_name: str, end_date: str):
        self.base_path = base_path
        self.file_ext = file_ext
        self.data_name = data_name
        self.end_date = end_date

    def generate_path(self, symbol: str) -> str:
        return f"{self.base_path}/{self.data_name}/{symbol}_{self.end_date}.{self.file_ext}"

    @ArgsChecker((None, pd.DataFrame, str), None)
    def save_data(self, df: pd.DataFrame, symbol: str):
        """ラベルデータを保存するメソッド"""
        path = self.generate_path(symbol)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if path.endswith(".csv"):
                df.to_csv(path, index=True)
            else:
                df.to_parquet(path, index=True)
            # print(f"データが {path} に保存されました")
        except Exception as e:
            print(f"{path}のデータ保存に失敗しました: {e}")

    @ArgsChecker((None, str), pd.DataFrame)
    def load_data(self, symbol: str) -> pd.DataFrame:
        """ラベルデータをロードするメソッド"""
        path = self.generate_path(symbol)
        try:
            if path.endswith(".csv"):
                df = pd.read_csv(path)
            else:
                df = pd.read_parquet(path)
            # print(f"データが {path} からロードされました")
            return df
        except Exception as e:
            print(f"{path}のデータロードに失敗しました: {e}")
            return pd.DataFrame()
