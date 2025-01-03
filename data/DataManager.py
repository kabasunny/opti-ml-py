import pandas as pd
import os
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート


class DataManager:
    @ArgsChecker((None, str), None)
    def __init__(self, path: str):
        self.path = path

    @ArgsChecker((None, pd.DataFrame), None)
    def save_data(self, df: pd.DataFrame):
        """ラベルデータを保存するメソッド"""
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            df.to_csv(self.path, index=True)
            print(f"データが {self.path} に保存されました")
        except Exception as e:
            print(f"データ保存に失敗しました: {e}")

    @ArgsChecker((None,), pd.DataFrame)
    def load_data(self) -> pd.DataFrame:
        """ラベルデータをロードするメソッド"""
        try:
            df = pd.read_csv(self.path, index_col=0)
            print(f"データが {self.path} からロードされました")
            return df
        except Exception as e:
            print(f"データロードに失敗しました: {e}")
            return pd.DataFrame()
