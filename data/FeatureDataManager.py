import os
import pandas as pd
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート


class FeatureDataManager:
    @ArgsChecker((None, str), None)
    def __init__(self, path: str):
        self.path = path

    @ArgsChecker((None, pd.DataFrame), None)
    def save_feature_data(self, data: pd.DataFrame):
        """データを指定されたパスに保存するメソッド"""
        parent_dir = os.path.dirname(self.path)
        os.makedirs(parent_dir, exist_ok=True)
        data.to_csv(self.path, index=False)
        print(f"データが {self.path} に保存されました")

    @ArgsChecker((None,), pd.DataFrame)
    def load_feature_data(self) -> pd.DataFrame:
        """指定されたパスからデータを読み込むメソッド"""
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"ファイルが存在しません: '{self.path}'")
        data = pd.read_csv(self.path)
        print(f"データが {self.path} からロードされました")
        return data
