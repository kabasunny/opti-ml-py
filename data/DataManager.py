import pandas as pd
import os
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート
from datetime import datetime


class DataManager:
    @ArgsChecker((None, str, str, str), None)
    def __init__(self, base_path: str, data_manager_name: str, file_ext: str):
        self.base_path = base_path
        self.file_ext = file_ext
        self.d_m_name = data_manager_name

    def generate_path(self, symbol: str, date_str: str) -> str:
        return f"{self.base_path}/{self.d_m_name}/{symbol}_{date_str}.{self.file_ext}"

    @ArgsChecker((None, pd.DataFrame, str), None)
    def save_data(self, df: pd.DataFrame, symbol: str):
        """ラベルデータを保存するメソッド"""
        # 現在の日付を計算
        date_str = datetime.now().strftime("%Y-%m-%d")

        path = self.generate_path(symbol, date_str)
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
        dir_path = f"{self.base_path}/{self.d_m_name}/"
        files = [
            f
            for f in os.listdir(dir_path)
            if f.startswith(symbol) and f.endswith(self.file_ext)
        ]

        if not files:
            print(f"{symbol}のデータファイルが存在しません。")
            return pd.DataFrame()

        # 最も新しい日付を含むファイルを選択
        latest_file = max(
            files,
            key=lambda x: pd.to_datetime(
                x.split("_")[-1].replace(f".{self.file_ext}", "")
            ),
        )
        path = os.path.join(dir_path, latest_file)

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
