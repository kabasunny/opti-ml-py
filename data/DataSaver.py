import os
import pandas as pd
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート


class DataSaver:  # 抽象クラスを継承
    @ArgsChecker((None, pd.DataFrame, str), None)
    def save_raw_data(self, data: pd.DataFrame, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        data.to_csv(save_path, index=False)
        print(f"Data saved to {save_path}")
