# preprocessing\DataIntegrator.py
import pandas as pd
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート


class DataIntegrator:
    @staticmethod
    @ArgsChecker((pd.DataFrame, pd.DataFrame), pd.DataFrame)
    def integrate_data(df1: pd.DataFrame, df2: pd.DataFrame, on: str) -> pd.DataFrame:
        """
        2つのデータフレームを指定したカラムで統合するメソッド。

        このメソッドは、2つのデータフレームを指定したカラムで統合し、
        統合するカラムは引数で指定。
        """
        return pd.merge(df1, df2, on=on)  # データフレームを統合して返す
