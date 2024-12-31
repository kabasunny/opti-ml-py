import pandas as pd
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート


class MissingValueHandler:
    @staticmethod
    @ArgsChecker((pd.DataFrame,), pd.DataFrame)
    def fill_missing_with_mean(df: pd.DataFrame) -> pd.DataFrame:
        """欠損値を各カラムの平均値で埋める"""
        return df.fillna(df.mean())

    @staticmethod
    @ArgsChecker((pd.DataFrame, dict), pd.DataFrame)
    def fill_missing_with_value(df: pd.DataFrame, fill_values: dict) -> pd.DataFrame:
        """欠損値を指定した値で埋める"""
        return df.fillna(fill_values)
