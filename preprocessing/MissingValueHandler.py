# preprocessing\MissingValueHandler.py
import pandas as pd
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート


class MissingValueHandler:
    @staticmethod
    @ArgsChecker((pd.DataFrame,), pd.DataFrame)
    def fill_missing_with_mean(df: pd.DataFrame) -> pd.DataFrame:
        """
        欠損値を各カラムの平均値で埋めるメソッド。

        数値データを含むカラムの欠損値をそのカラムの平均値で埋め、
        データフレームの統計的性質を維持しながら欠損値を処理。
        """
        numeric_cols = df.select_dtypes(include="number").columns
        numeric_cols = [
            col for col in numeric_cols if col != "symbol"
        ]  # 'symbol'を除外
        df[numeric_cols] = df[numeric_cols].fillna(
            df[numeric_cols].mean()
        )  # 欠損値を平均値で埋める
        return df

    @staticmethod
    @ArgsChecker((pd.DataFrame, dict), pd.DataFrame)
    def fill_missing_with_value(df: pd.DataFrame, fill_values: dict) -> pd.DataFrame:
        """
        欠損値を指定した値で埋めるメソッド。

        データフレーム内の欠損値を指定された値で埋め、
        欠損値を埋める値は辞書形式で提供され、カラム名をキー、埋める値を値とする。
        """
        return df.fillna(fill_values)  # 欠損値を指定した値で埋める
