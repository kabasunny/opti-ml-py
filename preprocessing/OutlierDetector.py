# preprocessing\OutlierDetector.py
import pandas as pd
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート


class OutlierDetector:
    @staticmethod
    @ArgsChecker((pd.DataFrame, float), pd.DataFrame)
    def detect_outliers(df: pd.DataFrame, threshold: float = 1.5) -> pd.DataFrame:
        """
        IQR法を用いて異常値を検出するメソッド。

        四分位範囲 (IQR) を使用してデータフレーム内の異常値を検出する。
        異常値は、第一四分位数 (Q1) から1.5倍のIQRより下、
        または第三四分位数 (Q3) から1.5倍のIQRより上の値として定義される。

        引数:
        df (pd.DataFrame): 異常値を検出するデータフレーム。
        threshold (float): IQRにかける係数（デフォルトは1.5）。

        戻り値:
        pd.DataFrame: 異常値がTrueとしてマークされたブール型データフレーム。
        """
        numeric_cols = df.select_dtypes(include="number").columns
        numeric_cols = [
            col for col in numeric_cols if col != "symbol"
        ]  # 'symbol'を除外
        Q1 = df[numeric_cols].quantile(0.25)  # 第一四分位数
        Q3 = df[numeric_cols].quantile(0.75)  # 第三四分位数
        IQR = Q3 - Q1
        is_outlier = (df[numeric_cols] < (Q1 - threshold * IQR)) | (
            df[numeric_cols] > (Q3 + threshold * IQR)
        )  # 異常値を検出
        return is_outlier  # 異常値を示すブール型データフレームを返す
