import pandas as pd
import numpy as np
from selectores.UnsupervisedFeatureSelectorABC import UnsupervisedFeatureSelectorABC
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート


class CorrelationFeatureSelector(UnsupervisedFeatureSelectorABC):
    """
    CorrelationFeatureSelectorクラスは、相関係数に基づいて特徴量を選択するためのクラス
    相関係数が高い特徴量同士を排除することで、冗長な特徴量を削除し、モデルの性能を向上させる

    Attributes:
        threshold (float): 相関係数の閾値
    """

    @ArgsChecker((None, float), None)
    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold

    @ArgsChecker((None, pd.DataFrame), pd.DataFrame)
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        相関係数に基づいて特徴量を選択するメソッド

        Args:
            df (pd.DataFrame): 入力データフレーム

        Returns:
            pd.DataFrame: 選択された特徴量のみを含むデータフレーム
        """
        correlation_matrix = df.corr().abs()
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool_)
        )
        to_drop = [
            column
            for column in upper_triangle.columns
            if any(upper_triangle[column] > self.threshold)
        ]
        selected_features = df.drop(columns=to_drop, axis=1)
        selected_df = selected_features
        return selected_df
