import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from selectores.SupervisedFeatureSelectorABC import SupervisedFeatureSelectorABC
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート


class TreeFeatureSelector(SupervisedFeatureSelectorABC):
    """
    TreeFeatureSelectorクラスは、ランダムフォレストに基づいて特徴量を選択するためのクラス
    複数の決定木を使ったアンサンブル学習手法であり、各特徴量の重要度を測定して重要な特徴量を選択する

    Attributes:
        model (RandomForestClassifier): ランダムフォレスト分類器
    """

    @ArgsChecker((None, int), None)
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state
        )

    @ArgsChecker((None, pd.DataFrame, str), pd.DataFrame)
    def select_features(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """
        ランダムフォレストに基づいて特徴量を選択するメソッド

        Args:
            df (pd.DataFrame): 入力データフレーム
            target_column (str): 目的変数（ターゲットラベル）のカラム名

        Returns:
            pd.DataFrame: 選択された特徴量のみを含むデータフレーム
        """
        features = df.drop(columns=[target_column])  # ターゲット列を除外
        target = df[target_column]

        # RandomForestClassifierを適用
        self.model.fit(features, target)
        importances = self.model.feature_importances_
        selected_features = features.columns[importances > np.mean(importances)]
        selected_df = df[selected_features]
        return selected_df
