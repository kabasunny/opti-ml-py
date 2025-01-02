import pandas as pd
from features.FeatureSelectorABC import FeatureSelectorABC
from features.PCAFeatureSelector import PCAFeatureSelector
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート


class CombinedFeatureSelector:
    def __init__(self):
        self.selectors = [PCAFeatureSelector(n_components=10)]

    @ArgsChecker((None, pd.DataFrame), pd.DataFrame)
    def select_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        全ての特徴量選択を統合するメソッド

        Args:
            df (pd.DataFrame): 入力データフレーム

        Returns:
            pd.DataFrame: 選択された特徴量のみを含むデータフレーム
        """
        selected_features = df
        for selector in self.selectors:
            selected_features = selector.select_features(selected_features)
        return selected_features
