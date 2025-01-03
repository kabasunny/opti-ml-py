import pandas as pd
from selectores.PCAFeatureSelector import PCAFeatureSelector
from selectores.CorrelationFeatureSelector import CorrelationFeatureSelector
from selectores.LassoFeatureSelector import LassoFeatureSelector
from selectores.TreeFeatureSelector import TreeFeatureSelector
from decorators.ArgsChecker import ArgsChecker


class CombinedFeatureSelector:
    @ArgsChecker((None, list), None)
    def __init__(self, selector_list_str: list):
        selector_mapping = {
            "pca": PCAFeatureSelector(n_components=10),
            "correlation": CorrelationFeatureSelector(threshold=0.9),
            "lasso": LassoFeatureSelector(alpha=0.01),
            "tree": TreeFeatureSelector(n_estimators=100),
        }
        self.selectors = [
            selector_mapping[selector]
            for selector in selector_list_str
            if selector in selector_mapping
        ]

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
