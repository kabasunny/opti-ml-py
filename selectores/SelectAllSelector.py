import pandas as pd
from selectores.UnsupervisedFeatureSelectorABC import UnsupervisedFeatureSelectorABC
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート


class SelectAllSelector(UnsupervisedFeatureSelectorABC):
    """
    SelectAllSelectorクラスは、すべての特徴量を選択するためのクラス

    Attributes:
        なし
    """

    def __init__(self):
        pass

    @ArgsChecker((None, pd.DataFrame), pd.DataFrame)
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        すべての特徴量を選択するメソッド

        Args:
            df (pd.DataFrame): 入力データフレーム

        Returns:
            pd.DataFrame: すべての特徴量を含むデータフレーム
        """
        return df
