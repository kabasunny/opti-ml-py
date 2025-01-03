from abc import ABC, abstractmethod
import pandas as pd

from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート


class FeatureSelectorABC(ABC):
    @abstractmethod
    @ArgsChecker((None, pd.DataFrame), pd.DataFrame)
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        特徴量を選択するための抽象メソッド

        Args:
            df (pd.DataFrame): 入力データフレーム

        Returns:
            pd.DataFrame: 選択された特徴量のみを含むデータフレーム
        """
        pass
