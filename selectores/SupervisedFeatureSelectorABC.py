from abc import ABC, abstractmethod
import pandas as pd

from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート


class SupervisedFeatureSelectorABC(ABC):
    @abstractmethod
    @ArgsChecker((None, pd.DataFrame, str), pd.DataFrame)
    def select_features(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """
        特徴量を選択するための抽象メソッド（正解ラベル必要）

        Args:
            df (pd.DataFrame): 入力データフレーム
            target_column (str): 目的変数（ターゲットラベル）のカラム名

        Returns:
            pd.DataFrame: 選択された特徴量のみを含むデータフレーム
        """
        pass
