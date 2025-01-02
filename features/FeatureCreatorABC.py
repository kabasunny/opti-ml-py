from abc import ABC, abstractmethod
import pandas as pd
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート


class FeatureCreatorABC(ABC):
    @abstractmethod
    @ArgsChecker((pd.DataFrame, pd.Timestamp), pd.DataFrame)
    def create_features(
        self, df: pd.DataFrame, trade_start_date: pd.Timestamp
    ) -> pd.DataFrame:
        """
        特徴量を作成するための抽象メソッド

        Args:
            df (pd.DataFrame): 入力データフレーム
            trade_start_date (pd.Timestamp): トレード開始日

        Returns:
            pd.DataFrame: 特徴量が追加されたデータフレーム
        """
        pass
