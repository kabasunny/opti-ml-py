import pandas as pd
from features.FeatureCreatorABC import FeatureCreatorABC
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート


class VolumeFeatureCreator(FeatureCreatorABC):
    @ArgsChecker((None, pd.DataFrame, pd.Timestamp), pd.DataFrame)
    def create_features(
        self, df: pd.DataFrame, trade_start_date: pd.Timestamp
    ) -> pd.DataFrame:
        """
        出来高特徴を生成するメソッド

        Args:
            df (pd.DataFrame): 入力データフレーム
            trade_start_date (pd.Timestamp): トレード開始日

        Returns:
            pd.DataFrame: 出来高特徴が追加されたデータフレーム
        """
        # 例として、出来高の移動平均を計算
        df["vsma10"] = df["volume"].rolling(window=10).mean()
        df["vsma30"] = df["volume"].rolling(window=30).mean()

        # trade_start_date 以降のデータをフィルタリング
        df_filtered = df[df["date"] >= trade_start_date].copy()

        return df_filtered
