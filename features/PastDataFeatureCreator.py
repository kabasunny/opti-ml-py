import pandas as pd
from features.FeatureCreatorABC import FeatureCreatorABC
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート


class PastDataFeatureCreator(FeatureCreatorABC):
    @ArgsChecker((None, pd.DataFrame, pd.Timestamp), pd.DataFrame)
    def create_features(
        self, df: pd.DataFrame, trade_start_date: pd.Timestamp
    ) -> pd.DataFrame:
        """
        過去のデータに基づいて特徴量を生成するメソッド

        Args:
            df (pd.DataFrame): 入力データフレーム
            trade_start_date (pd.Timestamp): トレード開始日

        Returns:
            pd.DataFrame: 特徴量が追加されたデータフレーム
        """
        # 1個前から5個前のデータに基づく特徴量を追加
        for i in range(1, 6):
            df[f"lag_{i}_close"] = df["close"].shift(i)
            df[f"lag_{i}_indicator"] = (df[f"lag_{i}_close"] > df["close"]).astype(int)
            df.drop(
                columns=[f"lag_{i}_close"], inplace=True
            )  # セレクターによる除外率が高い

        # フィルタリングせずに戻す
        return df
