import pandas as pd
from features.FeatureCreatorABC import FeatureCreatorABC
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート


class VolumeFeatureCreator(FeatureCreatorABC):
    @ArgsChecker((pd.DataFrame,), pd.DataFrame)
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        出来高特徴を生成するメソッド

        Args:
            df (pd.DataFrame): 入力データフレーム

        Returns:
            pd.DataFrame: 出来高特徴が追加されたデータフレーム
        """
        # 例として、出来高の移動平均を計算
        df["volume_sma_10"] = df["volume"].rolling(window=10).mean()
        df["volume_sma_30"] = df["volume"].rolling(window=30).mean()

        return df
