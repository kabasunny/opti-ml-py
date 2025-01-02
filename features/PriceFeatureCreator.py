import pandas as pd
from features.FeatureCreatorABC import FeatureCreatorABC
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート


class PriceFeatureCreator(FeatureCreatorABC):
    @ArgsChecker((None, pd.DataFrame, pd.Timestamp), pd.DataFrame)
    def create_features(
        self, df: pd.DataFrame, trade_start_date: pd.Timestamp
    ) -> pd.DataFrame:
        """
        価格特徴を生成するメソッド

        Args:
            df (pd.DataFrame): 入力データフレーム
            trade_start_date (pd.Timestamp): トレード開始日

        Returns:
            pd.DataFrame: 価格特徴が追加されたデータフレーム
        """
        # 例として、価格の移動平均やボリンジャーバンドを計算
        df["sma10"] = df["close"].rolling(window=10).mean()
        df["sma30"] = df["close"].rolling(window=30).mean()
        df["bb_up"] = df["sma10"] + 2 * df["close"].rolling(window=10).std()
        df["bb_low"] = df["sma10"] - 2 * df["close"].rolling(window=10).std()

        # trade_start_date 以降のデータをフィルタリング
        df_filtered = df[df["date"] >= trade_start_date].copy()

        return df_filtered
