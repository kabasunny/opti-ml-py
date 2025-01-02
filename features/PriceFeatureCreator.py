import pandas as pd
from features.FeatureCreatorABC import FeatureCreatorABC
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート


class PriceFeatureCreator(FeatureCreatorABC):
    @ArgsChecker((pd.DataFrame,), pd.DataFrame)
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        価格特徴を生成するメソッド

        Args:
            df (pd.DataFrame): 入力データフレーム

        Returns:
            pd.DataFrame: 価格特徴が追加されたデータフレーム
        """
        # 例として、価格の移動平均やボリンジャーバンドを計算
        df["price_sma_10"] = df["close"].rolling(window=10).mean()
        df["price_sma_30"] = df["close"].rolling(window=30).mean()
        df["price_bb_upper"] = (
            df["price_sma_10"] + 2 * df["close"].rolling(window=10).std()
        )
        df["price_bb_lower"] = (
            df["price_sma_10"] - 2 * df["close"].rolling(window=10).std()
        )

        return df
