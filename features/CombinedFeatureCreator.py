import pandas as pd
from features.PeakTroughAnalyzer import PeakTroughAnalyzer
from features.FourierAnalyzer import FourierAnalyzer
from features.VolumeFeatureCreator import VolumeFeatureCreator
from features.PriceFeatureCreator import PriceFeatureCreator
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート


class CombinedFeatureCreator:
    @ArgsChecker((None, list, pd.Timestamp), None)
    def __init__(self, feature_list_str: list, trade_start_date: pd.Timestamp):
        analyzer_mapping = {
            "peak_trough": PeakTroughAnalyzer,
            "fourier": FourierAnalyzer,
            "volume": VolumeFeatureCreator,
            "price": PriceFeatureCreator,
        }
        self.analyzers = [
            analyzer_mapping[feature]()
            for feature in feature_list_str
            if feature in analyzer_mapping
        ]
        self.trade_start_date = trade_start_date

    @ArgsChecker((None, pd.DataFrame), pd.DataFrame)
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        全ての特徴量を作成するメソッド

        Args:
            df (pd.DataFrame): 入力データフレーム

        Returns:
            pd.DataFrame: 全ての特徴量が追加されたデータフレーム
        """
        # dateカラムをTimestamp型に変換
        df["date"] = pd.to_datetime(df["date"])

        for analyzer in self.analyzers:
            df = analyzer.create_features(df, self.trade_start_date)
            # print(f"Data with features: {analyzer}")
            # print(df.head())
            # print(df.tail())
            # print(df.info())

        # trade_start_date 以降の日付のデータをフィルタリング
        df = df[df["date"] >= self.trade_start_date].copy()

        return df
