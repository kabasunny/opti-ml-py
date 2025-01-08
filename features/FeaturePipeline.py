import pandas as pd
from selectores.CombinedFeatureSelector import CombinedFeatureSelector
from preprocessing.Normalizer import Normalizer  # Normalizerクラスをインポート
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート
from data.DataManager import DataManager


class FeaturePipeline:
    @ArgsChecker((None, DataManager, DataManager, list, pd.Timestamp), None)
    def __init__(
        self,
        processed_data_manager: DataManager,
        normalized_f_d_manager: DataManager,
        analyzers: list,
        trade_start_date: pd.Timestamp,
    ):
        self.normalizer = Normalizer()  # Normalizerクラスのインスタンスを作成
        self.processed_data_manager = processed_data_manager
        self.normalized_f_d_manager = normalized_f_d_manager
        self.analyzers = analyzers
        self.trade_start_date = trade_start_date

    @ArgsChecker((None,), None)
    def run(self):
        """
        特徴量作成と選択を一連の流れで実行するメソッド

        Returns:
            pd.DataFrame: 選択された特徴量のみを含むデータフレーム
        """
        # print("Run Feature creation pipeline")
        # データをロード
        df = self.processed_data_manager.load_data()

        # 特徴量を作成
        # dateカラムをTimestamp型に変換
        df["date"] = pd.to_datetime(df["date"])

        for analyzer in self.analyzers:
            df = analyzer.create_features(df, self.trade_start_date)

        # trade_start_date 以降の日付のデータをフィルタリング
        df_with_features = df[df["date"] >= self.trade_start_date].copy()

        # 指定された列を削除
        columns_to_drop = [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
        df_with_features.drop(columns=columns_to_drop, inplace=True)

        # 正規化する列を指定
        columns_to_normalize = [
            "50dtme",
            "30wtme",
            "24mtme",
            "ff2",
            "ff3",
            "ff4",
            "vsma10",
            "vsma30",
            "sma10",
            "sma30",
            "bb_up",
            "bb_low",
        ]

        # 特徴量を正規化
        df_normalized = self.normalizer.normalize(
            df_with_features, columns_to_normalize
        )

        self.normalized_f_d_manager.save_data(df_normalized)

        print("Feature creation pipeline completed successfully")
