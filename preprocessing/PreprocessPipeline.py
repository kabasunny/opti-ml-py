import pandas as pd
from decorators.ArgsChecker import ArgsChecker
from preprocessing.MissingValueHandler import MissingValueHandler
from preprocessing.OutlierDetector import OutlierDetector
from preprocessing.Normalizer import Normalizer
from data.DataManager import DataManager


class PreprocessPipeline:
    @ArgsChecker((None, DataManager, DataManager), None)
    def __init__(
        self,
        raw_data_manager: DataManager,
        processed_data_manager: DataManager,
    ):
        self.raw_data_manager = raw_data_manager
        self.processed_data_manager = processed_data_manager

    @ArgsChecker((None,), None)
    def run(self):
        """データパイプラインの実行"""
        # print("Run Preprocess pipeline")
        # データの読み込み
        df = self.raw_data_manager.load_data()
        # print("Raw data loaded successfully")

        # データの前処理
        df = MissingValueHandler.fill_missing_with_mean(df)
        # print("Handled missing values")

        outliers = OutlierDetector.detect_outliers(df)
        # print("Outliers detected")

        # 正規化する列を指定
        columns_to_normalize = ["open", "high", "low", "close", "volume"]

        # 正規化の適用
        df = Normalizer.normalize(df, columns_to_normalize)
        # print("Normalized data")

        # データの保存
        self.processed_data_manager.save_data(df)
        # print("Processed data saved successfully")

        print("Preprocess pipeline completed successfully")
