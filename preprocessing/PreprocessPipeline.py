# preprocessing\PreprocessPipeline.py
import pandas as pd
from decorators.ArgsChecker import ArgsChecker
from preprocessing.MissingValueHandler import MissingValueHandler
from preprocessing.OutlierDetector import OutlierDetector
from preprocessing.Normalizer import Normalizer
from data.RawDataManager import RawDataManager
from data.ProcessedDataManager import ProcessedDataManager


class PreprocessPipeline:
    @ArgsChecker((None, RawDataManager, ProcessedDataManager), None)
    def __init__(
        self,
        raw_data_manager: RawDataManager,
        processed_data_manager: ProcessedDataManager,
    ):
        self.raw_data_manager = raw_data_manager
        self.processed_data_manager = processed_data_manager

    @ArgsChecker((None,), None)
    def run(self):
        """データパイプラインの実行"""
        # データの読み込み
        df = self.raw_data_manager.load_raw_data()
        print("Raw data loaded successfully")

        # データの前処理
        df = MissingValueHandler.fill_missing_with_mean(df)
        print("Handled missing values")

        outliers = OutlierDetector.detect_outliers(df)
        print("Outliers detected")

        df = Normalizer.normalize(df)
        print("Normalized data")

        # データの保存
        self.processed_data_manager.save_processed_data(df)
        print("Processed data saved successfully")

        print("Preprocessing pipeline completed successfully.")
