# preprocessing\PreprocessPipeline.py
import pandas as pd
from decorators.ArgsChecker import ArgsChecker
from preprocessing.MissingValueHandler import MissingValueHandler
from preprocessing.OutlierDetector import OutlierDetector
from preprocessing.Normalizer import Normalizer
from data.RawDataManager import RawDataManager
from data.ProcessedDataManager import ProcessedDataManager


class PreprocessPipeline:
    @ArgsChecker((None, str, str), None)
    def __init__(self, raw_data_path: str, save_path: str):
        self.raw_data_manager = RawDataManager(
            load_path=raw_data_path, save_path=raw_data_path
        )
        self.processed_data_manager = ProcessedDataManager(save_path=save_path)

    def load_data(self) -> pd.DataFrame:
        """生データを読み込む"""
        print("Loading raw data...")
        df = self.raw_data_manager.load_raw_data()
        print("Raw data loaded successfully.")
        return df

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """データ前処理を行う"""
        print("Handling missing values...")
        df = MissingValueHandler.fill_missing_with_mean(df)

        print("Detecting outliers...")
        outliers = OutlierDetector.detect_outliers(df)
        print("Outliers detected:")
        print(outliers)

        print("Normalizing data...")
        df = Normalizer.normalize(df)

        return df

    def save_data(self, df: pd.DataFrame):
        """前処理済みデータを保存する"""
        print("Saving processed data...")
        self.processed_data_manager.save_processed_data(df)
        print("Processed data saved successfully.")

    @ArgsChecker((None,), None)
    def run(self):
        """データパイプラインの実行"""
        df = self.load_data()
        df = self.preprocess_data(df)
        self.save_data(df)
        print("Preprocessing pipeline completed successfully.")
