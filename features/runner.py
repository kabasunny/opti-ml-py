import sys
import os
import pandas as pd  # trade_start_date のために必要

# プロジェクトのルートディレクトリを sys.path に追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from FeaturePipeline import FeaturePipeline
from data.DataManager import DataManager
from AnalyzerFactory import AnalyzerFactory

if __name__ == "__main__":
    processed_data_path = "data/processed/demo_processed_stock_data.csv"
    feature_data_path = "data/feature/demo_feature_data.csv"
    normalized_feature_data_path = "data/processed/demo_normalized_feature_data.csv"

    # ProcessedDataManager と FeatureDataManager のインスタンスを作成
    processed_data_manager = DataManager(processed_data_path)
    feature_data_manager = DataManager(feature_data_path)
    normalized_f_d_manager = DataManager(normalized_feature_data_path)

    trade_start_date = pd.Timestamp("2023-08-01")  # ここで trade_start_date を定義
    feature_list_str = ["peak_trough", "fourier", "volume", "price"]  # 特徴量リスト

    # AnalyzerFactoryを使用してアナライザーのリストを生成
    analyzers = AnalyzerFactory.create_analyzers(feature_list_str)

    # FeaturePipeline のインスタンスを作成し、実行
    pipeline = FeaturePipeline(
        processed_data_manager,
        feature_data_manager,
        normalized_f_d_manager,
        analyzers,
        trade_start_date,
    )

    pipeline.run()
