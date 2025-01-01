# opti-ml-py\preprocessing\runner.py
import sys
import os

# プロジェクトのルートディレクトリを sys.path に追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from PreprocessPipeline import PreprocessPipeline
from data.RawDataManager import RawDataManager
from data.ProcessedDataManager import ProcessedDataManager

if __name__ == "__main__":
    raw_data_path = "data/raw/demo_row_stock_data.csv"
    processed_data_path = "data/processed/demo_processed_stock_data.csv"

    # RawDataManager と ProcessedDataManager のインスタンスを作成
    raw_data_manager = RawDataManager(raw_data_path)
    processed_data_manager = ProcessedDataManager(processed_data_path)

    # PreprocessPipeline のインスタンスを作成し、引数としてデータマネージャを渡す
    pipeline = PreprocessPipeline(raw_data_manager, processed_data_manager)
    pipeline.run()
