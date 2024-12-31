# opti-ml-py\preprocessing\runner.py
import sys
import os

# プロジェクトのルートディレクトリを sys.path に追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from PreprocessPipeline import PreprocessPipeline

if __name__ == "__main__":
    raw_data_path = "data/raw/demo_row_stock_data.csv"
    save_path = "data/processed/demo_processed_stock_data.csv"

    pipeline = PreprocessPipeline(raw_data_path, save_path)
    pipeline.run()
