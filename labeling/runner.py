import sys
import os
import pandas as pd  # trade_start_date のために必要

# プロジェクトのルートディレクトリを sys.path に追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from labeling.LabelCreatePipeline import LabelCreatePipeline
from data.DataManager import DataManager
from labeling.TroughLabelCreator import TroughLabelCreator

if __name__ == "__main__":
    raw_data_path = "data/stock_data/formated_raw/2501_2025-01-07.parquet"
    label_data_path = "data/stock_data/labeled/2501_2025-01-08.parquet"

    # RawDataManager と LabelDataManager のインスタンスを作成
    raw_data_manager = DataManager(raw_data_path)
    label_data_manager = DataManager(label_data_path)

    trade_start_date = pd.Timestamp("2003-08-01")  # ここで trade_start_date を定義
    label_creator = TroughLabelCreator(
        trade_start_date
    )  # トラフラベルクリエータークラス

    # LabelCreationPipeline のインスタンスを作成し、実行
    pipeline = LabelCreatePipeline(raw_data_manager, label_data_manager, label_creator)
    pipeline.run()
