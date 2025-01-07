import sys
import os
import pandas as pd  # trade_start_date のために必要

# プロジェクトのルートディレクトリを sys.path に追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from LabelCreationPipeline import LabelCreationPipeline
from data.DataManager import DataManager
from labeling.TroughLabelCreator import TroughLabelCreator

if __name__ == "__main__":
    raw_data_path = "data/raw/demo_row_stock_data.csv"
    label_data_path = "data/label/demo_labels.csv"

    # RawDataManager と LabelDataManager のインスタンスを作成
    raw_data_manager = DataManager(raw_data_path)
    label_data_manager = DataManager(label_data_path)

    trade_start_date = pd.Timestamp("2003-08-01")  # ここで trade_start_date を定義
    label_creator = TroughLabelCreator(
        trade_start_date
    )  # トラフラベルクリエータークラス

    # LabelCreationPipeline のインスタンスを作成し、実行
    pipeline = LabelCreationPipeline(
        raw_data_manager, label_data_manager, label_creator
    )
    pipeline.run()
