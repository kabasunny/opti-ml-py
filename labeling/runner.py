# opti-ml-py\labeling\runner.py
import sys
import os

# プロジェクトのルートディレクトリを sys.path に追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from LabelCreationPipeline import LabelCreationPipeline
from data.RawDataManager import RawDataManager
from data.LabelDataManager import LabelDataManager
from labeling.TroughLabelCreator import TroughLabelCreator

if __name__ == "__main__":
    raw_data_path = "data/raw/demo_row_stock_data.csv"
    label_data_path = "data/label/demo_labels.csv"

    # RawDataManager と LabelDataManager のインスタンスを作成
    raw_data_manager = RawDataManager(raw_data_path)
    label_data_manager = LabelDataManager(label_data_path)
    label_creator = TroughLabelCreator()  # トラフラベルクリエータークラス

    # LabelCreationPipeline のインスタンスを作成し、実行
    pipeline = LabelCreationPipeline(
        raw_data_manager, label_data_manager, label_creator
    )
    pipeline.run()
