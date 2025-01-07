import sys
import os
import pandas as pd  # trade_start_date のために必要

# プロジェクトのルートディレクトリを sys.path に追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from selectores.SelectorPipeline import SelectorPipeline
from data.DataManager import DataManager
from selectores.SelectorFactory import SelectorFactory  # 新しく追加

if __name__ == "__main__":
    label_data_path = "data/stock_data/labeled/7203_2025-01-07.parquet"
    normalized_feature_data_path = (
        "data/stock_data/normalized_ft/7203_2025-01-07.parquet"
    )
    selected_feature_data_path = "data/stock_data/selected_ft/7203_2025-01-07.parquet"

    # データマネージャのインスタンスを作成
    label_data_manager = DataManager(label_data_path)
    normalized_f_d_manager = DataManager(normalized_feature_data_path)
    selected_f_d_manager = DataManager(selected_feature_data_path)

    # パイプラインの手順を定義
    selectors = SelectorFactory.create_selectors()

    # SelectorPipeline のインスタンスを作成し、実行
    pipeline = SelectorPipeline(
        label_data_manager, normalized_f_d_manager, selected_f_d_manager, selectors
    )
    selected_features_df = pipeline.run()

    # 結果を表示
    print("Selected features:")
    print(selected_features_df)
