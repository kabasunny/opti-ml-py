# opti-ml-py\models\predict_runner.py
import sys
import os

# プロジェクトのルートディレクトリを sys.path に追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import pandas as pd
from data.DataManager import DataManager
from for_real.ForRealPredictPipeline import ForRealPredictPipeline
from models.ModelSaverLoader import ModelSaverLoader
from model_types import model_types  # モデルタイプをインポート
from datetime import datetime


def runner():
    # データ保存ディレクトリのベースパスと拡張子を指定
    current_date_str = datetime.now().strftime("%Y-%m-%d")
    symbol = "1570"
    base_data_path = "data/stock_data"
    file_ext = "csv"  # CSVの代わりにparquetを使用

    # データマネージャのインスタンスを作成
    data_manager_names = [
        "selected_ft_with_label",
        "real_predictions",
    ]
    data_managers = {}
    for d_m_name in data_manager_names:
        data_managers[d_m_name] = DataManager(current_date_str, base_data_path, d_m_name, file_ext)

    # モデルセーブローダーのインスタンスを作成
    model_save_path = "models/trained_models"
    model_file_ext = "pkl"
    model_saver_loader = ModelSaverLoader(current_date_str, model_save_path, model_file_ext)


    # 予測パイプラインの作成と実行
    real_predict_pipeline = ForRealPredictPipeline(
        model_saver_loader,
        data_managers["selected_ft_with_label"],
        data_managers["real_predictions"],
        model_types,
    )
    real_predict_pipeline.run(symbol)


if __name__ == "__main__":
    runner()
