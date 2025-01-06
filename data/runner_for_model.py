import sys
import os

# プロジェクトのルートディレクトリを sys.path に追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import pandas as pd
from data.DataManager import DataManager
from data.DataForModelPipeline import DataForModelPipeline


def runner(symbol, end_date):
    # データ保存ディレクトリのベースパスと拡張子を指定
    base_data_path = "data/stock_data"
    file_ext = "parquet"  # CSVの代わりにparquetを使用 Goで使用可能

    # パス生成用関数
    def generate_path(sub_dir, symbol, end_date, extension):
        return f"{base_data_path}/{sub_dir}/{symbol}_{end_date}.{extension}"

    # 各パイプラインのデータ保存パス
    processed_data_path = generate_path("processed_raw", symbol, end_date, file_ext)
    label_data_path = generate_path("labeled", symbol, end_date, file_ext)
    feature_data_path = generate_path("feature", symbol, end_date, file_ext)
    selected_feature_data_path = generate_path(
        "selected_ft", symbol, end_date, file_ext
    )
    training_and_test_data_path = generate_path(
        "training_and_test", symbol, end_date, file_ext
    )
    practical_data_path = generate_path("practical", symbol, end_date, file_ext)

    # データマネージャのインスタンスを作成
    processed_data_manager = DataManager(processed_data_path)
    label_data_manager = DataManager(label_data_path)
    feature_data_manager = DataManager(feature_data_path)
    selected_feature_manager = DataManager(selected_feature_data_path)
    training_and_test_data_manager = DataManager(training_and_test_data_path)
    practical_data_manager = DataManager(practical_data_path)

    # PrepareDataForModelPipeline のインスタンスを作成し、実行
    pipeline = DataForModelPipeline(
        processed_data_manager,
        label_data_manager,
        feature_data_manager,
        selected_feature_manager,
        training_and_test_data_manager,
        practical_data_manager,
    )
    pipeline.run()


if __name__ == "__main__":
    symbol = "7203"
    end_date = pd.Timestamp("today").strftime("%Y-%m-%d")  # 今日の日付

    runner(symbol, end_date)
