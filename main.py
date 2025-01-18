import pandas as pd
from models.ModelSaverLoader import ModelSaverLoader
from TrainAutomatedPipeline import TrainAutomatedPipeline
from data.DataManager import DataManager
from datetime import datetime
from symbols import symbols  # 追加
from model_types import model_types  # モデルタイプをインポート

def main():
    
    current_date_str = datetime.now().strftime("%Y-%m-%d")

    before_period_days = 365 * 2  # 特徴量生成に必要なデータ期間

    model_saver_loader = ModelSaverLoader(
        current_date_str, model_save_path="models/trained_models", model_file_ext="pkl"
    )

    feature_list_str = ["peak_trough", "fourier", "volume", "price", "past"]

    base_data_path = "data/stock_data"
    file_ext = "csv"  # "parquet"

    data_manager_names = [
        "formated_raw",
        "processed_raw",
        "labeled",
        "normalized_feature",
        "selected_feature",
        "selected_ft_with_label",
        "training_and_test",
        "practical",
        "predictions",
    ]

    data_managers = {}
    for d_m_name in data_manager_names:
        data_managers[d_m_name] = DataManager(current_date_str, base_data_path, d_m_name, file_ext)

    selectors = [
        # "Tree",
        # "Lasso",
        # "Correlation",
        "PCA",
        "SelectAll",
    ]

    pipeline = TrainAutomatedPipeline(
        before_period_days,
        model_types,
        feature_list_str,
        model_saver_loader,
        data_managers,
        selectors,
    )

    while symbols:
        symbol = symbols.pop(0)
        pipeline.process_symbol(symbol)

if __name__ == "__main__":
    main()
