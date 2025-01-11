import pandas as pd
from model_training import model_training
from models.ModelSaverLoader import ModelSaverLoader


def main():
    # シンボルの配列を用意
    symbols = [
        # "1570",  # 意図的にエラーを発生するシンボル
        "7203",  # Toyota Motor Corporation
        "7201",  # Honda Motor Co., Ltd.
        "7261",  # Mazda Motor Corporation
        "7269",  # Suzuki Motor Corporation
        "7270",  # Subaru Corporation
        "7202",  # Isuzu Motors Limited
        "7205",  # Hino Motors, Ltd.
        "7224",  # Shizuoka Daihatsu Motor Co., Ltd.
    ]

    trade_start_date = pd.Timestamp("2003-08-01")
    before_period_days = 366 * 2  # スタート日より、さかのぼって2年間のデータを取得
    data_start_period = trade_start_date - pd.DateOffset(days=before_period_days)
    end_date = pd.Timestamp("today").strftime("%Y-%m-%d")
    model_types = [
        "LightGBM",
        "RandomForest",
        "XGBoost",
        "CatBoost",
        "AdaBoost",
        "SVM",
        "KNeighbors",
        "LogisticRegression",
    ]

    # ----------------------model----------------------
    model_save_path = "models/trained_models"
    model_file_ext = "pkl"
    model_saver_loader = ModelSaverLoader(model_save_path, model_file_ext)

    # ----------------------feature----------------------
    feature_list_str = ["peak_trough", "fourier", "volume", "price", "past"]

    model_created = False  # モデルが作成されたかどうかのフラグ

    while symbols:
        symbol = symbols.pop(0)
        print(f"Symbol of current data: {symbol}")
        model_created = model_training(
            symbol,
            trade_start_date,
            data_start_period,
            end_date,
            model_types,
            feature_list_str,
            model_saver_loader,
            model_created,
        )


if __name__ == "__main__":
    main()
