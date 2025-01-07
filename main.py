import pandas as pd
from create_model import create_model
from retrain_model import retrain_model


def main():
    # シンボルの配列を用意
    symbols = ["7203", "6758", "9433", "5803"]
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

    # 初めの一回はcreate_modelを呼び出し、シンボルの一番目を渡す
    symbol = symbols.pop(0)
    create_model(symbol, trade_start_date, data_start_period, end_date, model_types)

    # シンボルの二つ目以降はretrain_modelに渡し、シンボルがなくなるまでループ
    while symbols:
        symbol = symbols.pop(0)
        retrain_model(
            symbol, trade_start_date, data_start_period, end_date, model_types
        )


if __name__ == "__main__":
    main()
