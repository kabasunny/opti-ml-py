import pandas as pd
from create_model import create_model
from retrain_model import retrain_model


def main():
    # シンボルの配列を用意
    symbols = [
        "1570",  # 意図的にエラーを発生するシンボル
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

    model_created = False  # モデルが作成されたかどうかのフラグ

    while symbols:
        symbol = symbols.pop(0)
        print(f'Symbol of current data: {symbol}')

        try:
            if not model_created:
                create_model(
                    symbol, trade_start_date, data_start_period, end_date, model_types
                )
                model_created = True
            else:
                retrain_model(
                    symbol, trade_start_date, data_start_period, end_date, model_types
                )
        except Exception as e:
            print(f"{symbol} の処理中にエラーが発生しました: {e}")
            # エラーが発生した場合スキップします
            continue


if __name__ == "__main__":
    main()
