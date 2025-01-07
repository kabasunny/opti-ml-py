import pandas as pd
from create_model import create_model
from retrain_model import retrain_model
from data.DataManager import DataManager


def main():
    # シンボルの配列を用意
    symbols = ["1570", "7203", "6758", "9433", "5803"]
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
            print(f"{symbol} の処理中にエラーが発生しました")
            if check_data_availability(
                symbol, trade_start_date, before_period_days, end_date
            ):
                # データ不足のため、スキップします
                print(f"データ不足のため、スキップします")
                continue
            else:
                print(f"{e} \n処理を修了します")
                break


def check_data_availability(symbol, trade_start_date, before_period_days, end_date):
    extra_days = 14  # 余裕として追加する日数
    data_path = f"data/stock_data/formated_raw/{symbol}_{end_date}.parquet"
    data_manager = DataManager(data_path)
    df = data_manager.load_data()

    if df.empty:
        return False

    # データフレームの最古の日付を取得
    oldest_date = df.index.min()
    oldest_date = pd.to_datetime(oldest_date)  # Timestamp型に変換
    required_date = trade_start_date - pd.DateOffset(
        days=(before_period_days - extra_days)
    )

    # 最古の日付が、required_dateよりも新しい場合
    if oldest_date > required_date:
        return False

    return True


if __name__ == "__main__":
    main()
