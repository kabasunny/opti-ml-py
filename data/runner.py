# opti-ml-py\labeling\runner.py
import sys
import os

# プロジェクトのルートディレクトリを sys.path に追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import pandas as pd
from data.YahooFinanceStockDataFetcher import YahooFinanceStockDataFetcher
from data.JQuantsStockDataFetcher import JQuantsStockDataFetcher
from data.RawDataManager import RawDataManager  # RawDataManager クラスのインポート
from data.RawDataPipeline import RawDataPipeline  # DataPipeline クラスのインポート


# フラグを交互に切り替える関数
def toggle(flag):
    return not flag


if __name__ == "__main__":
    symbol = "7203"
    trade_start_date = pd.Timestamp("2023-08-01")
    before_period_days = 366 * 1
    data_period = trade_start_date - pd.DateOffset(days=before_period_days)
    end_date = pd.Timestamp("today")

    raw_data_path = "data/raw/demo_row_stock_data.csv"
    data_manager = RawDataManager(
        raw_data_path
    )  # RawDataManager クラスのインスタンスを作成

    use_jquants = True  # 初期値をTrueに設定
    max_iterations = 2  # 最大ループ回数を設定
    for _ in range(max_iterations):  # ここでは2回交互に実行します
        if use_jquants:
            print("\n★JQuantsStockDataFetcher★")
            fetcher = JQuantsStockDataFetcher("7203", "2023-01-01", "2023-12-31")
        else:
            print("\n★YahooFinanceStockDataFetcher★")
            fetcher = YahooFinanceStockDataFetcher("7203", "2023-01-01", "2023-12-31")

        # DataPipeline クラスのインスタンスを作成し、データパイプラインを実行
        pipeline = RawDataPipeline(fetcher, data_manager)
        pipeline.run()

        use_jquants = toggle(use_jquants)  # フラグを交互に切り替え
