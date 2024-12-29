# main.py

import os
import pandas as pd
from data.StockDataFetcher import StockDataFetcher


# テスト用のコード
def main():
    symbol = "7203.T"
    start_date = pd.Timestamp("2023-01-01")
    end_date = pd.Timestamp("2023-12-31")

    fetcher = StockDataFetcher(symbol, start_date, end_date)
    stock_data = fetcher.fetch_data()
    print("Daily data:")
    print(stock_data.head())

    # 保存先ディレクトリの確認と作成
    save_dir = "data/test"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # データをファイルに保存する例
    save_path = os.path.join(save_dir, "stock_data.csv")
    stock_data.save_to_csv(save_path)


if __name__ == "__main__":
    main()
