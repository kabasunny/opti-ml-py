import unittest
from unittest.mock import MagicMock
import pandas as pd
from data.DataPipeline import DataPipeline
from data.YahooFinanceStockDataFetcher import YahooFinanceStockDataFetcher
from data.DataSaver import DataSaver


class TestDataPipeline(unittest.TestCase):
    def setUp(self):
        # モックのデータフェッチャーとデータセーバーを作成
        self.fetcher = YahooFinanceStockDataFetcher("7203", "2023-01-01", "2023-12-31")
        self.saver = DataSaver()

        # モックの戻り値を設定
        self.mock_raw_data = pd.DataFrame(
            {
                "Date": ["2023-01-01", "2023-01-02"],
                "Open": [100, 110],
                "High": [120, 130],
                "Low": [90, 100],
                "Close": [110, 120],
                "Volume": [1000, 1500],
            }
        )
        self.mock_standardized_data = self.mock_raw_data.copy()
        self.mock_standardized_data["symbol"] = "7203"
        self.mock_standardized_data = self.mock_standardized_data.rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )

        # メソッドをモック
        self.fetcher.fetch_data = MagicMock(return_value=self.mock_raw_data)
        self.fetcher.standardize_data = MagicMock(
            return_value=self.mock_standardized_data
        )
        self.saver.save_raw_data = MagicMock()

    def test_data_pipeline(self):
        pipeline = DataPipeline(self.fetcher, self.saver)
        pipeline.run("data/test/processed_stock_data.csv")

        # 各メソッドが正しく呼び出されたかを確認
        self.fetcher.fetch_data.assert_called_once()
        self.fetcher.standardize_data.assert_called_once_with(self.mock_raw_data)
        self.saver.save_raw_data.assert_called_once_with(
            self.mock_standardized_data, "data/test/processed_stock_data.csv"
        )


if __name__ == "__main__":
    unittest.main()