import os
import pandas as pd
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート


class DataSaver:
    @ArgsChecker((None, pd.DataFrame, str), None)
    def save_raw_data(self, data, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        data.to_csv(save_path, index=False)
        print(f"Data saved to {save_path}")


if __name__ == "__main__":
    demo_data = pd.DataFrame(
        {
            "Date": ["2023-01-01", "2023-01-02"],
            "Open": [100, 110],
            "High": [120, 130],
            "Low": [90, 100],
            "Close": [110, 120],
            "Volume": [1000, 1500],
        }
    )

    demo_save_path = "data/raw/demo_stock_data.csv"

    saver = DataSaver()
    saver.save_raw_data(demo_data, demo_save_path)
