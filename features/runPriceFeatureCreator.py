import sys
import os

# プロジェクトのルートディレクトリを sys.path に追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import pandas as pd
from features.PriceFeatureCreator import (
    PriceFeatureCreator,
)  # PriceFeatureCreatorクラスをインポート

# データを準備する
data_path = os.path.join(project_root, "data/processed/demo_processed_stock_data.csv")
df = pd.read_csv(data_path)
df["date"] = pd.to_datetime(df["date"])  # 日付をdatetime型に変換

# trade_start_date を設定
trade_start_date = pd.Timestamp("2023-08-01")

# PriceFeatureCreatorクラスのインスタンスを作成
creator = PriceFeatureCreator()

# 特徴量を作成
df_with_features = creator.create_features(df, trade_start_date)

# 特徴量が正しく追加されたかを再確認
print(df_with_features[["date", "close", "sma10", "sma30", "bb_up", "bb_low"]].head(10))
print(df_with_features[["date", "close", "sma10", "sma30", "bb_up", "bb_low"]].tail(10))
