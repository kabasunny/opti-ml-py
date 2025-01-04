import sys
import os
import pandas as pd

# プロジェクトのルートディレクトリを sys.path に追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from selectores.CorrelationFeatureSelector import CorrelationFeatureSelector


def runCorrelationFeatureSelector(data_path: str, threshold: float = 0.9):
    """
    CorrelationFeatureSelector の動作を確認するための関数

    Args:
        data_path (str): データファイルのパス
        threshold (float): 相関の閾値
    """
    # データを準備する
    df = pd.read_csv(data_path)
    print("Original DataFrame:")
    print(df.head(10))

    # Unnamed: 0 列を削除
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df["date"] = pd.to_datetime(df["date"])  # 日付をdatetime型に変換

    # 不要な列を削除
    df = df.drop(columns=["date", "symbol", "open", "high", "low", "close", "volume"])

    # CorrelationFeatureSelectorのインスタンスを作成
    correlation_selector = CorrelationFeatureSelector(threshold=threshold)

    # 特徴量を選択
    selected_features_df = correlation_selector.select_features(df)

    # インデックスをリセット
    selected_features_df.reset_index(drop=True, inplace=True)

    # 結果を表示
    print("Selected features:")
    print(selected_features_df.head(10))


# 実行例
if __name__ == "__main__":
    data_path = "data/processed/demo_normalized_feature_data.csv"
    runCorrelationFeatureSelector(data_path, threshold=0.9)
