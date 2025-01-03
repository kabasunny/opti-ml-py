import sys
import os
import pandas as pd
import numpy as np

# プロジェクトのルートディレクトリを sys.path に追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from selectores.LassoFeatureSelector import LassoFeatureSelector


def runLassoFeatureSelector(data_path: str, alpha: float = 0.01):
    """
    LassoFeatureSelector の動作を確認するための関数

    Args:
        data_path (str): データファイルのパス
        alpha (float): Lasso 回帰の正則化パラメータ
    """
    # データを準備する
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])  # 日付をdatetime型に変換

    # 'target'列を仮定して追加（ここではランダムなデータを使用）
    np.random.seed(0)
    df["target"] = np.random.randint(0, 2, df.shape[0])

    # LassoFeatureSelectorのインスタンスを作成
    lasso_selector = LassoFeatureSelector(alpha=alpha)

    # 特徴量を選択
    selected_features_df = lasso_selector.select_features(df)

    # 結果を表示
    print("Selected features:")
    print(selected_features_df.head(10))
    print(selected_features_df.tail(10))


# 実行例
if __name__ == "__main__":
    data_path = "data/processed/demo_processed_stock_data.csv"
    runLassoFeatureSelector(data_path, alpha=0.01)
