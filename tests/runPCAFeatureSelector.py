import sys
import os
import pandas as pd

# プロジェクトのルートディレクトリを sys.path に追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from selectores.PCAFeatureSelector import PCAFeatureSelector


def runPCAFeatureSelector(data_path: str, n_components: int = 3):
    """
    PCAFeatureSelector の動作を確認するための関数

    Args:
        data_path (str): データファイルのパス
        n_components (int): 主成分の数
    """
    # データを準備する
    df = pd.read_csv(data_path)
    print("Original DataFrame:")
    print(df.head(10))

    # Unnamed: 0 列をインデックスとして扱い、削除
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df["date"] = pd.to_datetime(df["date"])  # 日付をdatetime型に変換

    # 不要な列を取り除く
    df = df.drop(columns=["date", "symbol", "open", "high", "low", "close", "volume"])

    # PCAセレクターのインスタンスを作成
    pca_selector = PCAFeatureSelector(n_components=n_components)

    # 特徴量を選択
    selected_features_df = pca_selector.select_features(df)

    # インデックスをリセット
    selected_features_df.reset_index(drop=True, inplace=True)

    # 結果を表示
    print("Selected features:")
    print(selected_features_df.head(10))


# 実行例
if __name__ == "__main__":
    data_path = "data/processed/demo_normalized_feature_data.csv"
    runPCAFeatureSelector(data_path, n_components=3)
