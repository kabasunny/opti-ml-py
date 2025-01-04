import sys
import os
import pandas as pd

# プロジェクトのルートディレクトリを sys.path に追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from selectores.TreeFeatureSelector import TreeFeatureSelector


def runTreeFeatureSelector(
    data_path: str,
    target_data_path: str,
    n_estimators: int = 100,
    target_column: str = "label",
):
    """
    TreeFeatureSelector の動作を確認するための関数

    Args:
        data_path (str): 特徴量データファイルのパス
        target_data_path (str): 正解ラベルデータファイルのパス
        n_estimators (int): RandomForestClassifierの木の数
        target_column (str): 正解ラベルのカラム名
    """
    # 特徴量データを準備する
    df = pd.read_csv(data_path)
    print("Original DataFrame:")
    print(df.head(10))

    # Unnamed: 0 列を削除
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df["date"] = pd.to_datetime(df["date"])  # 日付をdatetime型に変換

    # 正解ラベルデータを準備する
    target_df = pd.read_csv(target_data_path)
    print("Target DataFrame:")
    print(target_df.head(10))

    # ターゲットデータのdate列もdatetime型に変換
    target_df["date"] = pd.to_datetime(target_df["date"])

    # 正解ラベルデータをマージ
    df = df.merge(
        target_df[["date", "symbol", target_column]], on=["date", "symbol"], how="left"
    )

    # 不要な列を削除
    columns_to_drop = [
        "Unnamed: 0",
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    # TreeFeatureSelectorのインスタンスを作成
    tree_selector = TreeFeatureSelector(n_estimators=n_estimators)

    # 特徴量を選択
    selected_features_df = tree_selector.select_features(df, target_column)

    # インデックスをリセット
    selected_features_df.reset_index(drop=True, inplace=True)

    # 結果を表示
    print("Selected features:")
    print(selected_features_df.head(10))


# 実行例
if __name__ == "__main__":
    data_path = "data/processed/demo_normalized_feature_data.csv"
    target_data_path = "data/label/demo_labels.csv"
    runTreeFeatureSelector(
        data_path, target_data_path, n_estimators=100, target_column="label"
    )
