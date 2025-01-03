import sys
import os
import pandas as pd  # trade_start_date のために必要

# プロジェクトのルートディレクトリを sys.path に追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from selectores.PCAFeatureSelector import PCAFeatureSelector
from selectores.CorrelationFeatureSelector import CorrelationFeatureSelector
from selectores.LassoFeatureSelector import LassoFeatureSelector
from selectores.TreeFeatureSelector import TreeFeatureSelector
from selectores.SelectorPipeline import SelectorPipeline
from data.DataManager import DataManager

if __name__ == "__main__":
    target_data_path = "data/label/demo_labels.csv"
    normalized_feature_data_path = "data/processed/demo_normalized_feature_data.csv"
    selected_feature_data_path = "data/processed/demo_selected_feature_data.csv"

    # データマネージャのインスタンスを作成
    target_data_manager = DataManager(target_data_path)
    normalized_f_d_manager = DataManager(normalized_feature_data_path)
    selected_f_d_manager = DataManager(selected_feature_data_path)

    # パイプラインの手順を定義
    tree_selector = TreeFeatureSelector(
        n_estimators=1
    )  # より厳しい設定（n_estimatorsを減らす）
    lasso_selector = LassoFeatureSelector(
        alpha=0.008
    )  # より厳しい設定（alphaを増やす）
    pca_selector = PCAFeatureSelector(
        n_components=2
    )  # より厳しい設定（n_componentsを減らす）
    corr_selector = CorrelationFeatureSelector(
        threshold=0.2
    )  # より厳しい設定（相関の閾値を下げる）
    selectors = [tree_selector, lasso_selector, pca_selector, corr_selector]

    # SelectorPipeline のインスタンスを作成し、実行
    pipeline = SelectorPipeline(
        target_data_manager, normalized_f_d_manager, selected_f_d_manager, selectors
    )
    selected_features_df = pipeline.run()

    # 結果を表示
    print("Selected features:")
    print(selected_features_df.head(10))
