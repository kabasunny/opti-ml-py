import pandas as pd
from features.CombinedFeatureCreator import CombinedFeatureCreator
from features.CombinedFeatureSelector import CombinedFeatureSelector
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート
from data.ProcessedDataManager import ProcessedDataManager
from data.FeatureDataManager import FeatureDataManager


class FeaturePipeline:
    @ArgsChecker(
        (None, ProcessedDataManager, FeatureDataManager, list, pd.Timestamp), None
    )
    def __init__(
        self,
        processed_data_manager: ProcessedDataManager,
        feature_data_manager: FeatureDataManager,
        feature_list_str: list,
        trade_start_date: pd.Timestamp,
    ):
        self.feature_creator = CombinedFeatureCreator(
            feature_list_str, trade_start_date
        )
        self.feature_selector = CombinedFeatureSelector()
        self.processed_data_manager = processed_data_manager
        self.feature_data_manager = feature_data_manager

    @ArgsChecker((None,), pd.DataFrame)
    def run(self) -> pd.DataFrame:
        """
        特徴量作成と選択を一連の流れで実行するメソッド

        Returns:
            pd.DataFrame: 選択された特徴量のみを含むデータフレーム
        """
        # データをロード
        df = self.processed_data_manager.load_processed_data()

        # 特徴量を作成
        df_with_features = self.feature_creator.create_all_features(df)

        # 特徴量を選択
        selected_features_df = self.feature_selector.select_all_features(
            df_with_features
        )

        # 特徴量データを保存
        self.feature_data_manager.save_feature_data(selected_features_df)

        return selected_features_df
