import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from features.FeatureCreatorABC import FeatureCreatorABC
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート


class PeakTroughAnalyzer(FeatureCreatorABC):
    @ArgsChecker((None, pd.DataFrame, pd.Timestamp), pd.DataFrame)
    def create_features(
        self, df: pd.DataFrame, trade_start_date: pd.Timestamp
    ) -> pd.DataFrame:
        """
        ピーク・トラフ解析に基づいて特徴量を作成するメソッド

        Args:
            df (pd.DataFrame): 入力データフレーム
            trade_start_date (pd.Timestamp): トレード開始日

        Returns:
            pd.DataFrame: 特徴量が追加されたデータフレーム
        """
        # 特徴量名と設定
        feature_configs = {
            "50dtme": {"window_size": 50, "resample": None},
            "30wtme": {"window_size": 30, "resample": "W"},
            "24mtme": {"window_size": 24, "resample": "ME"},
        }

        # 特徴量列を初期化
        for feature_name in feature_configs:
            df[feature_name] = np.nan

        # 共通の特徴量計算関数
        def calculate_feature(data, feature_name, config):
            window_size = config["window_size"]
            resample_rule = config["resample"]

            # データをリサンプリング（必要に応じて）
            if resample_rule:
                data = data.set_index("date").resample(resample_rule).last()

            for idx in range(len(df)):
                current_date = df.iloc[idx]["date"]

                # trade_start_date 以降のデータのみ特徴量を計算
                if current_date < trade_start_date:
                    continue

                # 現在のウィンドウを取得
                if resample_rule:
                    try:
                        window_end_idx = pd.Index(data.index).get_indexer(
                            [current_date], method="ffill"
                        )[0]
                    except IndexError:
                        print(f"データが見つかりません: {current_date}")
                        continue
                else:
                    window_end_idx = df.index[idx]

                window_start_idx = max(0, window_end_idx - window_size)
                window_data = data.iloc[window_start_idx:window_end_idx]

                # トラフの検出
                prices = window_data["close"].values
                troughs, _ = find_peaks(-prices)

                # トラフ間隔の平均を計算
                if len(troughs) > 1:
                    intervals = np.diff(troughs)
                    mean_interval = np.mean(intervals)
                else:
                    mean_interval = np.nan

                # 結果を格納
                df.loc[df["date"] == current_date, feature_name] = mean_interval

        # 特徴量計算を適用
        for feature_name, config in feature_configs.items():
            calculate_feature(df, feature_name, config)

        # フィルタリングせずに戻す
        return df
