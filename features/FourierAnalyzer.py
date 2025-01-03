import pandas as pd
import numpy as np
from numpy.fft import fft
from features.FeatureCreatorABC import FeatureCreatorABC
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート


class FourierAnalyzer(FeatureCreatorABC):
    @ArgsChecker((None, pd.DataFrame, pd.Timestamp), pd.DataFrame)
    def create_features(
        self, df: pd.DataFrame, trade_start_date: pd.Timestamp
    ) -> pd.DataFrame:
        """
        フーリエ解析に基づいて特徴量を作成するメソッド

        Args:
            df (pd.DataFrame): 入力データフレーム
            trade_start_date (pd.Timestamp): トレード開始日

        Returns:
            pd.DataFrame: 特徴量が追加されたデータフレーム
        """
        # 特徴量列を初期化
        for i in range(2, 5):  # ff2, ff3, ff4 のみを初期化
            df[f"ff{i}"] = np.nan

        # フーリエ変換を行う関数
        def fft_analysis(prices):
            n = len(prices)
            fft_result = fft(prices)
            fft_amplitude = np.abs(fft_result)[: n // 2]
            fft_freq = np.fft.fftfreq(n, d=1)[: n // 2]

            # 周期計算
            non_zero_idx = fft_freq > 0
            fft_period = 1 / fft_freq[non_zero_idx]
            fft_amplitude = fft_amplitude[non_zero_idx]

            # 振幅が大きい順に並べ替える
            sorted_indices = np.argsort(fft_amplitude)[::-1]
            sorted_periods = fft_period[sorted_indices]

            # 上位6つの支配的な周期を取得
            return sorted_periods[:6]

        # `trade_start_date` のインデックスを取得
        start_idx = df.index[df["date"] >= trade_start_date][0]

        # 各日付に対してフーリエ特徴量を計算
        for idx in range(start_idx, len(df)):
            # 利用可能なデータ範囲を計算
            window_start_idx = max(0, idx - (start_idx - 0))
            window_data = df.iloc[window_start_idx : idx + 1]["close"]

            # フーリエ特徴量を計算
            dominant_periods = fft_analysis(window_data.values)

            # 結果を格納
            for i, period in enumerate(
                dominant_periods[2:5]
            ):  # ff2, ff3, ff4 のみを格納
                df.at[idx, f"ff{i + 2}"] = period

        return df
