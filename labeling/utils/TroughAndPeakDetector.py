import numpy as np
from scipy.signal import find_peaks
import pandas as pd
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート


class TroughAndPeakDetector:
    @staticmethod
    @ArgsChecker((pd.Series,), pd.Series)
    def detect_troughs(prices: pd.Series) -> pd.Series:
        """
        トラフ（谷）検出関数 終値のデータから谷（安値）を検出

        Args:
            prices (pd.Series): 株価の終値データ

        Returns:
            pd.Series: トラフのインデックスを含む配列
        """
        troughs, _ = find_peaks(-prices)
        return pd.Series(troughs)  # インデックスを Series に変換して返す

    @staticmethod
    @ArgsChecker((pd.Series,), pd.Series)
    def detect_peaks(prices: pd.Series) -> pd.Series:
        """
        ピーク（山）検出関数 終値のデータから山（高値）を検出

        Args:
            prices (pd.Series): 株価の終値データ

        Returns:
            pd.Series: ピークのインデックスを含む配列
        """
        peaks, _ = find_peaks(prices)
        return pd.Series(peaks)  # インデックスを Series に変換して返す
