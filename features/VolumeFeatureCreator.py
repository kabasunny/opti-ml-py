import pandas as pd
import numpy as np  # numpy をインポート
from features.FeatureCreatorABC import FeatureCreatorABC
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート

class VolumeFeatureCreator(FeatureCreatorABC):
    @ArgsChecker((None, pd.DataFrame, pd.Timestamp), pd.DataFrame)
    def create_features(
        self, df: pd.DataFrame, trade_start_date: pd.Timestamp
    ) -> pd.DataFrame:
        """
        出来高特徴を生成するメソッド

        Args:
            df (pd.DataFrame): 入力データフレーム
            trade_start_date (pd.Timestamp): トレード開始日

        Returns:
            pd.DataFrame: 出来高特徴が追加されたデータフレーム
        """
        # 出来高の移動平均
        df["vsma10"] = df["volume"].rolling(window=10).mean()
        df["vsma30"] = df["volume"].rolling(window=30).mean()

        # 出来高の標準偏差
        df["vstd10"] = df["volume"].rolling(window=10).std().fillna(0)
        df["vstd30"] = df["volume"].rolling(window=30).std().fillna(0)

        # 出来高の変動率 (Volume Rate of Change, VROC)
        df["vroc"] = df["volume"].pct_change(periods=10).replace([np.inf, -np.inf], np.nan).fillna(0)

        # 出来高ボリンジャーバンド
        df["v_bb_up"] = df["vsma10"] + 2 * df["vstd10"]
        df["v_bb_low"] = df["vsma10"] - 2 * df["vstd10"]

        # フィルタリングせずに戻す
        return df
