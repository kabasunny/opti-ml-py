import pandas as pd
from features.FeatureCreatorABC import FeatureCreatorABC
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート

class PriceFeatureCreator(FeatureCreatorABC):
    @ArgsChecker((None, pd.DataFrame, pd.Timestamp), pd.DataFrame)
    def create_features(
        self, df: pd.DataFrame, trade_start_date: pd.Timestamp
    ) -> pd.DataFrame:
        """
        価格特徴を生成するメソッド

        Args:
            df (pd.DataFrame): 入力データフレーム
            trade_start_date (pd.Timestamp): トレード開始日

        Returns:
            pd.DataFrame: 価格特徴が追加されたデータフレーム
        """
        # 移動平均 (SMA)
        df["sma10"] = df["close"].rolling(window=10).mean()
        df["sma30"] = df["close"].rolling(window=30).mean()
        df["sma90"] = df["close"].rolling(window=90).mean()
        df["sma180"] = df["close"].rolling(window=180).mean()
        df["sma360"] = df["close"].rolling(window=360).mean()

        # ボリンジャーバンド (BB)
        df["bb_up"] = df["sma10"] + 2 * df["close"].rolling(window=10).std()
        df["bb_low"] = df["sma10"] - 2 * df["close"].rolling(window=10).std()

        # 相対力指数 (RSI)
        window_length = 14
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window_length).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window_length).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # モメンタム (Momentum)
        df["momentum"] = df["close"].diff(periods=10)

        # 平均方向性指数 (ADX)
        def calculate_adx(df, window=14):
            high = df["high"]
            low = df["low"]
            close = df["close"]

            plus_dm = high.diff().where(high.diff() > low.diff(), 0)
            minus_dm = low.diff().where(low.diff() > high.diff(), 0)
            tr = pd.concat([high - low, high - close.shift(), close.shift() - low], axis=1).max(axis=1)
            atr = tr.rolling(window).mean()

            plus_di = 100 * (plus_dm.rolling(window).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(window).mean() / atr)
            dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
            adx = dx.rolling(window).mean()

            return adx

        df["adx"] = calculate_adx(df)

        # MACD（移動平均収束拡散法）
        exp1 = df["close"].ewm(span=12, adjust=False).mean()
        exp2 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = exp1 - exp2
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # フィルタリングせずに戻す
        return df
