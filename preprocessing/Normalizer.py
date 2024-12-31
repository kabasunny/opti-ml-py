# preprocessing\Normalizer.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート


class Normalizer:
    @staticmethod
    @ArgsChecker((pd.DataFrame,), pd.DataFrame)
    def normalize(df: pd.DataFrame) -> pd.DataFrame:
        """
        データを0から1の範囲に正規化するメソッド。

        データフレーム内の数値データを0と1の範囲にスケーリングし、
        異なるスケールのデータを統一されたスケールに変換。
        """
        numeric_cols = df.select_dtypes(include="number").columns
        numeric_cols = [
            col for col in numeric_cols if col != "symbol"
        ]  # 'symbol'を除外
        scaler = MinMaxScaler()  # MinMaxScalerをインスタンス化
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])  # データを正規化S
        return df
