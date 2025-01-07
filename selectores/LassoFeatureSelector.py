import pandas as pd
from sklearn.linear_model import Lasso
from selectores.SupervisedFeatureSelectorABC import SupervisedFeatureSelectorABC
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート


class LassoFeatureSelector(SupervisedFeatureSelectorABC):
    """
    LassoFeatureSelectorクラスは、Lasso回帰に基づいて特徴量を選択するためのクラス
    L1正則化により、不要な特徴量の係数をゼロにすることで特徴選択を行い、過学習を防ぎながら重要な特徴量を抽出

    Attributes:
        lasso (Lasso): Lasso回帰モデル
    """

    @ArgsChecker((None, float), None)
    def __init__(self, alpha: float = 0.01):
        self.lasso = Lasso(alpha=alpha)

    @ArgsChecker((None, pd.DataFrame, str), pd.DataFrame)
    def select_features(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """
        Lasso回帰に基づいて特徴量を選択するメソッド

        Args:
            df (pd.DataFrame): 入力データフレーム
            target_column (str): 目的変数（ターゲットラベル）のカラム名

        Returns:
            pd.DataFrame: 選択された特徴量のみを含むデータフレーム
        """
        features = df.drop(columns=[target_column])  # ターゲット列を除外
        target = df[target_column]
        # Lasso回帰を適用
        self.lasso.fit(features, target)
        selected_features = features.columns[self.lasso.coef_ != 0]
        selected_df = df[selected_features]
        return selected_df
