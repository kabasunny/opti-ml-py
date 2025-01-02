import pandas as pd
from sklearn.decomposition import PCA
from features.FeatureSelectorABC import FeatureSelectorABC
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート


class PCAFeatureSelector(FeatureSelectorABC):
    def __init__(self, n_components: int):
        self.pca = PCA(n_components=n_components)

    @ArgsChecker(
        (
            None,
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        PCAに基づいて特徴量を選択するメソッド

        Args:
            df (pd.DataFrame): 入力データフレーム

        Returns:
            pd.DataFrame: 選択された特徴量のみを含むデータフレーム
        """
        # 'date' 列を除外して特徴量のみを抽出
        features = df.drop(columns=["date"])

        # NaN 値を削除または補完（ここでは削除を選択）
        features = features.dropna()

        selected_features = self.pca.fit_transform(features)
        return pd.DataFrame(selected_features)
