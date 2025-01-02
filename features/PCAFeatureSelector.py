import pandas as pd
from sklearn.decomposition import PCA
from features.FeatureSelectorABC import FeatureSelectorABC
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート


class PCAFeatureSelector(FeatureSelectorABC):
    def __init__(self, n_components: int):
        self.pca = PCA(n_components=n_components)

    @ArgsChecker((pd.DataFrame,), pd.DataFrame)
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        PCAに基づいて特徴量を選択するメソッド

        Args:
            df (pd.DataFrame): 入力データフレーム

        Returns:
            pd.DataFrame: 選択された特徴量のみを含むデータフレーム
        """
        features = df.drop(columns=["label"])  # 'label' 列を除外して特徴量のみを抽出
        selected_features = self.pca.fit_transform(features)
        return pd.DataFrame(selected_features)
