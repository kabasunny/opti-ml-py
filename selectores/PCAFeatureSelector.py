import pandas as pd
from sklearn.decomposition import PCA
from selectores.FeatureSelectorABC import FeatureSelectorABC
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート


class PCAFeatureSelector:
    def __init__(self, n_components: int):
        self.pca = PCA(n_components=n_components)
        self.feature_names = None

    @ArgsChecker((None, pd.DataFrame), pd.DataFrame)
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        PCAに基づいて特徴量を選択するメソッド

        Args:
            df (pd.DataFrame): 入力データフレーム

        Returns:
            pd.DataFrame: 選択された特徴量のみを含むデータフレーム
        """
        features = df.drop(columns=["date", "symbol"])  # 'date'と'symbol'列を除外
        self.pca.fit(features)
        selected_features = self.pca.transform(features)
        self.feature_names = [f"PC{i+1}" for i in range(selected_features.shape[1])]

        # 主成分の寄与度を表示
        loadings = pd.DataFrame(
            self.pca.components_.T, columns=self.feature_names, index=features.columns
        )
        print("PCA Loadings (各主成分への寄与度):")
        print(loadings)

        return pd.DataFrame(selected_features, columns=self.feature_names)
