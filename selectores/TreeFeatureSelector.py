import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from selectores.FeatureSelectorABC import FeatureSelectorABC


class TreeFeatureSelector(FeatureSelectorABC):
    def __init__(self, n_estimators: int = 100):
        self.model = RandomForestClassifier(n_estimators=n_estimators)

    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df.drop(
            columns=["date", "symbol"]
        )  # 必要に応じて日付やシンボル列を除外
        self.model.fit(features, df["target"])  # 'target'は目的変数
        importances = self.model.feature_importances_
        selected_features = features.columns[importances > np.mean(importances)]
        return df[selected_features]
