import pandas as pd
from sklearn.linear_model import Lasso
from selectores.FeatureSelectorABC import FeatureSelectorABC


class LassoFeatureSelector(FeatureSelectorABC):
    def __init__(self, alpha: float = 0.01):
        self.lasso = Lasso(alpha=alpha)

    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df.drop(
            columns=["date", "symbol"]
        )  # 必要に応じて日付やシンボル列を除外
        self.lasso.fit(features, df["target"])  # 'target'は目的変数
        selected_features = features.columns[self.lasso.coef_ != 0]
        return df[selected_features]
