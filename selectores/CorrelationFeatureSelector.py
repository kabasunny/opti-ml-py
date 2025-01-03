import pandas as pd
import numpy as np
from selectores.FeatureSelectorABC import FeatureSelectorABC


class CorrelationFeatureSelector(FeatureSelectorABC):
    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold

    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        correlation_matrix = df.corr().abs()
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool_)
        )
        to_drop = [
            column
            for column in upper_triangle.columns
            if any(upper_triangle[column] > self.threshold)
        ]
        selected_features = df.drop(columns=to_drop, axis=1)
        return selected_features
