from selectores.PCAFeatureSelector import PCAFeatureSelector
from selectores.CorrelationFeatureSelector import CorrelationFeatureSelector
from selectores.LassoFeatureSelector import LassoFeatureSelector
from selectores.TreeFeatureSelector import TreeFeatureSelector


class SelectorFactory:
    """
    SelectorFactoryクラスは、各種特徴量選択器を生成するためのファクトリクラス。
    """

    @staticmethod
    def create_selectors():
        tree_selector = TreeFeatureSelector(
            n_estimators=10, random_state=42
        )  # より厳しい設定（n_estimatorsを減らす）
        lasso_selector = LassoFeatureSelector(
            alpha=0.005
        )  # より厳しい設定（alphaを増やす）
        pca_selector = PCAFeatureSelector(
            n_components=3
        )  # より厳しい設定（n_componentsを減らす）
        corr_selector = CorrelationFeatureSelector(
            threshold=0.5
        )  # より厳しい設定（相関の閾値を下げる）
        selectors = [tree_selector, lasso_selector, pca_selector, corr_selector]
        return selectors
