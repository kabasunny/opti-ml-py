from selectores.PCAFeatureSelector import PCAFeatureSelector
from selectores.CorrelationFeatureSelector import CorrelationFeatureSelector
from selectores.LassoFeatureSelector import LassoFeatureSelector
from selectores.TreeFeatureSelector import TreeFeatureSelector
from selectores.SelectAllSelector import SelectAllSelector


class SelectorFactory:
    """
    SelectorFactoryクラスは、各種特徴量選択器を生成するためのファクトリクラス。
    """

    @staticmethod
    def create_selectors(selector_names):
        selectors = []
        for selector_name in selector_names:
            # コメントアウトは選択閾値を緩める方法
            if selector_name == "Tree":
                selectors.append(TreeFeatureSelector(n_estimators=50, random_state=42))  # n_estimatorsを増やす
            elif selector_name == "Lasso":
                selectors.append(LassoFeatureSelector(alpha=0.001))  # alphaを減らす
            elif selector_name == "PCA":
                selectors.append(PCAFeatureSelector(n_components=5))  # n_componentsを増やす
            elif selector_name == "Correlation":
                selectors.append(CorrelationFeatureSelector(threshold=0.8))  # thresholdを増やす
            elif selector_name == "AllSelect": 
                selectors.append(SelectAllSelector())
            else:
                raise ValueError(f"Unknown selector: {selector_name}")
        return selectors
