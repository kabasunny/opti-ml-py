# opti-ml-py\labeling\LabelCreatorABC.py
from abc import ABC, abstractmethod
import pandas as pd


class LabelCreatorABC(ABC):
    @abstractmethod
    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """ラベルを生成する抽象メソッド"""
        pass
