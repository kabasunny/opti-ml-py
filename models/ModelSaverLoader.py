import os
import pickle
from typing import List
from decorators.ArgsChecker import ArgsChecker


class ModelSaverLoader:
    def __init__(self, filepaths: List[str]):
        self.filepaths = filepaths

    def save_models(self, models: List[object]):
        for model, filepath in zip(models, self.filepaths):
            # ディレクトリが存在しない場合は作成
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "wb") as file:
                pickle.dump(model, file)

    def load_models(self) -> List[object]:
        models = []
        for filepath in self.filepaths:
            with open(filepath, "rb") as file:
                model = pickle.load(file)
                models.append(model)
        return models
