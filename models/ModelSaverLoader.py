# opti-ml-py\models\ModelSaverLoader.py
import os
import pickle
from typing import List
from decorators.ArgsChecker import ArgsChecker


class ModelSaverLoader:
    @ArgsChecker((None, list, list), None)
    def save_models(self, models: List[object], filepaths: List[str]):
        for model, filepath in zip(models, filepaths):
            # ディレクトリが存在しない場合は作成
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "wb") as file:
                pickle.dump(model, file)

    @ArgsChecker((None, list[str]), list)
    def load_models(self, filepaths: List[str]) -> List[object]:
        models = []
        for filepath in filepaths:
            with open(filepath, "rb") as file:
                model = pickle.load(file)
                models.append(model)
        return models
