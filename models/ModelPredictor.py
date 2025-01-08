from typing import List
import pandas as pd
from models.BaseModelABC import BaseModelABC


class ModelPredictor:
    @staticmethod
    def predict(models: List[BaseModelABC], X_test: pd.DataFrame) -> pd.DataFrame:
        predictions = {}
        for model in models:
            model_name = type(model).__name__.replace("Model", "")
            predictions[model_name] = model.predict(X_test)
        predictions_df = pd.DataFrame(predictions)
        return predictions_df

    @staticmethod
    def evaluate(
        models: List[BaseModelABC], feature: pd.DataFrame, label: pd.Series
    ) -> pd.DataFrame:
        evaluations = []
        model_names = []
        for model in models:
            model_name = type(model).__name__.replace("Model", "")
            evaluations.append(model.evaluate(feature, label))
            model_names.append(model_name)
        evaluations_df = pd.DataFrame(
            evaluations,
            columns=["Accuracy", "Precision", "Recall", "F1"],
            index=model_names,
        )
        return evaluations_df
