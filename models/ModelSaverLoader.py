import os
import pickle
from typing import List
from datetime import datetime


class ModelSaverLoader:
    def __init__(self, model_base_path: str, model_file_ext: str):
        self.model_base_path = model_base_path
        self.model_file_ext = model_file_ext

    def save_models(self, models: List[object]):
        today_date = datetime.today().strftime("%Y%m%d")
        for model in models:
            model_name = model.__class__.__name__.replace("Model", "")
            filepath = f"{self.model_base_path}/{model_name}_{today_date}.{self.model_file_ext}"
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "wb") as file:
                pickle.dump(model, file)

    def load_models(self, model_types: List[str]) -> List[object]:
        models = []
        print(f"Loading models from {self.model_base_path}")
        for model_type in model_types:
            for filename in os.listdir(self.model_base_path):
                if model_type in filename and filename.endswith(self.model_file_ext):
                    filepath = os.path.join(self.model_base_path, filename)
                    # print(f"Trying to load model from {filepath}")
                    with open(filepath, "rb") as file:
                        model = pickle.load(file)
                        models.append(model)
                        # print(f"Loaded model {model_type}")
                        break
            else:
                print(f"Model file for {model_type} does not exist")
        return models
