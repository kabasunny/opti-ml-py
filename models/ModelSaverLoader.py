import os
import pickle
from typing import List


class ModelSaverLoader:
    def __init__(self, date_str: str, model_save_path: str, model_file_ext: str):
        self.model_base_path = model_save_path
        self.model_file_ext = model_file_ext
        self.date_str = date_str

    def generate_path(self, model_name: str) -> str:
        return (
            f"{self.model_base_path}/{self.date_str}/{model_name}.{self.model_file_ext}"
        )

    def save_models(self, models: List[object]):
        for model in models:
            model_name = model.__class__.__name__.replace("Model", "")
            filepath = self.generate_path(model_name)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "wb") as file:
                pickle.dump(model, file)
            # print(f"モデルが {filepath} に保存されました")

    def load_models(self, model_types: List[str]) -> List[object]:
        models = []
        dir_path = f"{self.model_base_path}/{self.date_str}/"
        if not os.path.exists(dir_path):
            # 日付のインスタンス変数に合致するディレクトリがない場合、過去の日付を探索
            parent_dir = f"{self.model_base_path}/"
            dir_list = sorted(os.listdir(parent_dir), reverse=True)
            for d in dir_list:
                potential_path = os.path.join(parent_dir, d)
                if os.path.isdir(potential_path):
                    dir_path = potential_path
                    break

        for model_type in model_types:
            for filename in os.listdir(dir_path):
                if model_type in filename and filename.endswith(self.model_file_ext):
                    filepath = os.path.join(dir_path, filename)
                    with open(filepath, "rb") as file:
                        model = pickle.load(file)
                        models.append(model)
                    break
            else:
                print(f"{model_type}のモデルファイルが存在しません。")
        return models

    def check_existing_models(self, model_names: List[str]) -> str:
        """保存済みモデルが存在するかチェック"""
        parent_dir = f"{self.model_base_path}/"
        dir_list = sorted(os.listdir(parent_dir), reverse=True)
        for d in dir_list:
            potential_path = os.path.join(parent_dir, d)
            if os.path.isdir(potential_path):
                for model_name in model_names:
                    filepath = f"{potential_path}/{model_name}.{self.model_file_ext}"
                    if os.path.exists(filepath):
                        return d
        return None
