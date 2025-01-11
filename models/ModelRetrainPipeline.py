from models.ModelTrainer import ModelTrainer
from models.ModelSaverLoader import ModelSaverLoader
from data.DataExtractor import DataExtractor
from data.DataManager import DataManager
from typing import List


class ModelRetrainPipeline:
    def __init__(
        self,
        training_and_test_manager: DataManager,
        saver_loader: ModelSaverLoader,
        model_types: List[str],  # モデルタイプを追加
    ):
        self.training_and_test_manager = training_and_test_manager
        self.saver_loader = saver_loader
        self.model_types = model_types  # モデルタイプを保持
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = None

    def run(self, symbol):
        full_data = self.training_and_test_manager.load_data(symbol)
        self.X_train, self.X_test, self.y_train, self.y_test = (
            DataExtractor.extract_data(full_data)
        )
        self.models = self.saver_loader.load_models(self.model_types)  # モデルをロード
        self.models, results_df = ModelTrainer.train(
            self.models, self.X_train, self.y_train, self.X_test, self.y_test
        )
        print(results_df)
        self.saver_loader.save_models(self.models)

        print("Model Re Train Pipeline completed successfully")
