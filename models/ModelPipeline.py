# opti-ml-py\models\ModelPipeline.py
from models.ModelTrainer import ModelTrainer
from models.ModelSaverLoader import ModelSaverLoader
from models.BaseModelABC import BaseModelABC
from data.DataExtractor import DataExtractor
from decorators.ArgsChecker import ArgsChecker
from data.DataManager import DataManager


class ModelPipeline:
    @ArgsChecker((None, DataManager, list, ModelSaverLoader), None)
    def __init__(
        self,
        training_and_test_manager: DataManager,
        models: list[BaseModelABC],
        saver_loader: ModelSaverLoader,
    ):
        self.training_and_test_manager = training_and_test_manager
        self.models = models
        self.saver_loader = saver_loader
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def extract_data(self):
        full_data = self.training_and_test_manager.load_data()
        self.X_train, self.X_test, self.y_train, self.y_test = (
            DataExtractor.extract_data(full_data)
        )

    def train_models(self):
        self.models = ModelTrainer.train(
            self.models, self.X_train, self.y_train, self.X_test, self.y_test
        )

    def save_models(self, save_paths: list[str]):
        self.saver_loader.save_models(self.models, save_paths)

    def load_models(self, load_paths: list[str]):
        self.models = self.saver_loader.load_models(load_paths)

    def evaluate_models(self):
        ModelTrainer.evaluate(self.models, self.X_test, self.y_test)
