from models.ModelTrainer import ModelTrainer
from models.ModelSaverLoader import ModelSaverLoader
from models.BaseModelABC import BaseModelABC
from data.DataExtractor import DataExtractor
from decorators.ArgsChecker import ArgsChecker
from data.DataManager import DataManager
from typing import List


class ModelTrainPipeline:
    @ArgsChecker((None, DataManager, List[BaseModelABC], ModelSaverLoader), None)
    def __init__(
        self,
        training_and_test_manager: DataManager,
        models: List[BaseModelABC],
        saver_loader: ModelSaverLoader,
    ):
        self.training_and_test_manager = training_and_test_manager
        self.models = models
        self.saver_loader = saver_loader
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def run(self):
        full_data = self.training_and_test_manager.load_data()
        self.X_train, self.X_test, self.y_train, self.y_test = (
            DataExtractor.extract_data(full_data)
        )
        self.models, results_df = ModelTrainer.train(
            self.models, self.X_train, self.y_train, self.X_test, self.y_test
        )
        print(results_df)

        # モデルの保存前に確認
        if self.saver_loader.check_existing_models(self.models):
            confirm = (
                input("モデルが既に存在します。上書きしてよいですか？ (Y/N): ")
                .strip()
                .upper()
            )
            if confirm != "Y":
                print("モデルの保存をキャンセルしました。")
                return

        self.saver_loader.save_models(self.models)
        # self.models = self.saver_loader.load_models()
        # ModelTrainer.evaluate(self.models, self.X_test, self.y_test)

        print("Model Train Pipeline completed successfully")
