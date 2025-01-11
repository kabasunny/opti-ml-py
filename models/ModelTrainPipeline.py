from models.ModelTrainer import ModelTrainer
from models.ModelSaverLoader import ModelSaverLoader
from models.BaseModelABC import BaseModelABC
from data.DataExtractor import DataExtractor
from decorators.ArgsChecker import ArgsChecker
from data.DataManager import DataManager
from typing import List
from models.ModelFactory import ModelFactory


class ModelTrainPipeline:
    @ArgsChecker((None, DataManager, ModelSaverLoader, List[str]), None)
    def __init__(
        self,
        training_and_test_manager: DataManager,
        saver_loader: ModelSaverLoader,
        model_types: List[str],
    ):
        self.training_and_test_manager = training_and_test_manager
        self.models = ModelFactory.create_models(model_types)
        self.saver_loader = saver_loader
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def run(self, symbol):

        # モデルが存在する場合、事前に確認
        if self.saver_loader.check_existing_models(self.models):
            confirm = (
                input("現在のモデルを上書きしてよいですか? (Y/N): ").strip().upper()
            )
            if confirm != "Y":
                print("モデルの上書きをパスしました")
                return

        full_data = self.training_and_test_manager.load_data(symbol)
        self.X_train, self.X_test, self.y_train, self.y_test = (
            DataExtractor.extract_data(full_data)
        )
        self.models, results_df = ModelTrainer.train(
            self.models, self.X_train, self.y_train, self.X_test, self.y_test
        )
        print(results_df)

        self.saver_loader.save_models(self.models)

        print("Model Train Pipeline completed successfully")
