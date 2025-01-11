from models.ModelTrainer import ModelTrainer
from models.ModelSaverLoader import ModelSaverLoader
from data.DataExtractor import DataExtractor
from data.DataManager import DataManager
from typing import List
from models.ModelFactory import ModelFactory
from decorators.ArgsChecker import ArgsChecker


class ModelPipeline:
    @ArgsChecker((None, DataManager, ModelSaverLoader, List[str]), None)
    def __init__(
        self,
        training_and_test_manager: DataManager,
        saver_loader: ModelSaverLoader,
        model_types: List[str],
    ):
        self.training_and_test_manager = training_and_test_manager
        self.saver_loader = saver_loader
        self.model_types = model_types
        self.model_created = False
        self.models = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def run(self, symbol):
        if self.model_created:
            self.models = self.saver_loader.load_models(self.model_types)
            print("Loaded existing models for retraining")
        else:
            # モデルが存在する場合、事前に確認
            if self.saver_loader.check_existing_models(self.model_types):
                confirm = (
                    input("現在のモデルを上書きしてよいですか? (Y/N): ").strip().upper()
                )
                if confirm != "Y":
                    print("モデルの上書きをパスしました")
                    return
            self.models = ModelFactory.create_models(self.model_types)
            print("Created new models for training")
            self.model_created = True

        full_data = self.training_and_test_manager.load_data(symbol)
        self.X_train, self.X_test, self.y_train, self.y_test = (
            DataExtractor.extract_data(full_data)
        )
        self.models, results_df = ModelTrainer.train(
            self.models, self.X_train, self.y_train, self.X_test, self.y_test
        )

        print(results_df)
        self.saver_loader.save_models(self.models)

        if self.model_created:
            print("Model Re Train Pipeline completed successfully")
        else:
            print("Model Train Pipeline completed successfully")

        self.model_created = True  # モデルが作成されたことを設定
