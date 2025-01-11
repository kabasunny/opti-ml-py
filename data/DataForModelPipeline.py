from data.DataForModelPreparation import DataForModelPreparation
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート
from data.DataManager import DataManager


class DataForModelPipeline:
    @ArgsChecker(
        (
            None,
            DataManager,
            DataManager,
            DataManager,
            DataManager,
        ),
        None,
    )
    def __init__(
        self,
        label_data_manager,
        selected_feature_manager,
        training_and_test_data_manager,
        practical_data_manager,
    ):
        self.label_data_manager = label_data_manager
        self.selected_feature_manager = selected_feature_manager
        self.training_and_test_data_manager = training_and_test_data_manager
        self.practical_data_manager = practical_data_manager

    @ArgsChecker((None, str), None)
    def run(self, symbol):
        # print("Run Data For Model Pipeline")

        # ラベル付きのデータを作成する
        full_data = DataForModelPreparation.create_full_data(
            self.label_data_manager,
            self.selected_feature_manager,
            symbol,
        )
        correct_data, incorrect_data = DataForModelPreparation.split_data_by_label(
            full_data
        )

        # データをモデル用に分割
        (
            correct_data_train_eval,
            correct_data_practical_test,
            incorrect_data_train_eval,
            incorrect_data_practical_test,
        ) = DataForModelPreparation.split_data_for_modeling(
            correct_data, incorrect_data
        )

        # 訓練データとテストデータを準備
        combined_data = DataForModelPreparation.prepare_training_and_test_data(
            correct_data_train_eval,
            incorrect_data_train_eval,
        )
        # print(f"combined_data{len(combined_data)}")

        # 訓練データとテストデータを保存
        self.training_and_test_data_manager.save_data(combined_data, symbol)

        # 実践テストデータを準備
        practical_data = DataForModelPreparation.prepare_practical_data(
            correct_data_practical_test, incorrect_data_practical_test
        )
        # print(f"practical_data{len(practical_data)}")
        # 実践テストデータを保存
        self.practical_data_manager.save_data(practical_data, symbol)

        print("Data For Model Pipeline completed successfully")
