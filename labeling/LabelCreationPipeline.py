# opti-ml-py\labeling\LabelCreationPipeline.py
import pandas as pd
from decorators.ArgsChecker import ArgsChecker
from data.LabelDataManager import LabelDataManager
from data.RawDataManager import RawDataManager
from labeling.LabelCreatorABC import LabelCreatorABC


class LabelCreationPipeline:
    @ArgsChecker((None, RawDataManager, LabelDataManager, LabelCreatorABC), None)
    def __init__(self, raw_data_manager, label_data_manager, label_creator):
        self.raw_data_manager = raw_data_manager
        self.label_data_manager = label_data_manager
        self.label_creator = label_creator

    @ArgsChecker((None,), None)
    def run(self):
        """データパイプラインの実行"""
        # データの読み込み
        df = self.raw_data_manager.load_raw_data()
        print("Raw data loaded successfully")

        # ラベルの作成
        labels = self.label_creator.create_labels(df)
        print("Labels created")

        # ラベルデータの保存
        self.label_data_manager.save_label_data(labels)
        print("Labels saved successfully")

        print("Label creation pipeline completed successfully.")