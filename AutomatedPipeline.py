from data.YahooFinanceStockDataFetcher import YahooFinanceStockDataFetcher
from data.DataManager import DataManager
from data.RawDataPipeline import RawDataPipeline
from preprocessing.PreprocessPipeline import PreprocessPipeline
from labeling.LabelCreatePipeline import LabelCreatePipeline
from labeling.TroughLabelCreator import TroughLabelCreator
from features.FeaturePipeline import FeaturePipeline
from selectores.SelectorPipeline import SelectorPipeline
from data.DataForModelPipeline import DataForModelPipeline
from features.AnalyzerFactory import AnalyzerFactory
from selectores.SelectorFactory import SelectorFactory
from models.ModelPipeline import ModelPipeline
from models.ModelPredictPipeline import ModelPredictPipeline


class AutomatedPipeline:
    def __init__(
        self,
        before_period_days,  # 特徴量生成に必要な日数
        model_types,
        feature_list_str,
        model_saver_loader,
        data_managers,
    ):
        self.before_period_days = before_period_days
        self.model_types = model_types
        self.feature_list_str = feature_list_str
        self.model_saver_loader = model_saver_loader
        self.model_created = False  # モデルが作成済みかどうかのフラグ

        self.data_managers = data_managers

        # 各パイプラインをインスタンス変数として保持
        self.raw_data_pipeline = RawDataPipeline(
            self.data_managers["formated_raw"],
            fetcher=YahooFinanceStockDataFetcher(),
        )
        self.preprocess_pipeline = PreprocessPipeline(
            self.data_managers["formated_raw"], self.data_managers["processed_raw"]
        )
        self.label_create_pipeline = LabelCreatePipeline(
            self.data_managers["formated_raw"],
            self.data_managers["labeled"],
            self.before_period_days,
            TroughLabelCreator(),
        )
        self.feature_pipeline = FeaturePipeline(
            self.data_managers["processed_raw"],
            self.data_managers["normalized_feature"],
            self.before_period_days,
            AnalyzerFactory.create_analyzers(self.feature_list_str),
        )
        self.selector_pipeline = SelectorPipeline(
            self.data_managers["labeled"],
            self.data_managers["normalized_feature"],
            self.data_managers["selected_feature"],
            SelectorFactory.create_selectors(),
        )
        self.data_for_model_pipeline = DataForModelPipeline(
            self.data_managers["labeled"],
            self.data_managers["selected_feature"],
            self.data_managers["training_and_test"],
            self.data_managers["practical"],
        )
        self.model_pipeline = ModelPipeline(
            self.data_managers["training_and_test"],
            self.model_saver_loader,
            self.model_types,
        )
        self.model_predict_pipeline = ModelPredictPipeline(
            self.model_saver_loader,
            self.data_managers["training_and_test"],
            self.data_managers["practical"],
            self.data_managers["predictions"],
            self.model_types,
        )

    def process_symbol(self, symbol):
        print(f"Symbol of current data: {symbol}")

        try:
            self.raw_data_pipeline.run(symbol)

            self.preprocess_pipeline.run(symbol)
            self.label_create_pipeline.run(symbol)
            self.feature_pipeline.run(symbol)
            self.selector_pipeline.run(symbol)
            self.data_for_model_pipeline.run(symbol)

            self.model_pipeline.run(symbol)
            self.model_predict_pipeline.run(symbol)

        except Exception as e:
            print(f"{symbol} の処理中にエラーが発生しました: {e}")
