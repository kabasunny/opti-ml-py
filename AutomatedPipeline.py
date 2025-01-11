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
from models.ModelTrainPipeline import ModelTrainPipeline
from models.ModelRetrainPipeline import ModelRetrainPipeline
from models.ModelPredictPipeline import ModelPredictPipeline


class AutomatedPipeline:
    def __init__(
        self,
        trade_start_date,
        data_start_period,
        end_date,
        model_types,
        feature_list_str,
        model_saver_loader,
        data_managers,
    ):
        self.trade_start_date = trade_start_date
        self.data_start_period = data_start_period
        self.end_date = end_date
        self.model_types = model_types
        self.feature_list_str = feature_list_str
        self.model_saver_loader = model_saver_loader
        self.model_created = False  # モデルが作成済みかどうかのフラグ

        self.data_managers = data_managers

    def process_symbol(self, symbol):
        print(f"Symbol of current data: {symbol}")

        RawDataPipeline(
            self.data_managers["formated_raw"],
            fetcher=YahooFinanceStockDataFetcher(
                symbol, self.data_start_period, self.end_date
            ),
        ).run(symbol)

        PreprocessPipeline(
            self.data_managers["formated_raw"], self.data_managers["processed_raw"]
        ).run(symbol)

        LabelCreatePipeline(
            self.data_managers["formated_raw"],
            self.data_managers["labeled"],
            label_creator=TroughLabelCreator(self.trade_start_date),
        ).run(symbol)

        FeaturePipeline(
            self.data_managers["processed_raw"],
            self.data_managers["normalized_feature"],
            self.trade_start_date,
            analyzers=AnalyzerFactory.create_analyzers(self.feature_list_str),
        ).run(symbol)

        SelectorPipeline(
            self.data_managers["labeled"],
            self.data_managers["normalized_feature"],
            self.data_managers["selected_feature"],
            selectors=SelectorFactory.create_selectors(),
        ).run(symbol)

        DataForModelPipeline(
            self.data_managers["labeled"],
            self.data_managers["selected_feature"],
            self.data_managers["training_and_test"],
            self.data_managers["practical"],
        ).run(symbol)

        try:
            if self.model_created:
                ModelRetrainPipeline(
                    self.data_managers["training_and_test"],
                    self.model_saver_loader,
                    self.model_types,
                ).run(symbol)
            else:
                ModelTrainPipeline(
                    self.data_managers["training_and_test"],
                    self.model_saver_loader,
                    self.model_types,
                ).run(symbol)
                self.model_created = True
        except Exception as e:
            print(f"{symbol} の処理中にエラーが発生しました: {e}")

        ModelPredictPipeline(
            self.model_saver_loader,
            self.data_managers["training_and_test"],
            self.data_managers["practical"],
            self.data_managers["predictions"],
            self.model_types,
        ).run(symbol)
