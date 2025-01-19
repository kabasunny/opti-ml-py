from data.YahooFinanceStockDataFetcher import YahooFinanceStockDataFetcher
from data.RawDataPipeline import RawDataPipeline
from preprocessing.PreprocessPipeline import PreprocessPipeline
from labeling.LabelCreatePipeline import LabelCreatePipeline
from labeling.TroughLabelCreator import TroughLabelCreator
from features.FeaturePipeline import FeaturePipeline
from selectores.SelectorPipeline import SelectorPipeline
from proto_conversion.ProtoConvertPipeline import ProtoConvertPipeline
from features.AnalyzerFactory import AnalyzerFactory
from selectores.SelectorFactory import SelectorFactory
from for_real.ForRealPredictPipeline import ForRealPredictPipeline
import time
from proto_conversion.print_ml_stock_response import print_ml_stock_response_summary



class RealDataAutomatedPipeline:
    def __init__(
        self,
        before_period_days,  # 特徴量生成に必要な日数
        model_types,
        feature_list_str,
        model_saver_loader,
        data_managers,
        selectors,
        proto_saver_loader
    ):
        self.before_period_days = before_period_days
        self.model_types = model_types
        self.feature_list_str = feature_list_str
        self.model_saver_loader = model_saver_loader
        self.model_created = False 
        self.data_managers = data_managers
        self.selectors = selectors
        self.proto_saver_loader = proto_saver_loader

        # 各パイプラインをインスタンス変数として保持
        self.raw_data_pipeline = RawDataPipeline(
            self.data_managers["formated_raw"],
            fetcher=YahooFinanceStockDataFetcher(),
        )
        self.preprocess_pipeline = PreprocessPipeline(
            self.data_managers["formated_raw"], 
            self.data_managers["processed_raw"]
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
            self.data_managers["selected_ft_with_label"],
            SelectorFactory.create_selectors(self.selectors),
        )
        self.real_predict_pipeline = ForRealPredictPipeline(
            self.model_saver_loader,
            data_managers["selected_ft_with_label"],
            data_managers["real_predictions"],
            self.model_types,
        )

        # ProtoConvertPipelineの初期化と実行
        self.proto_convert_pipeline = ProtoConvertPipeline(
            self.data_managers["formated_raw"],
            self.data_managers["real_predictions"],
            self.proto_saver_loader, 
            self.model_types,
            )

    def process_symbol(self, symbol):
        print(f"Symbol of current data: {symbol}")

        try:
            pipelines = [
                ("RawDataPipeline", self.raw_data_pipeline),
                ("PreprocessPipeline", self.preprocess_pipeline),
                ("LabelCreatePipeline", self.label_create_pipeline),
                ("FeaturePipeline", self.feature_pipeline),
                ("SelectorPipeline", self.selector_pipeline),
                ("ForRealPredictPipeline", self.real_predict_pipeline),
            ]

            for pipeline_name, pipeline in pipelines:
                start_time = time.time()
                pipeline.run(symbol)
                elapsed_time = time.time() - start_time
                print(f"{pipeline_name} processing time: {elapsed_time:.4f} 秒")

        except Exception as e:
            print(f"{symbol} の処理中にエラーが発生しました: {e}")

    def finish_prosess(self, symbols):
        print(f"Proto file processing for all symbols")
        print(symbols)
        self.proto_convert_pipeline.run(symbols) # リストを受けるため他のパイプラインと異なる

        
        # # 保存したプロトコルバッファーの読み込み
        # loaded_proto_response = self.proto_saver_loader.load_proto_response_from_file()
        # print_ml_stock_response_summary(loaded_proto_response)