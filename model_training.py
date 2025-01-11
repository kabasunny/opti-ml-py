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
from models.ModelFactory import ModelFactory
from models.ModelTrainPipeline import ModelTrainPipeline
from models.ModelRetrainPipeline import ModelRetrainPipeline
from models.ModelPredictPipeline import ModelPredictPipeline


# 再トレーニング用のmain関数を定義
def model_training(
    symbol,
    trade_start_date,
    data_start_period,
    end_date,
    model_types,
    feature_list_str,
    model_saver_loader,
    model_created,
):
    # ----------------------data----------------------
    base_data_path = "data/stock_data"
    file_ext = "parquet"

    r_d_m = DataManager(base_data_path, file_ext, "formated_raw", end_date)
    prsd_d_m = DataManager(base_data_path, file_ext, "processed_raw", end_date)
    l_d_m = DataManager(base_data_path, file_ext, "labeled", end_date)
    n_f_d_m = DataManager(base_data_path, file_ext, "normalized_feature", end_date)
    s_f_d_m = DataManager(base_data_path, file_ext, "selected_feature", end_date)
    tr_tt_d_m = DataManager(base_data_path, file_ext, "training_and_test", end_date)
    prct_d_m = DataManager(base_data_path, file_ext, "practical", end_date)
    pred_d_m = DataManager(base_data_path, file_ext, "predictions", end_date)

    # ----------------------pipeline----------------------
    # print("★ DataPipeline ★")
    fetcher = YahooFinanceStockDataFetcher(symbol, data_start_period, end_date)
    RawDataPipeline(fetcher, r_d_m).run(symbol)
    # print("★ PreprocessPipeline ★")
    PreprocessPipeline(r_d_m, prsd_d_m).run(symbol)
    # print("★ LabelCreationPipeline ★")
    label_creator = TroughLabelCreator(trade_start_date)
    LabelCreatePipeline(r_d_m, l_d_m, label_creator).run(symbol)
    # print("★ FeatureCreationPipeline ★")
    analyzers = AnalyzerFactory.create_analyzers(feature_list_str)
    FeaturePipeline(
        prsd_d_m,
        n_f_d_m,
        analyzers,
        trade_start_date,
    ).run(symbol)
    # print("★ FeatureSelectionPipeline ★")
    selectors = SelectorFactory.create_selectors()
    SelectorPipeline(l_d_m, n_f_d_m, s_f_d_m, selectors).run(symbol)
    # print("★ DataForModelPipeline ★")
    DataForModelPipeline(
        l_d_m,
        s_f_d_m,
        tr_tt_d_m,
        prct_d_m,
    ).run(symbol)

    print(f"Symbol of current data: {symbol}")

    try:
        print(model_created)
        if model_created:
            print("★ ModelRetrainPipeline ★")
            ModelRetrainPipeline(tr_tt_d_m, model_saver_loader, model_types).run(symbol)
        else:
            print("★ ModelTrainPipeline ★")
            models = ModelFactory.create_models(model_types)
            print(models)
            ModelTrainPipeline(tr_tt_d_m, model_saver_loader, models).run(symbol)
            model_created = True
    except Exception as e:
        print(f"{symbol} の処理中にエラーが発生しました: {e}")
        # エラーが発生した場合スキップ

    # print("★ ModelPredictPipeline ★")
    ModelPredictPipeline(
        model_saver_loader,
        tr_tt_d_m,
        prct_d_m,
        pred_d_m,
        model_types,
    ).run(symbol)

    return model_created


if __name__ == "__main__":
    model_training()
