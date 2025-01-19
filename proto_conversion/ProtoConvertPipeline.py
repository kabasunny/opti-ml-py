from proto_conversion.ml_stock_service_pb2 import (
    MLStockResponse,
    MLSymbolData,
    MLDailyData,
    ModelPredictions,
)

class ProtoConvertPipeline:
    def __init__(self, raw_data_manager, real_pred_data_manager, proto_saver_loader, model_types):
        self.raw_data_manager = raw_data_manager
        self.real_pred_data_manager = real_pred_data_manager
        self.proto_saver_loader = proto_saver_loader
        self.model_types = model_types

    def run(self, symbols): # リストを受けるため他のパイプラインと異なる
        responses = []
        for symbol in symbols:
            # raw_data_manager からデータを読み込む
            raw_data_df = self.raw_data_manager.load_data(symbol)
            daily_data_list = [
                MLDailyData(
                    date=str(row["date"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=int(row["volume"]),
                )
                for _, row in raw_data_df.iterrows()
            ]

            # predictions_data_manager からデータを読み込む
            predictions_df = self.real_pred_data_manager.load_data(symbol)
            signal_dates = predictions_df[predictions_df["label"] == 1]["date"].tolist()

            model_predictions = {}
            for model in self.model_types:
                prediction_dates = predictions_df[predictions_df[model] == 1][
                    "date"
                ].tolist()
                model_predictions[model] = ModelPredictions(
                    prediction_dates=[str(date) for date in prediction_dates]
                )

            symbol_data = MLSymbolData(
                symbol=symbol,
                daily_data=daily_data_list,
                signals=[str(signal) for signal in signal_dates],
                model_predictions=model_predictions,
            )

            responses.append(symbol_data)  # MLSymbolData を直接追加

        # MLSymbolData のリストを含む結合レスポンスを作成
        combined_response = MLStockResponse(symbol_data=responses)

        # 保存処理を実行
        self.proto_saver_loader.save_proto_response_to_file(combined_response)
