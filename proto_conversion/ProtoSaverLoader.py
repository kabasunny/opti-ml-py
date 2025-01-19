
from proto_conversion.ml_stock_service_pb2 import MLStockResponse

from proto_conversion.print_ml_stock_response import print_ml_stock_response_summary

class ProtoSaverLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def save_proto_response_to_file(self, proto_response):
        # Save the proto response to a binary file
        with open(self.file_path, "wb") as f:
            f.write(proto_response.SerializeToString())
        # print_ml_stock_response_summary(proto_response)

        return self.file_path

    def load_proto_response_from_file(self):
        if self.file_path is None:
            raise ValueError("file_path is not set")

        # Load the proto response from a binary file
        with open(self.file_path, "rb") as f:
            proto_response = MLStockResponse()
            proto_response.ParseFromString(f.read())
        return proto_response
