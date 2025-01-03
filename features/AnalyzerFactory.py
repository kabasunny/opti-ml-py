# AnalyzerFactory.py
from features.PeakTroughAnalyzer import PeakTroughAnalyzer
from features.FourierAnalyzer import FourierAnalyzer
from features.VolumeFeatureCreator import VolumeFeatureCreator
from features.PriceFeatureCreator import PriceFeatureCreator


class AnalyzerFactory:
    @staticmethod
    def create_analyzers(feature_list_str):
        analyzer_mapping = {
            "peak_trough": PeakTroughAnalyzer,
            "fourier": FourierAnalyzer,
            "volume": VolumeFeatureCreator,
            "price": PriceFeatureCreator,
        }
        return [
            analyzer_mapping[feature]()
            for feature in feature_list_str
            if feature in analyzer_mapping
        ]