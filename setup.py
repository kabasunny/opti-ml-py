# opti-ml-py\setup.py
import os


def create_directories_and_files(base_path):
    directories = [
        "data/raw",
        "data/processed",
        "preprocessing",
        "labeling",
        "features",
        "models",
        "utils",
        "tests",
        "configs",
        "notebooks",
    ]

    files = {
        "data/__init__.py": "",
        "data/StockDataFetcherBase.py": "",
        "data/YahooFinanceStockDataFetcher.py": "",
        "data/AlphaVantageStockDataFetcher.py": "",
        "preprocessing/__init__.py": "",
        "preprocessing/MissingValueHandler.py": "",
        "preprocessing/OutlierDetector.py": "",
        "preprocessing/Normalizer.py": "",
        "preprocessing/DataIntegrator.py": "",
        "labeling/__init__.py": "",
        "labeling/LabelCreator.py": "",
        "features/__init__.py": "",
        "features/FeatureCreator.py": "",
        "features/PeakTroughAnalyzer.py": "",
        "features/FourierAnalyzer.py": "",
        "features/FeatureSelector.py": "",
        "models/__init__.py": "",
        "models/Trainer.py": "",
        "models/Evaluator.py": "",
        "models/InferenceEngine.py": "",
        "models/EnsembleEvaluator.py": "",
        "models/BaseModel.py": "",
        "utils/__init__.py": "",
        "utils/Logger.py": "",
        "utils/HelperFunctions.py": "",
        "tests/test_StockDataFetcher.py": "",
        "tests/test_Preprocessing.py": "",
        "tests/test_FeatureEngineering.py": "",
        "tests/test_ModelTraining.py": "",
        "tests/test_Inference.py": "",
        "configs/config.yaml": "",
        "configs/model_params.yaml": "",
        "configs/feature_params.yaml": "",
        "notebooks/example_notebook.ipynb": "",
        "main.py": "",
        "README.md": "",
    }

    # Create directories
    for directory in directories:
        path = os.path.join(base_path, directory)
        os.makedirs(path, exist_ok=True)
        print(f"Directory created: {path}")

    # Create files
    for filepath, content in files.items():
        path = os.path.join(base_path, filepath)
        with open(path, "w") as f:
            f.write(content)
        print(f"File created: {path}")


# Usage
if __name__ == "__main__":
    create_directories_and_files(".")
