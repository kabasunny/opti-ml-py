# opti-ml-py\models\Evaluator.py
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# from decorators.ArgsChecker import ArgsChecker
# from models.BaseModelABC import BaseModelABC


class Evaluator:
    @staticmethod
    # @ArgsChecker((BaseModelABC), (float, float, float, float))
    def evaluate_model(model, X_test, y_test):

        y_pred = model.predict(X_test)
        if hasattr(y_pred, "shape") and len(y_pred.shape) == 2 and y_pred.shape[1] > 1:
            y_pred = y_pred.argmax(axis=1)
        else:
            y_pred = (y_pred > 0.5).astype(int)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # model_name = type(model).__name__.replace(
        #     "Classifier", ""
        # )  # 'Classifier' を削除
        # print(f"[{model_name}] トレーニング修了時のテスト結果")
        # print(f"Accuracy: {accuracy:.4f}")
        # print(f"Precision: {precision:.4f}")
        # print(f"Recall: {recall:.4f}")
        # print(f"F1 Score: {f1:.4f}")

        return accuracy, precision, recall, f1
