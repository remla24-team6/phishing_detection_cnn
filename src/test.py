"""
    Tests the model on test data.
"""

import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score, precision_score
from common.utils import load_from_pickle_file, load_from_json_file, save_to_json_file
from common.timer import predict_with_time


N_SIZE_TESTDATA = 10000

def test():
    """Performs the testing of the trained model.
    """

    X_test, y_test = load_from_pickle_file("data/tokenized/test.pkl")
    model = load_model("model/model.keras")

    X_test = X_test[:N_SIZE_TESTDATA]
    y_pred, time = predict_with_time(model, X_test)

    y_pred_binary = (np.array(y_pred) > 0.5).astype(int)
    y_test = y_test[:N_SIZE_TESTDATA].reshape(-1, 1)

    metrics = load_from_json_file("reports/metrics.json")

    test_accuracy = accuracy_score(y_test, y_pred_binary)
    print(f"Test Accuracy: {test_accuracy}")
    metrics["test_accuracy"] = test_accuracy
    avg_precision = precision_score(y_test, y_pred_binary)
    print(f"Test avg_precision: {avg_precision}")
    metrics["avg_precision"] = avg_precision
    avg_recall = recall_score(y_test, y_pred_binary)
    print(f"Test avg_recall: {avg_recall}")
    metrics["avg_recall"] = avg_recall
    avg_f1 = f1_score(y_test, y_pred_binary)
    print(f"Test avg_f1: {avg_f1}")
    metrics["avg_f1"] = avg_f1
    roc_auc = roc_auc_score(y_test, y_pred_binary)
    print(f"Test roc_auc: {roc_auc}")
    metrics["roc_auc"] = roc_auc
    metrics["test size"] = len(X_test)
    metrics["inference time"] = time


    save_to_json_file(metrics, "reports/metrics.json")


if __name__ == "__main__":
    test()
