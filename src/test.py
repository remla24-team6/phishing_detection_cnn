"""
    Tests the model on test data.
"""

import numpy as np
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, recall_score, f1_score, precision_score
from utils import load_from_pickle_file, load_from_json_file, save_to_json_file



def test():
    """Performs the testing of the trained model.
    """

    x_test, y_test = load_from_pickle_file("data/tokenized/test.pkl")
    model = load_model("model/model.keras")

    y_pred = model.predict(x_test[:10000], batch_size=1000)

    y_pred_binary = (np.array(y_pred) > 0.5).astype(int)
    y_test = y_test[:10000].reshape(-1, 1)

    metrics = load_from_json_file("reports/metrics.json")

    test_accuracy= accuracy_score(y_test, y_pred_binary)
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
    
    save_to_json_file(metrics, "reports/metrics.json")


if __name__ == "__main__":
    test()
