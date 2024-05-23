import pytest
import json
from src import build_features, train


def read_metrics(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def test_reproducibility():
    build_features.preprocess()
    
    # First training run
    train.train()

    # Read metrics from the first run
    file_path = 'reports/metrics.json'
    metrics_1 = read_metrics(file_path)

    train_accuracy_1 = metrics_1['train_accuracy']
    train_loss_1 = metrics_1['train_loss']
    val_accuracy_1 = metrics_1['val_accuracy']
    val_loss_1 = metrics_1['val_loss']
    test_accuracy_1 = metrics_1['test_accuracy']
    avg_precision_1 = metrics_1['avg_precision']
    avg_recall_1 = metrics_1['avg_recall']
    avg_f1_1 = metrics_1['avg_f1']
    roc_auc_1 = metrics_1['roc_auc']

    # Second training run
    train.train()

    # Read metrics from the second run
    metrics_2 = read_metrics(file_path)

    train_accuracy_2 = metrics_2['train_accuracy']
    train_loss_2 = metrics_2['train_loss']
    val_accuracy_2 = metrics_2['val_accuracy']
    val_loss_2 = metrics_2['val_loss']
    test_accuracy_2 = metrics_2['test_accuracy']
    avg_precision_2 = metrics_2['avg_precision']
    avg_recall_2 = metrics_2['avg_recall']
    avg_f1_2 = metrics_2['avg_f1']
    roc_auc_2 = metrics_2['roc_auc']

    # Assert that metrics are the same
    assert train_accuracy_1 == train_accuracy_2, f"Train accuracies differ: {train_accuracy_1} != {train_accuracy_2}"
    assert train_loss_1 == train_loss_2, f"Train losses differ: {train_loss_1} != {train_loss_2}"
    assert val_accuracy_1 == val_accuracy_2, f"Validation accuracies differ: {val_accuracy_1} != {val_accuracy_2}"
    assert val_loss_1 == val_loss_2, f"Validation losses differ: {val_loss_1} != {val_loss_2}"
    assert test_accuracy_1 == test_accuracy_2, f"Test accuracies differ: {test_accuracy_1} != {test_accuracy_2}"
    assert avg_precision_1 == avg_precision_2, f"Avg precisions differ: {avg_precision_1} != {avg_precision_2}"
    assert avg_recall_1 == avg_recall_2, f"Avg recalls differ: {avg_recall_1} != {avg_recall_2}"
    assert avg_f1_1 == avg_f1_2, f"Avg F1 scores differ: {avg_f1_1} != {avg_f1_2}"
    assert roc_auc_1 == roc_auc_2, f"ROC AUCs differ: {roc_auc_1} != {roc_auc_2}"

if __name__ == "__main__":
    pytest.main()
