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
    file_path = 'model/metrics.json'
    metrics_1 = read_metrics(file_path)

    train_accuracy_1 = metrics_1['train_accuracy']
    train_loss_1 = metrics_1['train_loss']
    val_accuracy_1 = metrics_1['val_accuracy']
    val_loss_1 = metrics_1['val_loss']

    # Second training run
    train.train()

    # Read metrics from the second run
    metrics_2 = read_metrics(file_path)

    train_accuracy_2 = metrics_2['train_accuracy']
    train_loss_2 = metrics_2['train_loss']
    val_accuracy_2 = metrics_2['val_accuracy']
    val_loss_2 = metrics_2['val_loss']

    # Assert that metrics are the same
    assert train_accuracy_1 == train_accuracy_2, f"Train accuracies differ: {train_accuracy_1} != {train_accuracy_2}"
    assert train_loss_1 == train_loss_2, f"Train losses differ: {train_loss_1} != {train_loss_2}"
    assert val_accuracy_1 == val_accuracy_2, f"Validation accuracies differ: {val_accuracy_1} != {val_accuracy_2}"
    assert val_loss_1 == val_loss_2, f"Validation losses differ: {val_loss_1} != {val_loss_2}"

if __name__ == "__main__":
    pytest.main()