import pytest
import json
import numpy as np

from src import train
from src import test
from src.common import utils

MAX_MEMORY_USAGE = 250
N_ITERATIONS = 1
MODEL_PATH = "./model/model.keras"
X_TEST_PATH = "data/tokenized/test.pkl"

CLASSIFIER_THRESHOLD = 0.5
SKIP_MEMORY_TEST = True

ACCURACY_METRIC = 'val_accuracy'

    
def test_nondeterminism_robustness():
    original_metrics = utils.load_from_json_file("reports/metrics.json")
    original_score = original_metrics[ACCURACY_METRIC]
    
    for seed in [1,2]:
        train.train(random_seed=seed)
        test.test()
        
    metrics = utils.load_from_json_file("reports/metrics.json")
    assert abs(original_score - metrics[ACCURACY_METRIC]) <= 0.05
    

if __name__ == "__main__":
    pytest.main() 