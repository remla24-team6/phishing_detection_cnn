import pytest
import json
import numpy as np

from tensorflow.keras.models import load_model
from src.features import build_features
from src import train
from src.common import timer as t
from src.common import utils

MAX_SERVING_LATENCY = 0.001
P_PERCENTILE = 0.99
N_LATENCIES = 1
MODEL_PATH = "./model/model.keras"
X_TEST_PATH = "data/tokenized/test.pkl"

    
SKIP_INFERENCE_TEST = False

@pytest.mark.skipif(SKIP_INFERENCE_TEST == True, reason="Takes 3 minutes to run.")
def test_inference():

    build_features.preprocess()

    # First training run
    train.train(num_features=10)
    
    model = load_model(MODEL_PATH)

    X_test, _ = utils.load_from_pickle_file(X_TEST_PATH)
    
    latencies = np.array([t.predict_with_time(model, X_test)[1] for _ in range(N_LATENCIES)])
    avg_latencies = latencies / len(X_test) 
    avg_latency_p99 = np.quantile(avg_latencies, P_PERCENTILE)
    
    assert avg_latency_p99 < MAX_SERVING_LATENCY, 'Serving latency at 99th percentile should be < 0.02 sec'

if __name__ == "__main__":
    pytest.main()
