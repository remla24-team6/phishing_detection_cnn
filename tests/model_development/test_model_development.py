import pytest
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import load_model
from src.features import build_features
from src.train import train
from src.test import test
from src.common.utils import load_from_json_file


def set_seeds(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

@pytest.mark.parametrize("run", range(3))  # Test three runs for non-determinism
def test_model_non_determinism(run):
    print(f"Running non-determinism test (run {run})")
    set_seeds(
        42 + run
    )
    train(num_features=10)
    test()
    metrics = load_from_json_file("reports/metrics.json")
    accuracy = metrics["test_accuracy"]
    
    assert 0 <= accuracy <= 1.0


@pytest.mark.parametrize("run", range(3))
def test_model_determinism(run):
    print(f"Running determinism test (run {run})")
    
    set_seeds(42)

    train(num_features=10)

    test()

    metrics = load_from_json_file("reports/metrics.json")
    accuracy = metrics["test_accuracy"]

    if run == 0:
        test_model_determinism.baseline_accuracy = accuracy
    else:
        assert accuracy == pytest.approx(
            test_model_determinism.baseline_accuracy, rel=1e-3
        )


if __name__ == "__main__":
    build_features()
    test_model_determinism.baseline_accuracy = None
    pytest.main()
    pass
