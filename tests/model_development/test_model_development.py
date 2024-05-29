import random
import pytest
import numpy as np
import tensorflow as tf
from src.features import build_features
from src.train import train
from src.test import test
from src.common.utils import load_from_json_file

SKIP_DETERMINISM_TEST = True
NUM_DETERMINISM_RUNS = 3


def set_seeds(seed=42):
    """Sets all the random seeds for numpy, torch and random."""
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


@pytest.mark.skipif(SKIP_DETERMINISM_TEST is True, reason="Takes 3 minutes to run.")
@pytest.mark.parametrize(
    "run", range(NUM_DETERMINISM_RUNS)
)  # Test three runs for non-determinism
def test_model_non_determinism(run):
    """Trains the model NUM_DETERMINISM_RUNS number of times with different random seeds and checks
    if the accuracy remains in a reasonable range."""
    print(f"Running non-determinism test (run {run})")
    set_seeds(42 + run)
    train(num_features=10)
    test()
    metrics = load_from_json_file("reports/metrics.json")
    accuracy = metrics["test_accuracy"]

    assert 0.7 <= accuracy <= 1.0


@pytest.mark.skipif(SKIP_DETERMINISM_TEST is True, reason="Takes 3 minutes to run.")
@pytest.mark.parametrize("run", range(NUM_DETERMINISM_RUNS))
def test_model_determinism(run):
    """Trains the model NUM_DETERMINISM_RUNS number of times with the same random seed and checks
    if the accuracy is roughly the same every single time."""
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
