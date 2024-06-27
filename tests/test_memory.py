import pytest
import json
import numpy as np

from src.features import build_features
from src import train
from src.common import memory as mem
from src.common.utils import load_training_params
from functools import partial

MAX_MEMORY_USAGE = 250
N_ITERATIONS = 1

SKIP_MEMORY_TEST = False

@pytest.mark.skipif(SKIP_MEMORY_TEST == True, reason="Takes 3 minutes to run.")
def test_memory():
    build_features.preprocess()

    train_fn = partial(train.train, num_features=10)

    memory_usages = [mem.train_with_memory(train_fn) for _ in range(N_ITERATIONS)]
    print(memory_usages)
    max_memory_usage = max(memory_usages)/1024.0/1024.0
    assert max_memory_usage < MAX_MEMORY_USAGE, 'Maximum memory usage < 250 MByte.'

if __name__ == "__main__":
    pytest.main() 
