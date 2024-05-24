import pytest
import json
import numpy as np

from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
from src.common import memory as mem
from src.train import train


MAX_MEMORY_USAGE = 250
N_ITERATIONS = 1
MODEL_PATH = "./model/model.keras"
X_TEST_PATH = "data/tokenized/test.pkl"

SKIP_MEMORY_TEST = True

@pytest.fixture()
def trained_model():
    model = load_model(MODEL_PATH)
    yield model
    
    
def test_nondeterminism_robustness(trained_model):
    original_score = accuracy_score(trained_model) # score between 0..100
    
    for seed in [1,2]:
        train.train(random_state=seed)
        model_variant = load_model(MODEL_PATH)
        assert abs(original_score - accuracy_score(model_variant)) <= 0.05
    

if __name__ == "__main__":
    pytest.main() 
    pass