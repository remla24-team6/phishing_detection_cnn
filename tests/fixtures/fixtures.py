import pytest
from tensorflow.keras.models import load_model
from src.common.utils import load_from_pickle_file
from src.features.build_features import load_dataset

MODEL_PATH = "./model/model.keras"


@pytest.fixture()
def trained_model():
    model = load_model(MODEL_PATH)
    yield model

@pytest.fixture()
def dataset_raw_train():
    X_train, y_train = load_dataset("data/DL Dataset/train.txt")
    return [X_train, y_train]

@pytest.fixture()
def dataset_raw_test():
    X_test, y_test = load_dataset("data/DL Dataset/test.txt")
    return [X_test, y_test]
    

@pytest.fixture()
def dataset_tokenized_train():
    X_train, y_train = load_from_pickle_file("data/tokenized/train.pkl")
    return (X_train, y_train)

@pytest.fixture()
def dataset_tokenized_test():
    X_test, y_test = load_from_pickle_file("data/tokenized/test.pkl")
    return (X_test, y_test)
    