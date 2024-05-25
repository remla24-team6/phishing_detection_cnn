"""
    Method to train the model.
"""
from typing import Optional
import os
from models.model import build_cnn_model
from common.utils import load_from_pickle_file, load_training_params, save_to_json_file
from typing import Optional

MODEL_SAVE_PATH = "model"
if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

DEFAULT_DIRECTORY = "model/"
DEFAULT_FILENAME = "model.keras"


def train(num_features: Optional[int] = 0):
    """Loads the precrocessed data and performs the model training.
    Args:
        num_features (int): Number of training features to train the model on.
    """
    params = load_training_params()
    model = build_cnn_model(params=params)
    model.compile(
        loss=params["loss_function"],
        optimizer=params["optimizer"],
        metrics=["accuracy"],
    )

    x_train, y_train = load_from_pickle_file(pickle_path="data/tokenized/train.pkl")
    x_val, y_val = load_from_pickle_file(pickle_path="data/tokenized/val.pkl")

    if not num_features:
        num_features = x_train.shape[0]

    hist = model.fit(
        x_train[:num_features],
        y_train[:num_features],
        batch_size=params["batch_train"],
        epochs=params["epoch"],
        shuffle=True,
        validation_data=(x_val, y_val),
    )

    if not os.path.exists(DEFAULT_DIRECTORY):
        os.makedirs(DEFAULT_DIRECTORY)

    model.save(DEFAULT_DIRECTORY + DEFAULT_FILENAME)

    metrics = {
        "train_accuracy": hist.history["accuracy"][0],
        "train_loss": hist.history["loss"][0],
        "val_accuracy": hist.history["val_accuracy"][0],
        "val_loss": hist.history["val_loss"][0],
    }

    save_to_json_file(metrics, "model/metrics.json")
    model.save("model/model.keras")


if __name__ == "__main__":
    train()
