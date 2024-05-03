"""
    Method to train the model.
"""
import os
from model import build_cnn_model
from utils import load_from_pickle_file, load_training_params


DEFAULT_DIRECTORY = "model/"
DEFAULT_FILENAME = "model.keras"


def train():
    """Loads the precrocessed data and performs the model training.
    """
    params = load_training_params()
    model = build_cnn_model(params=params)
    model.compile(
        loss=params["loss_function"],
        optimizer=params["optimizer"],
        metrics=["accuracy"],
    )

    x_train, y_train = load_from_pickle_file(pickle_path="output/tokenized/train.pkl")
    x_val, y_val = load_from_pickle_file(pickle_path="output/tokenized/val.pkl")

    _ = model.fit(
        x_train[:10000],
        y_train[:10000],
        batch_size=params["batch_train"],
        epochs=params["epoch"],
        shuffle=True,
        validation_data=(x_val[:10000], y_val[:10000]),
    )

    if not os.path.exists(DEFAULT_DIRECTORY):
        os.makedirs(DEFAULT_DIRECTORY)

    model.save(DEFAULT_DIRECTORY + DEFAULT_FILENAME)


if __name__ == "__main__":
    train()
