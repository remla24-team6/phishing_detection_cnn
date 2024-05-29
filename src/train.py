"""
    Method to train the model.
"""
from typing import Optional
import os
import json
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.models import load_model
from models.model import build_cnn_model
from common.utils import load_from_pickle_file, load_training_params, save_to_json_file


MODEL_SAVE_PATH = "model"
if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

DEFAULT_DIRECTORY = "model/"
DEFAULT_FILENAME = "model.keras"

RESTORE_CHECKPOINT = True

CHECKPOINT_DIR = 'checkpoints'
EPOCH_FILE_DIR = os.path.join(CHECKPOINT_DIR, 'epoch.json')
class EpochSaver(Callback):  # pylint: disable=too-few-public-methods
    # Disabled since tensorflow requires a class as a callback
    """
        Keras callback to save the current epoch number to a file at the end of each epoch.
        This can be useful for resuming training from the last saved epoch in case of interruptions.
        Attributes: filepath (str): The file path where the epoch number will be saved.
    """
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath

    def on_epoch_end(self, epoch):
        """
        Initialize the EpochSaver with the specified file path.

        Args:
            filepath (str): The file path where the epoch number will be saved.
        """
        with open(self.filepath, 'w') as file:
            json.dump({'epoch': epoch + 1}, file)  # Save the next epoch to run


checkpoint_path = os.path.join(CHECKPOINT_DIR, 'model_checkpoint.keras')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        save_best_only=True,
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        verbose=1
    )


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

    # Restore model from checkpoint if available
    initial_epoch = 0
    if RESTORE_CHECKPOINT and os.path.exists(checkpoint_path):
        print("Loading model from checkpoint.")
        model = load_model(checkpoint_path)
        if os.path.exists(EPOCH_FILE_DIR):
            with open(EPOCH_FILE_DIR, 'r') as file:
                data = json.load(file)
                initial_epoch = data.get('epoch', 0)
    else:
        print("No checkpoint found. Training new model.")

    x_train, y_train = load_from_pickle_file(pickle_path="data/tokenized/train.pkl")
    x_val, y_val = load_from_pickle_file(pickle_path="data/tokenized/val.pkl")
    if not num_features:
        num_features = x_train.shape[0]
    try:
        hist = model.fit(
            x_train[:num_features],
            y_train[:num_features],
            batch_size=params["batch_train"],
            initial_epoch=initial_epoch,
            epochs=params["epoch"],
            shuffle=True,
            validation_data=(x_val[:num_features], y_val[:num_features]),
            callbacks=[checkpoint_callback, EpochSaver(EPOCH_FILE_DIR)]
        )
        # If Training is a success, Delete all checkpoints.
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            if os.path.exists(EPOCH_FILE_DIR):
                os.remove(EPOCH_FILE_DIR)
            print("Checkpoints deleted after successful training.")

        if not os.path.exists(DEFAULT_DIRECTORY):
            os.makedirs(DEFAULT_DIRECTORY)

        print("Saving Model...")
        model.save(DEFAULT_DIRECTORY + DEFAULT_FILENAME)
        print("Model Saved")

        metrics = {
            "train_accuracy": hist.history['accuracy'][0],
            "train_loss": hist.history['loss'][0],
            "val_accuracy": hist.history['val_accuracy'][0],
            "val_loss": hist.history['val_loss'][0],
        }

        save_to_json_file(metrics, "model/metrics.json")

    except Exception as error:  # pylint: disable=broad-except
        # Disable to cover any unexpected failure
        print(f"Training interrupted: {error}, Checkpoint saved.")


if __name__ == "__main__":
    train()
