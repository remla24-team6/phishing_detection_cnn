"""
    Method to train the model.
"""
from typing import Optional
import os
import json
from models.model import build_cnn_model
from common.utils import load_from_pickle_file, load_training_params, save_to_json_file

from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.models import load_model


MODEL_SAVE_PATH = "model"
if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

DEFAULT_DIRECTORY = "model/"
DEFAULT_FILENAME = "model.keras"

RESTORE_CHECKPOINT = True

checkpoint_dir = 'checkpoints'
epoch_file_path = os.path.join(checkpoint_dir, 'epoch.json')
class EpochSaver(Callback):
    def __init__(self, filepath):
        super(EpochSaver, self).__init__()
        self.filepath = filepath

    def on_epoch_end(self, epoch, logs=None):
        with open(self.filepath, 'w') as f:
            json.dump({'epoch': epoch + 1}, f)  # Save the next epoch to run


"""Define checkpoint save location
"""
checkpoint_path = os.path.join(checkpoint_dir, 'model_checkpoint.keras')
os.makedirs(checkpoint_dir, exist_ok=True)
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
        if os.path.exists(epoch_file_path):
            with open(epoch_file_path, 'r') as f:
                data = json.load(f)
                initial_epoch = data.get('epoch', 0)
    else:
        print("No checkpoint found. Training new model.")

    x_train, y_train = load_from_pickle_file(pickle_path="data/tokenized/train.pkl")
    x_val, y_val = load_from_pickle_file(pickle_path="data/tokenized/val.pkl")

    try:
        hist = model.fit(
            x_train[:num_features],
            y_train[:num_features],
            batch_size=params["batch_train"],
            initial_epoch=initial_epoch,
            epochs=params["epoch"],
            shuffle=True,
            validation_data=(x_val[:10000], y_val[:10000]),
            callbacks=[checkpoint_callback, EpochSaver(epoch_file_path)]
        )
        
        # If Training is a success, Delete all checkpoints.
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            if os.path.exists(epoch_file_path):
                os.remove(epoch_file_path)
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

    except Exception as e:
        print(f"Training interrupted: {e}, Checkpoint saved.")



if __name__ == "__main__":
    train()
