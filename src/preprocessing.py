"""
    Provides utilities to preprocess the dataset.
"""

import os
from typing import Tuple, List
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import save_to_pickle_file, load_training_params

MAX_SEQUENCE_LENGTH = 200
OOV_TOKEN = "-n-"
OUTPUT_PATH = os.path.join("output", "tokenized")

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)


def load_dataset(data_path: str) -> Tuple[List[str], List[str]]:
    """Loads the data split from the path. The path should be a .txt file that
    has been created from the get_data step. his should be stored in the data folder.

    Args:
        data_path (str): The path to the split .txt file.

    Returns:
        Tuple[List[str], List[str]]: Returns a tuple of raw_x and raw_y. raw_x is a
        list of strings for all the sentences in the split and raw_y is their corresponding label.
    """
    print(f"Loading dataset: {data_path}")

    try:
        with open(data_path, "r") as data_file:
            loaded_data = [line.strip() for line in data_file.readlines()[1:]]
    except IOError as err:
        print(f"Error: {err}")

    raw_x = [line.split("\t")[1] for line in loaded_data]
    raw_y = [line.split("\t")[0] for line in loaded_data]

    return raw_x, raw_y


def preprocess():
    """Reads in the raw data for all the splits, preprocesses the raw data (tokenising the sentences and encoding the labels)
    and stores the preprocessed data.
    """
    params = load_training_params()

    raw_x_train, raw_y_train = load_dataset(
        data_path=os.path.join(params["dataset_dir"], "train.txt")
    )
    raw_x_test, raw_y_test = load_dataset(
        data_path=os.path.join(params["dataset_dir"], "test.txt")
    )
    raw_x_val, raw_y_val = load_dataset(
        data_path=os.path.join(params["dataset_dir"], "val.txt")
    )

    tokenizer = Tokenizer(lower=True, char_level=True, oov_token=OOV_TOKEN)
    tokenizer.fit_on_texts(raw_x_train + raw_x_val + raw_x_test)
    char_index = tokenizer.word_index

    x_train = pad_sequences(
        tokenizer.texts_to_sequences(raw_x_train), maxlen=MAX_SEQUENCE_LENGTH
    )
    x_val = pad_sequences(
        tokenizer.texts_to_sequences(raw_x_val), maxlen=MAX_SEQUENCE_LENGTH
    )
    x_test = pad_sequences(
        tokenizer.texts_to_sequences(raw_x_test), maxlen=MAX_SEQUENCE_LENGTH
    )

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(raw_y_train)
    y_val = encoder.transform(raw_y_val)
    y_test = encoder.transform(raw_y_test)

    save_to_pickle_file(
        obj=char_index, pickle_path=os.path.join(OUTPUT_PATH, "char_index.pkl")
    )
    save_to_pickle_file(
        obj=(x_train, y_train), pickle_path=os.path.join(OUTPUT_PATH, "train.pkl")
    )
    save_to_pickle_file(
        obj=(x_val, y_val), pickle_path=os.path.join(OUTPUT_PATH, "val.pkl")
    )
    save_to_pickle_file(
        obj=(x_test, y_test), pickle_path=os.path.join(OUTPUT_PATH, "test.pkl")
    )


if __name__ == "__main__":
    preprocess()
