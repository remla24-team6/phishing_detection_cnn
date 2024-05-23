"""
    Provides utilities to preprocess the dataset.
"""

import os
from typing import Tuple, List,  Dict, Any
import yaml
import pickle
from ml_lib_remla.preprocessing import Preprocessing

OUTPUT_PATH = os.path.join("data", "tokenized")

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
    except FileNotFoundError as file_not_found_error:
        raise FileNotFoundError(f"Could not find file {data_path}.") from file_not_found_error
    except OSError as exception:
        raise OSError(f"An error occurred accessing file {data_path}: {exception}") from exception

    raw_x = [line.split("\t")[1] for line in loaded_data]
    raw_y = [line.split("\t")[0] for line in loaded_data]

    return raw_x, raw_y

def preprocess():
    """Reads in the raw data for all the splits, preprocesses the raw data (tokenising the sentences and encoding the labels)
    and stores the preprocessed data.
    """

    with open('training_params.yaml', 'r') as file:
        params = yaml.safe_load(file)

    preprocessor = Preprocessing()
    
    # Define the file names in lists
    data_files = ["train.txt", "test.txt", "val.txt"]
    output_files = ["train.pkl", "test.pkl", "val.pkl"]

    # Loop over the datasets for processing
    for data_file, output_file in zip(data_files, output_files):
        raw_x, raw_y = load_dataset(data_path=os.path.join(params["dataset_dir"], data_file))
        x = preprocessor.tokenize_batch(raw_x)
        y = preprocessor.encode_label_batch(raw_y)

        # Save processed data to respective output files
        with open(os.path.join(OUTPUT_PATH, output_file), 'wb') as file:
            pickle.dump((x, y), file)
        
    with open(os.path.join(OUTPUT_PATH, "char_index.pkl"), 'wb') as file:
        pickle.dump(preprocessor.tokenizer.word_index, file)

if __name__ == "__main__":
    preprocess()
