"""
    Utilities to load/save data.
"""
import pickle
from typing import Dict, Any
import json
import yaml


def load_training_params(pickle_path='training_params.yaml') -> Dict[str, Any]:
    """Loads and returns the training parameters.

    Raises:
        FileNotFoundError: Raising FileNotFound if training_params.yaml is
        not found in the root folder of the project.

    Returns:
        Dict[str, Any]: Returns the training params as a dictionary
    """
    try:
        with open(pickle_path, 'r') as file:
            params = yaml.safe_load(file)
    except FileNotFoundError as file_not_found_error:
        raise FileNotFoundError(f"Could not find file {pickle_path}.") from file_not_found_error
    except OSError as exception:
        raise OSError(f"An error occurred accessing file {pickle_path}: {exception}") from exception

    return params


def load_from_pickle_file(pickle_path: str) -> Any:
    """Loads a pickle file into memory.

    Args:
        pickle_path (str): Path to pickle file

    Returns:
        Any: Returns the loaded object.
    """
    try:
        with open(pickle_path, 'rb') as file:
            loaded_file = pickle.load(file)
    except FileNotFoundError as file_not_found_error:
        raise FileNotFoundError(f"Could not find file {pickle_path}.") from file_not_found_error
    except OSError as exception:
        raise OSError(f"An error occurred accessing file {pickle_path}: {exception}") from exception

    return loaded_file


def save_to_pickle_file(obj: Any, pickle_path: str) -> Any:
    """Saves the given object into a pickle file defined by the pickle_path

    Args:
        obj: (Any): The object that needs to be dumped.
        pickle_path (str): Path to pickle file where object will be stored.

    Returns:
        Any: Returns the loaded object.
    """

    with open(pickle_path, 'wb') as file:
        pickle.dump(obj, file)


def save_to_json_file(data_dict: Dict[str, Any], save_path: str):
    """Dumps a dictionary into JSON

    Args:
        data_dict (Dict[str, Any]): The dictionary to save into a JSON file.
        save_path (str): The path to save the JSON to.
    """
    with open(save_path, "w") as outfile:
        json.dump(data_dict, outfile)


def load_from_json_file(data_path: str) -> Dict[str, Any]:
    """Loads data from a JSON file.

    Args:
        data_path (str): Path to load data from

    Returns:
        Dict[str, Any]: Returns the loaded dictionary.
    """

    with open(data_path, "r") as infile:
        loaded_dict = json.load(infile)

    return loaded_dict
