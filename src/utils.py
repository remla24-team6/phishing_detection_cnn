import yaml
import pickle
from typing import Dict, Any

def load_training_params() -> Dict[str, Any]:
    """Loads and returns the training parameters.

    Raises:
        FileNotFoundError: Raising FileNotFound if training_params.yaml is not found in the root folder of the project.

    Returns:
        Dict[str, Any]: Returns the training params as a dictionary.
    """
    try:
        with open('training_params.yaml', 'r') as file:
            params = yaml.safe_load(file)
    except:
        raise FileNotFoundError("Could not find file training_params.yaml.")

    return params

def load_from_pickle_file(pickle_path: str) -> Any:
    """Loads a pickle file into memory.

    Args:
        pickle_path (str): Path to pickle file
    
    Returns:
        Any: Returns the loaded object.
    """
    try:
        with open(pickle_path, 'rb') as f:
            loaded_file = pickle.load(f)
    except:
        raise FileNotFoundError(f"Could not find file {pickle_path}.")

    return loaded_file


def save_to_pickle_file(obj: Any, pickle_path: str) -> Any:
    """Saves the given object into a pickle file defined by the pickle_path

    Args:
        obj: (Any): The object that needs to be dumped.
        pickle_path (str): Path to pickle file where object will be stored.
    
    Returns:
        Any: Returns the loaded object.
    """
    
    with open(pickle_path, 'wb') as f:
        pickle.dump(obj, f)
