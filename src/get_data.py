"""
    Script to download dataset.
"""

from kaggle.api.kaggle_api_extended import KaggleApi
import os

DATA_FOLDER = "data"
# This url refers to the guy that initially uploaded the dataset.
DATA_URL = "aravindhannamalai/dl-dataset"


def get_data():
    """
    Downloads the raw data from Kaggle and stores it in the data folder.
    Note: Using the Kaggle API requires authentication using an API token. Please refer
    here for further information - https://www.kaggle.com/docs/api?utm_me=
        Returns:
            None
    """

    api = KaggleApi()
    api.authenticate()
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
    dataset = api.dataset_download_files(DATA_URL, path=DATA_FOLDER, unzip=True)
    print(dataset)
    return dataset

if __name__ == "__main__":
    get_data()
