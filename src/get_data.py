import os
from kaggle.api.kaggle_api_extended import KaggleApi


def get_data():
    api = KaggleApi()
    api.authenticate()
    # This url refers to the guy that initially uploaded the dataset.
    api.dataset_download_files('aravindhannamalai/dl-dataset', path='data', unzip=True)


if __name__ == "__main__":
    get_data()