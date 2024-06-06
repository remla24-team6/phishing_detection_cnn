import pytest
import json
from src import train
import numpy as np 

from ml_lib_remla.preprocessing import Preprocessing
from tests.fixtures.fixtures import dataset_raw_test, trained_model

N_DATAPOINTS = 1000

def read_metrics(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def test_data_slice(dataset_raw_test, trained_model):
    
    print("Load data...")
    X_orig, y_test = dataset_raw_test
    X_orig = np.array(X_orig[:N_DATAPOINTS])
    y_test = np.array(y_test[:N_DATAPOINTS])
    
    X_slice_indices_com = [index for index, url in enumerate(X_orig) if url.find('.com')]
    X_slice_indices_uk = [index for index, url in enumerate(X_orig) if url.find('.uk')]
    X_slice_com = X_orig[X_slice_indices_com]
    y_pred_com = y_test[X_slice_indices_com]
    X_slice_uk = X_orig[X_slice_indices_uk]
    y_pred_uk = y_test[X_slice_indices_uk]

    print("Create slices...")
    
    slices = [(X_slice_com, y_pred_com), (X_slice_uk, y_pred_uk)]
    
    preprocessor = Preprocessing()
    
    print("Test slices...")
    for X, y in slices:

        X_tokenized = preprocessor.tokenize_batch(X)
        predictions = trained_model.predict(X_tokenized).flatten()
        
        correct = np.array(predictions) == y
        
        assert len(correct) > 0
        
        
if __name__ == "__main__":
    pytest.main()
