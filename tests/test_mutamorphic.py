import pytest
import json
import numpy as np
from urllib.parse import urlparse, urlunparse
import re
from ml_lib_remla.preprocessing import Preprocessing

from tensorflow.keras.models import load_model

from tests.fixtures.fixtures import dataset_raw_test

import warnings
warnings.filterwarnings('ignore')

N_SAMPLES = 100
TLD_LIST = ['.io', '.ai', '.dev']
THRESHOLD = 0.5

def parse_urls(urls):
    return [urlparse(url) for url in urls]

def replace_scheme(scheme):
    return 'https' if scheme == 'http' else 'http' if scheme == 'https' else scheme

def replace_tld(netloc, tld):
    tld_pattern = re.compile(r'\.(com|org|de|net|uk|us|mobi|co\.uk|gov|edu|io|ai|dev|biz|info|mil|int|arpa)\b', re.IGNORECASE)
    return tld_pattern.sub(lambda _: '.' + tld.lstrip('.'), netloc)

def generate_mutants(parsed_urls):
    mutants = []
    for url in parsed_urls:
        for tld in TLD_LIST:
            scheme_replaced = url._replace(scheme=replace_scheme(url.scheme))
            tld_replaced = scheme_replaced._replace(netloc=replace_tld(scheme_replaced.netloc, tld))
            mutants.append(urlunparse(tld_replaced))
    return np.array([url for url in mutants if url])

def check_failing_tests(y_pred_original, y_pred_mutant):
    n_mutants = len(y_pred_mutant) // len(y_pred_original)
    y_pred_original = y_pred_original.flatten()
    y_pred_mutant = y_pred_mutant.reshape(len(y_pred_original), n_mutants).T

    labels_original = (y_pred_original > THRESHOLD).astype(int)
    labels_mutant = (y_pred_mutant > THRESHOLD).astype(int)

    failing_tests = np.argwhere(labels_original != labels_mutant)
    return failing_tests

def test_mutamorphic(dataset_raw_test):
    X_orig, y_orig = dataset_raw_test

    X_orig = X_orig[:N_SAMPLES]
    y_orig = y_orig[:N_SAMPLES]
    parsed_urls = parse_urls(X_orig)
    mutant_candidates = generate_mutants(parsed_urls)

    preprocessor = Preprocessing()
    X_orig_tokenized = preprocessor.tokenize_batch(X_orig)
    X_mutant_tokenized = preprocessor.tokenize_batch(mutant_candidates)

    model = load_model(MODEL_PATH)
    y_pred_original = model.predict(X_orig_tokenized)
    y_pred_mutant = model.predict(X_mutant_tokenized)

    failing_tests = check_failing_tests(y_pred_original, y_pred_mutant)
    assert len(failing_tests) < len(y_orig)


    

if __name__ == "__main__":
    pytest.main() 
