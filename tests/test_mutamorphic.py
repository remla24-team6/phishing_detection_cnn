import pytest
import numpy as np
from urllib.parse import urlparse, urlunparse
import re
from ml_lib_remla.preprocessing import Preprocessing

from tests.fixtures.fixtures import dataset_raw_test, trained_model

import warnings
warnings.filterwarnings('ignore')

N_SAMPLES = 100
TLD_LIST = ['.io', '.ai', '.dev']
TLD_REPAIR_LIST = ['.uk', '.org', '.de']
THRESHOLD = 0.5

def parse_urls(urls):
    return [urlparse(url) for url in urls]

def replace_scheme(scheme):
    return 'https' if scheme == 'http' else 'http' if scheme == 'https' else scheme

def replace_tld(netloc, tld):
    tld_pattern = re.compile(r'\.(com|org|de|net|uk|us|mobi|co\.uk|gov|edu|io|ai|dev|biz|info|mil|int|arpa)\b', re.IGNORECASE)
    return tld_pattern.sub(lambda _: '.' + tld.lstrip('.'), netloc)

def input_generation(parsed_urls, tld_list=TLD_LIST):
    mutants = []
    for url in parsed_urls:
        for tld in tld_list:
            scheme_replaced = url._replace(scheme=replace_scheme(url.scheme))
            tld_replaced = scheme_replaced._replace(netloc=replace_tld(scheme_replaced.netloc, tld))
            mutants.append(urlunparse(tld_replaced))
    return np.array([url for url in mutants if url])

def oracle_generation(y_pred_original, y_pred_mutant, threshold=THRESHOLD):
    n_mutants = len(y_pred_mutant) // len(y_pred_original)
    y_pred_original = y_pred_original.flatten()
    y_pred_mutant = y_pred_mutant.reshape(len(y_pred_original), n_mutants).T

    labels_original = (np.array(y_pred_original) > threshold).astype(int)
    labels_mutants = (np.array(y_pred_mutant) > threshold).astype(int)
    
    labels_mutant = np.max(labels_mutants, axis=0)

    failing_tests = np.argwhere(labels_original != labels_mutant)
    return failing_tests

def automatic_repair(model,preprocessor, X_failing_mutants):
    
    mutant_candidates = input_generation(X_failing_mutants, tld_list=TLD_REPAIR_LIST)
    
    X_mutants = preprocessor.tokenize_batch(mutant_candidates)
    y_prob_mutants = model.predict(X_mutants)
    y_prob_mutants.reshape(len(X_failing_mutants), len(TLD_REPAIR_LIST))
    
    y_prob_mutants, labels_repaired = labels(y_prob_mutants)
    labels_repaired = np.max(labels_repaired, axis=0)
    
    return labels_repaired

def labels(y_pred):
    labels = (np.array(y_pred) > THRESHOLD).astype(int)
    return y_pred, labels

def test_mutamorphic(dataset_raw_test, trained_model):
    X_orig, y_orig = dataset_raw_test

    X_orig = X_orig[:N_SAMPLES]
    y_orig = y_orig[:N_SAMPLES]
    parsed_urls = parse_urls(X_orig)
    mutant_candidates = input_generation(parsed_urls)

    preprocessor = Preprocessing()
    X_orig_tokenized = preprocessor.tokenize_batch(X_orig)
    X_mutant_tokenized = preprocessor.tokenize_batch(mutant_candidates)

    y_prob_original, labels_original = labels(trained_model.predict(X_orig_tokenized))
    y_prob_mutants = trained_model.predict(X_mutant_tokenized)

    failing_tests = oracle_generation(y_prob_original, y_prob_mutants)

    labels_final = (np.array(y_prob_original) > THRESHOLD).astype(int)
    
    if len(failing_tests) > 0:
        X_failing_mutants = mutant_candidates.reshape(len(X_orig), len(TLD_LIST))
        y_prob_mutants = y_prob_mutants.reshape(len(X_orig), len(TLD_LIST))
        X_failing_mutants = X_failing_mutants[failing_tests].flatten()
        y_prob_mutants = y_prob_mutants[failing_tests]
    
        labels_repaired = automatic_repair(trained_model, preprocessor=preprocessor, X_failing_mutants=X_failing_mutants, y_prob_original=y_prob_original)
        labels_final[failing_tests] = labels_repaired
    
    # Heuristic to check that we get at least 90% correct.
    assert np.sum(np.equal(labels_final, labels_original)) > len(labels_original) // 1.1
    

if __name__ == "__main__":
    pytest.main() 
