import pytest


@pytest.fixture
def data():
    return 'fixture returning data'

@pytest.fixture
def split_data(data):
    return 'Fixture splitting data'

def test_reproducibility(split_data):
    
    assert 1 == 2, f"Accuracies differ: {1} != {1}"

if __name__ == "__main__":
    pytest.main()
