import pytest
import os


def test_legitimate_ratio():
    legitimate = 0
    phishing = 0
    with open(os.path.join("data", "DL Dataset", "train.txt"), "r") as f:
        for line in f:
            if line.startswith("legitimate"):
                legitimate += 1
            if line.startswith("phishing"):
                phishing += 1
    total = legitimate + phishing
    legitimate_ratio = legitimate / total
    print(f"legitimate_ratio={legitimate_ratio:.2f}")
    assert 0.4 < legitimate_ratio < 0.6


if __name__ == "__main__":
    # pytest.main()
    test_legitimate_ratio()
