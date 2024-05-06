"""
    Tests the model on test data.
"""

import numpy as np
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from utils import load_from_pickle_file


def test():
    """Performs the testing of the trained model.
    """

    x_test, y_test = load_from_pickle_file("data/tokenized/test.pkl")
    model = load_model("model/model.keras")

    y_pred = model.predict(x_test[:10000], batch_size=1000)

    y_pred_binary = (np.array(y_pred) > 0.5).astype(int)
    y_test = y_test[:10000].reshape(-1, 1)

    # Calculate classification report
    report = classification_report(y_test, y_pred_binary)
    print(f"Classification Report:\n{report}")

    # Calculate confusion matrix
    confusion_mat = confusion_matrix(y_test, y_pred_binary)
    print(f"Confusion Matrix:\n{confusion_mat}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_binary)}")

    sns.heatmap(confusion_mat, annot=True)


if __name__ == "__main__":
    test()
