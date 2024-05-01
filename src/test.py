import pickle
import numpy as np
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score

def test():
    with open('output/tokenized/x_test.pkl', 'rb') as f:
        x_test = pickle.load(f)
    with open('output/tokenized/y_test.pkl', 'rb') as f:
        y_test = pickle.load(f)

    model = load_model('output/model.keras')

    y_pred = model.predict(x_test[:10000], batch_size=1000)

    y_pred_binary = (np.array(y_pred) > 0.5).astype(int)
    y_test = y_test[:10000].reshape(-1, 1)

    # Calculate classification report
    report = classification_report(y_test, y_pred_binary)
    print('Classification Report:')
    print(report)

    # Calculate confusion matrix
    confusion_mat = confusion_matrix(y_test, y_pred_binary)
    print('Confusion Matrix:', confusion_mat)
    print('Accuracy:', accuracy_score(y_test, y_pred_binary))

    sns.heatmap(confusion_mat, annot=True)

if __name__ == "__main__":
    test()