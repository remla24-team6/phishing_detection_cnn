from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

output_path = os.path.join('output', 'tokenized')

def preprocess():
    train = [line.strip() for line in open("data/DL Dataset/train.txt", "r").readlines()[1:]]
    raw_x_train = [line.split("\t")[1] for line in train]
    raw_y_train = [line.split("\t")[0] for line in train]

    test = [line.strip() for line in open("data/DL Dataset/test.txt", "r").readlines()]
    raw_x_test = [line.split("\t")[1] for line in test]
    raw_y_test = [line.split("\t")[0] for line in test]

    val=[line.strip() for line in open("data/DL Dataset/val.txt", "r").readlines()]
    raw_x_val=[line.split("\t")[1] for line in val]
    raw_y_val=[line.split("\t")[0] for line in val]

    tokenizer = Tokenizer(lower=True, char_level=True, oov_token='-n-')
    tokenizer.fit_on_texts(raw_x_train + raw_x_val + raw_x_test)
    char_index = tokenizer.word_index
    sequence_length = 200
    x_train = pad_sequences(tokenizer.texts_to_sequences(raw_x_train), maxlen=sequence_length)
    x_val = pad_sequences(tokenizer.texts_to_sequences(raw_x_val), maxlen=sequence_length)
    x_test = pad_sequences(tokenizer.texts_to_sequences(raw_x_test), maxlen=sequence_length)

    encoder = LabelEncoder()

    y_train = encoder.fit_transform(raw_y_train)
    y_val = encoder.transform(raw_y_val)
    y_test = encoder.transform(raw_y_test)

    with open(os.path.join(output_path, 'char_index.pkl'), 'wb') as f:
        pickle.dump(char_index, f)

    with open(os.path.join(output_path, 'x_train.pkl'), 'wb') as f:
        pickle.dump(x_train, f)
    with open(os.path.join(output_path, 'x_val.pkl'), 'wb') as f:
        pickle.dump(x_val, f)
    with open(os.path.join(output_path, 'x_test.pkl'), 'wb') as f:
        pickle.dump(x_test, f)

    with open(os.path.join(output_path, 'y_train.pkl'), 'wb') as f:
        pickle.dump(y_train, f)
    with open(os.path.join(output_path, 'y_val.pkl'), 'wb') as f:
        pickle.dump(y_val, f)
    with open(os.path.join(output_path, 'y_test.pkl'), 'wb') as f:
        pickle.dump(y_test, f)


if __name__ == "__main__":
    preprocess()