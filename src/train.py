from model import model
import yaml
import pickle

def train():
    model.compile(loss=params['loss_function'], optimizer=params['optimizer'], metrics=['accuracy'])

    with open('output/tokenized/x_train.pkl', 'rb') as f:
        x_train = pickle.load(f)
    with open('output/tokenized/y_train.pkl', 'rb') as f:
        y_train = pickle.load(f)
    with open('output/tokenized/x_val.pkl', 'rb') as f:
        x_val = pickle.load(f)
    with open('output/tokenized/y_val.pkl', 'rb') as f:
        y_val = pickle.load(f)

    hist = model.fit(x_train[:10000], y_train[:10000],
                    batch_size=params['batch_train'],
                    epochs=params['epoch'],
                    shuffle=True,
                    validation_data=(x_val[:10000], y_val[:10000])
                    )

    model.save('output/model.keras')

if __name__ == "__main__":
    with open('training_params.yaml', 'r') as file:
        params = yaml.safe_load(file)
    train()