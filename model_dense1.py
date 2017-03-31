from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import CSVLogger

import numpy as np
np.random.seed(123)
from keras.utils import np_utils

from keras.optimizers import SGD


def load_data():
    print('loading data...')
    input_mat = np.load('feats/inputs.npy', allow_pickle=True)
    target_mat= np.load('feats/targets.npy', allow_pickle=True)
    return (input_mat, target_mat)


def create_model():
    print('-----creating model-----')
    model = Sequential()
    model.add(Dense(1024, batch_input_shape=(None, 748), activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(128, activation='sigmoid'))


    sgd = SGD(lr=0.003, decay=0.0)
    model.compile(loss='mean_squared_error',
                  optimizer='sgd',
                  metrics=['accuracy'])
    return model

def train_and_evaluate_model(X_train, Y_train, model, epochs, val, save_name):
    print('training and evaluation')
    csv_logger = CSVLogger(save_name + '.log')
    model.fit(X_train, Y_train, validation_split=val, batch_size=200, nb_epoch=epochs, verbose=1, callbacks=[csv_logger])

def save_model(model, name):
    model_json = model.to_json()
    with open(name + ".json", "w") as json_file:
        json_file.write(model_json)

import sys
import time
# 1. name
# 2. number of epochs
# 3. percentage eval
# python3 model_dense1.py dense1 100 0.2
if __name__ == "__main__":
    print('Starting...')
    save_name = sys.argv[1]
    timestr = time.strftime("%Y%m%d-%H%M%S")
    save_name = save_name + "_" + timestr
    epochs = int(sys.argv[2])
    val = float(sys.argv[3])
    X_train, Y_train = load_data()
    X_train = X_train.T
    Y_train = Y_train.T

    print(X_train.shape)
    print(Y_train.shape)
    model = create_model()
    train_and_evaluate_model(X_train, Y_train, model, epochs, val, save_name)
    save_model(model, "./models/" + save_name)
    model.save_weights("./models/" + save_name + "_weights.h5")
    #n_folds = 1
    #skf = StratifiedKFold(labels, n_folds=n_folds, shuffle=True)