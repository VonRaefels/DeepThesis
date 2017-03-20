from keras.models import Sequential
from keras.layers import Dense, Activation

import numpy as np
np.random.seed(123)
from keras.utils import np_utils

from keras.optimizers import SGD


def load_data():
    print('loading data...')
    input_mat = np.save('inputs_model1.npy', allow_pickle=True)
    target_mat= np.save('targets_model1.npy', allow_pickle=True)
    return (input_mat, target_mat)


def create_model():
    print('-----creating model-----')
    model = Sequential()
    model.add(Dense(1024, batch_input_shape=(None, 154), activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(154, activation='sigmoid'))


    sgd = SGD(lr=0.003, decay=0.0)
    model.compile(loss='mean_squared_error',
                  optimizer='sgd',
                  metrics=['accuracy'])

def train_and_evaluate_model():
    print('training and evaluation')
    model.fit(X_train, Y_train, batch_size=199, nb_epoch=2000, verbose=1)


if __name__ == "__main__":
    print('Starting...')
    data, targets = load_data()
    #n_folds = 1
    #skf = StratifiedKFold(labels, n_folds=n_folds, shuffle=True)