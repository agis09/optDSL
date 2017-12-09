import numpy as np
import pandas as pd
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.core import Dropout
from keras.optimizers import SGD
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.initializers import TruncatedNormal

df = pd.read_csv("data1-3.csv", encoding="SHIFT-JIS")

X = df.loc[:2744, "a_0q65":"a_68q86"]
X = X.as_matrix().astype('float32')
Y = df.loc[:2744, "初診時：転帰"]
Y = Y.as_matrix().astype('float32')
Y = np.reshape(Y, (2745, 1))
print("X", X)
print('Y', Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
'''
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=0.1)
'''

"""モデル設定"""


def weight_variable(shape):
    return K.truncated_normal(shape, stddev=0.01)


n_hiddens = []  # 隠れ層
result = []
for times in range(1):
    layer_loop = 3
    for i in range(layer_loop):
        n_hiddens.append(200)
    n_in = len(X[0])
    n_out = 1
    p_keep = 0.5
    activation = 'relu'

    model = Sequential()
    for i, input_dim in enumerate(([n_in] + n_hiddens)[:-1]):
        model.add(Dense(n_hiddens[i], input_dim=input_dim,
                        kernel_initializer=weight_variable))
        model.add(BatchNormalization())
        model.add(Activation(activation))
        model.add(Dropout(p_keep))

    model.add(Dense(n_out, kernel_initializer=TruncatedNormal(stddev=0.01)))
    model.add(Activation('sigmoid'))
    learning_late = 0.01
    model.compile(loss='binary_crossentropy',
                  optimizer=SGD(lr=learning_late, momentum=0.09),
                  metrics=['accuracy'])

    """モデル学習"""
    for epoch_loop in range(200, 201):
        epochs = epoch_loop*5
        for batch_loop in range(2, 3):
            batch_size = batch_loop*10

            hist = model.fit(X_train, Y_train, epochs=epochs,
                             batch_size=batch_size,
                             validation_data=(X_test, Y_test))

            """予測精度評価"""
            print(model.evaluate(X_test, Y_test))
            result.append([hist.history['val_acc'][-1], layer_loop, epoch_loop, batch_loop])
            print(result)
            val_acc = hist.history['val_acc']
            plt.rc('font', family='serif')
            fig = plt.figure()
            plt.plot(range(epochs), val_acc, label='acc', color='black')
            plt.xlabel('epochs')
            plt.show()

