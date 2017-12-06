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


df = pd.read_csv("data1-1.csv", encoding="SHIFT-JIS")

X = df.loc[:, "a_0q21":"p_18q131"]
X = X.as_matrix().astype('int')
Y = df.loc[:, "初診時：転帰"]
Y = Y.as_matrix().astype('int')

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8)
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, train_size=0.9)


"""モデル設定"""


def weight_variable(shape):
    return K.truncated_normal(shape, stddev=0.01)


n_in = len(X[0])
n_hiddens = [200, 200, 200]  # 隠れ層200次元
n_out = 1
p_keep = 0.5
activation = 'relu'

model = Sequential()
for i, input_dim in enumerate(([n_in] + n_hiddens)[:-1]):
    model.add(Dense(n_hiddens[i], input_dim=input_dim,
              kernel_initializer=weight_variable))
    model.add(Activation(activation))
    model.add(Dropout(p_keep))

model.add(Dense(n_out, kernel_initializer=weight_variable))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy',
              optimizer=SGD(lr=0.01),
              metrics=['accuracy'])

"""モデル学習"""

epochs = 1000
batch_size = 100

hist = model.fit(X_train, Y_train, epochs=epochs,
                 batch_size=batch_size,
                 validation_data=(X_validation, Y_validation))

"""予測精度評価"""

val_acc = hist.history['val_acc']

plt.rc('font', family='serif')
fig = plt.figure()
plt.plot(range(epochs), val_acc, label='acc', color='black')
plt.xlabel('epochs')
plt.show()
