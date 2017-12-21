import numpy as np
import pandas as pd
import datetime
import csv
import os
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.core import Dropout
from keras.optimizers import SGD
from keras.optimizers import Adam
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.initializers import TruncatedNormal
from keras.callbacks import EarlyStopping

df = pd.read_csv("train_copy.csv", encoding="SHIFT-JIS")

X = df.loc[:, "c1":"c88"]
X = X.as_matrix().astype('double')
Y = df.loc[:, "target"]
Y = Y.as_matrix().astype('double')
Y = np.reshape(Y, (len(Y), 1))
print("X", X)
print('Y', Y)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

"""モデル設定"""


def weight_variable(shape):
    return K.truncated_normal(shape, stddev=0.01)


n_hiddens = []  # 隠れ層
epochs = 1000
result_acc = []
result_loss = []
best_acc = [0.0, 0, 0, 0]
best_loss = [1.0, 0, 0, 0]

n_in = len(X[0])
n_out = 1
activation = 'sigmoid'

model = Sequential()
model.add(Dense(input_dim=n_in, units=n_out,
                kernel_initializer=weight_variable))
model.add(BatchNormalization())
model.add(Activation(activation))

model.add(Dense(n_out, kernel_initializer=TruncatedNormal(stddev=0.01)))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999),
              metrics=['accuracy'])

model.load_weights('LR_weights1.hdf5')

"""モデル学習"""

batch_size = 10
hist = model.fit(X_train, Y_train, epochs=epochs,
                 batch_size=batch_size,
                 validation_data=(X_test, Y_test))

model.save_weights('LR_weights2.hdf5')

"""プロット"""
val_acc = hist.history['val_acc']
val_loss = hist.history['val_loss']

plt.rc('font', family='serif')
fig = plt.figure()
plt.plot(range(epochs), val_acc, label='acc', color='black')
plt.xlabel('epochs')
# plt.show()
plt.savefig('LR_.png')
ax_acc = fig.add_subplot(111)
ax_acc.plot(range(epochs), val_acc, label='acc', color='black')
ax_loss = ax_acc.twinx()
ax_loss.plot(range(epochs), val_loss, label='loss', color='gray')
plt.xlabel('epochs')
plt.savefig('LR_.png')

"""予測精度評価"""
print(model.evaluate(X_test, Y_test))


