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

df = pd.read_csv("data1-1.csv", encoding="SHIFT-JIS")

X = df.loc[:, "a_0q21":"p_18q131"]
X = X.as_matrix().astype('int')
Y = df.loc[:, "初診時：転帰"]
Y = Y.as_matrix().astype('int')

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8)


"""モデル設定"""

n_in = len(X[0])
n_hidden = 200  # 隠れ層200次元
n_out = 1

model = Sequential()
model.add(Dense(n_hidden, input_dim=n_in))
model.add(Activation('tanh'))
model.add(Dropout(0.5))

model.add(Dense(n_hidden))
model.add(Activation('tanh'))
model.add(Dropout(0.5))

model.add(Dense(n_hidden))
model.add(Activation('tanh'))
model.add(Dropout(0.5))

model.add(Dense(n_out))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy',
              optimizer=SGD(lr=0.01),
              metrics=['accuracy'])

"""モデル学習"""

epochs = 1000
batch_size = 100

model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)

"""予測精度評価"""

loss_and_metrics = model.evaluate(X_test, Y_test)
print(loss_and_metrics)