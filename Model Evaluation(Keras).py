import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


# 実験データ生成
N = 300
X, y = datasets.make_moons(N, noise=0.3)

Y = y.reshape(N, 1)
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8)


"""モデル設定"""
model = Sequential()
model.add(Dense(3, input_dim=2))
model.add(Activation('sigmoid'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer=SGD(lr=0.05),
              metrics=['accuracy'])     # accuracy計算


"""モデル学習"""

model.fit(X_train, y_train, epochs=500, batch_size=20)

loss_and_metrics = model.evaluate(X_test, y_test)

print(loss_and_metrics)
