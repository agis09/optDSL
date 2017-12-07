import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.core import Dropout
from keras.optimizers import SGD
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

mnist = datasets.fetch_mldata('MNIST original', data_home='.')
n = len(mnist.data)
N = 10000   # MNISTの一部データで実験
indices = np.random.permutation(range(n))[:N]    # ランダムにN枚選択
X = mnist.data[indices]
y = mnist.target[indices]
Y = np.eye(10)[y.astype(int)]   # 1-of-K表現に変換
 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8)
 
"""モデル設定"""
 
n_in = len(X[0])    # 入力層次元784(28*28)
n_hidden = 200  # 隠れ層200次元
n_out = len(Y[0])   # 出力層次元10
 
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
odel.add(Activation('softmax'))
 
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.01),
              metrics=['accuracy'])

"""モデル学習"""
 
epochs = 1000
batch_size = 100

model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)
 
 """予測精度評価"""
 
loss_and_metrics = model.evaluate(X_test, Y_test)
print(loss_and_metrics)
