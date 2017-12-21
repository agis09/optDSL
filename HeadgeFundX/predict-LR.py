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

df = pd.read_csv("test_updated.csv", encoding="SHIFT-JIS")
result = []
id_v = {}

ID = df.loc[:, "data_id"]
Period = df.loc[:, "period"]
C = df.loc[:, "c1":"c88"]
for p in range(1, 2):
    X = df.loc[:, "c1":"c88"]
    X = X.as_matrix().astype('double')
    print("X", X)
    print("進捗:", p)
    if len(X) == 0:
        continue
    """モデル設定"""

    def weight_variable(shape):
        return K.truncated_normal(shape, stddev=0.01)


    n_hiddens = []  # 隠れ層
    epochs = 30
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

    model.load_weights('LR_weights2.hdf5')
    """モデル予測"""

    hist = model.predict(X, verbose=1)
    for i, j in zip(ID, hist):
        id_v.update({i:j[0]})

    """出力"""

with open('predict_LR2.csv', 'a', newline='') as file:
    csvWriter = csv.writer(file)
    for i in sorted(id_v.keys()):
        csvWriter.writerow([i, id_v[i]])