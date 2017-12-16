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
for p in range(1, 15):
    X = np.empty((0, 88))
    id = []
    for loop in range(361500):
        if df.loc[loop, "period"][4:] == str(p):
            id.append(df.loc[loop, "data_id"])
            x = df.loc[loop:loop, "c1":"c88"]
            x = x.as_matrix().astype('double')
            # print(x)
            X = np.append(X, np.array(x), axis=0)

    print("X", X)
    print("進捗:", p)
    if len(X) == 0:
        continue
    """モデル設定"""

    def weight_variable(shape):
        return K.truncated_normal(shape, stddev=0.01)


    n_hiddens = []  # 隠れ層
    epochs = 1500

    for times in range(1, 2):
        layer_loop = times
        for i in range(layer_loop):
            n_hiddens.append(100)
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
        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999),
                      metrics=['accuracy'])
        model.load_weights('layer2_trainall_weights.hdf5')
        """モデル予測"""

        hist = model.predict(X, verbose=1)
        for i, j in zip(id, hist):
            id_v.update({i:j[0]})

    """出力"""

with open('predict.csv', 'a', newline='') as file:
    csvWriter = csv.writer(file)
    for i in sorted(id_v.keys()):
        csvWriter.writerow([i, id_v[i]])