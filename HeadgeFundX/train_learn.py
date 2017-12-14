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

df = pd.read_csv("train_copy.csv", encoding="SHIFT-JIS")

for loop in range(14):

    X = df.loc[loop*40000:loop*40000+39999, "c1":"c88"]
    X = X.as_matrix().astype('double')
    Y = df.loc[loop*40000:loop*40000+39999, "target"]
    Y = Y.as_matrix().astype('double')
    Y = np.reshape(Y, (len(Y), 1))
    print("X", X)
    print('Y', Y)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    """モデル設定"""


    def weight_variable(shape):
        return K.truncated_normal(shape, stddev=0.01)


    n_hiddens = []  # 隠れ層
    epochs = 1500
    result_acc = []
    result_loss = []
    best_acc = [0.0, 0, 0, 0]
    best_loss = [1.0, 0, 0, 0]

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

        """モデル学習"""
        # epochs = 100
        batch_size = 50

        hist = model.fit(X_train, Y_train, epochs=epochs,
                         batch_size=batch_size,
                         validation_data=(X_test, Y_test))

        """予測精度評価"""
        print(model.evaluate(X_test, Y_test))

        result_acc.append([layer_loop])
        result_loss.append([layer_loop])
        result_acc.append(hist.history['val_acc'])
        result_loss.append(hist.history['val_loss'])

        if best_loss[0] > hist.history['val_loss'][-1]:
            best_loss = [hist.history['val_loss'][-1], layer_loop]
        if best_acc[0] < hist.history['val_acc'][-1]:
            best_acc = [hist.history['val_acc'][-1], layer_loop]

        """プロット"""
        val_acc = hist.history['val_acc']
        val_loss = hist.history['val_loss']

        plt.rc('font', family='serif')
        fig = plt.figure()
        plt.plot(range(epochs), val_acc, label='acc', color='black')
        plt.xlabel('epochs')
        # plt.show()
        plt.savefig('train' + str(loop+1) + 'acc_hiddenlayers' + str(times) + '.png')
        ax_acc = fig.add_subplot(111)
        ax_acc.plot(range(epochs), val_acc, label='acc', color='black')
        ax_loss = ax_acc.twinx()
        ax_loss.plot(range(epochs), val_loss, label='loss', color='gray')
        plt.xlabel('epochs')
        plt.savefig('train' + str(loop+1) + 'acc-loss_hiddenlayers_' + str(times) + '.png')

        """save weights"""
        model.save_weights('layer' + str(times) + '_train' + str(loop+1) + '_weights.hdf5')
        w_list = []
        header = ['acc', 'loss']

        for (acc, loss) in zip(result_acc, result_loss):
            for (i, j) in zip(acc, loss):
                w_list.append([i, j])
            w_list.append(['##################', '##################'])
        with open('layer' + str(times) + '_train' + str(loop + 1) + 'result.csv', 'a', newline='') as file:
            csvWriter = csv.writer(file)
            csvWriter.writerow(header)
            csvWriter.writerows(w_list)
            csvWriter.writerow(['best_acc', 'layer_num'])
            csvWriter.writerow(best_acc)
            csvWriter.writerow(['best_loss', 'layer_num'])
            csvWriter.writerow(best_loss)


