import numpy as np
import pandas as pd
import datetime
import csv
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.core import Dropout
from keras.optimizers import SGD
from keras.optimizers import Adam
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.initializers import TruncatedNormal

df = pd.read_csv("data1-3.csv", encoding="SHIFT-JIS")

X = df.loc[:2744, "a_0q65":"a_68q86"]
X = X.as_matrix().astype('float32')
Y = df.loc[:2744, "初診時：転帰"]
Y = Y.as_matrix().astype('float32')
Y = np.reshape(Y, (len(Y), 1))
print("X", X)
print('Y', Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

print(X.shape)
data_0 = np.empty((0, len(X[0])), float)
data_1 = np.empty((0, len(X[0])), float)
for i in range(len(X)):
    if Y[i][0] == 0.0:
        data_0 = np.append(data_0, [X[i]], axis=0)
    else:
        data_1 = np.append(data_1, [X[i]], axis=0)

x = np.concatenate((data_0, data_1), axis=0)

"""2D"""

pca = PCA(n_components=2)
pca.fit(X=x)

data_0_pca = pca.transform(X=data_0)
data_1_pca = pca.transform(X=data_1)

plt.scatter(data_0_pca[:, 0], data_0_pca[:, 1], color='blue', s=20, label='label_0')
plt.scatter(data_1_pca[:, 0], data_1_pca[:, 1], color='red', s=20, label='label_1')

plt.grid()
plt.legend()

plt.show()

"""3D"""
pca = PCA(n_components=3)
pca.fit(X=x)

data_0_pca = pca.transform(X=data_0)
data_1_pca = pca.transform(X=data_1)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(data_0_pca[:, 0], data_0_pca[:, 1], data_0_pca[:, 2], zdir='z', s=20, c='blue', label='label_0')
ax.scatter(data_1_pca[:, 0], data_1_pca[:, 1], data_1_pca[:, 2], zdir='z', s=20, c='red', label='label_1')
plt.grid()
plt.legend()
plt.show()
