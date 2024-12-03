import pandas as pd 
import numpy as np


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, LSTM, Activation, Input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras_tuner.engine.hyperparameters import HyperParameters
from keras_tuner.tuners import RandomSearch

## data preparation 
# split the data into training and testing
data = np.load('preprocessed_data.npy')
train_size = int(len(data) * 0.8)
test_size = len(data) - train_size

# takes remaining rows from train size and take the first feature for testing - ensure model generalizes well 
train, test = data[0:train_size, :], data[train_size:len(data),:1]
print(len(train), len(test))

# time-series dataset for lstm 
def create_ts_data(data, time_step):
    X_train, Y_train = [], []
    for i in range(len(data) - time_step - 1):
        X_train.append(data[i:(i + time_step), 0])
        Y_train.append(data[i + time_step, 0])
    return np.array(X_train), np.array(Y_train)

# training period of 3 years 
days = 300
X_train, Y_train = create_ts_data(train, days)
X_test, Y_test = create_ts_data(test, days)

X_train= X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test= X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
