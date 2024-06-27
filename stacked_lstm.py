import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.src.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

data = np.array([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25],
    [26, 27, 28, 29, 30],
    [31, 32, 33, 34, 35]
])

X, Y = [], []


def create_sequences(data, time_steps=1, future_steps=1):
    for i in range(len(data) - time_steps - future_steps + 1):
        X.append(data[i:(i + time_steps), :])
        Y.append(data[(i + time_steps):(i + time_steps + future_steps), :])

    print(np.array(X))
    print("\n")
    print(np.array(Y))
    return np.array(X), np.array(Y)

#TODO: m (=input data dimensions) input units; dxl (d = features to be predicted, number of time steps to be predicted into future) output units

X, Y = create_sequences(data, time_steps=3, future_steps=3)
