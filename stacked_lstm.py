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

time_steps = 3  # Use the most recent value
future_steps = 3  # Predict the next 3 values

print(data[0:(0 + time_steps), :])
print(data[0:(0 + time_steps), :].shape)

def create_sequences(data, time_steps, future_steps):
    for i in range(len(data) - time_steps - future_steps + 1):
        X.append(data[i:(i + time_steps), :])
        Y.append(data[(i + time_steps):(i + time_steps + future_steps), :])

    print(np.array(X))
    print("\n")
    print(np.array(Y))
    return np.array(X), np.array(Y)


X, Y = create_sequences(data, time_steps, future_steps)
print("X: " + str(X))
print("Y: " + str(Y))

# Reshape X to fit LSTM input shape (samples, time steps, features)
print(X.shape)
print(X.shape[0])
print(X.shape[2])
X = X.reshape((X.shape[0], time_steps, X.shape[2]))

model = Sequential()

# todo: Experiment with different number of units in hidden layers
model.add(LSTM(X.shape[2], return_sequences=True, input_shape=(time_steps, X.shape[2])))  #todo: try ", activation='sigmoid'" -->should be sigmoid by default i believe
model.add(LSTM(50, return_sequences=True))  # Second LSTM layer #todo: Experiment with different number of units in hidden layers
model.add(LSTM(50))  # Third LSTM layer, does not return sequences
model.add(Dense(future_steps * X.shape[2]))  # Output layer for regression (use appropriate activation for classification tasks)
model.compile(optimizer='adam', loss='mse')  # Use 'binary_crossentropy' for binary classification #standard: mse

#todo: check what the 2nd reshape here does?
model.fit(X, Y.reshape((Y.shape[0], future_steps * X.shape[2])), epochs=100, verbose=1)

#TODO: m (=input data dimensions) input units; dxl (d = features to be predicted, number of time steps to be predicted into future) output units

# Prepare the most recent sequence for prediction
recent_sequence = np.array([[31, 32, 33, 34, 35]])  #insert sequence to be predicted here

# Reshape recent_sequence to fit LSTM input shape (samples, time steps, features)
recent_sequence = recent_sequence.reshape((recent_sequence.shape[0], time_steps, recent_sequence.shape[1]))

# Predict future_steps sequences
predicted_sequences = model.predict(recent_sequence)

# Reshape predicted sequences to match the original y shape
predicted_sequences = predicted_sequences.reshape((future_steps, data.shape[1]))

print(f"Predicted sequences: \n{predicted_sequences}")