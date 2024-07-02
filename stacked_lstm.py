import random

import numpy as np
from keras import Sequential
from keras.src.callbacks import Callback
from keras.src.layers import LSTM, Dense

def generate_test_array(start=0, end=1000, sequence_length=5):
    sequences = []
    for i in range(start, end + 1, sequence_length):
        sequence = list(range(i, i + sequence_length))
        sequences.append(sequence)
    return np.array(sequences)

def test_stacked_LSTM():
    # data = np.array([
    #     [1, 2, 3, 4, 5],
    #     [6, 7, 8, 9, 10],
    #     [11, 12, 13, 14, 15],
    #     [16, 17, 18, 19, 20],
    #     [21, 22, 23, 24, 25],
    #     [26, 27, 28, 29, 30],
    #     [31, 32, 33, 34, 35]
    # ])

    data = generate_test_array()

    time_steps = 3  # Use the 3 most recent value
    future_steps = 3  # Predict the next 3 values

    print(data[0:(0 + time_steps), :])
    print(data[0:(0 + time_steps), :].shape)

    X, Y = create_sequences(data, time_steps, future_steps)
    print("X: " + str(X))
    print("Y: " + str(Y))

    # Reshape X to fit LSTM input shape (samples, time steps, features)
    print(X.shape)
    print(X.shape[0])
    print(X.shape[2])
    X = X.reshape((X.shape[0], time_steps, X.shape[2]))
    print(X)

    model = Sequential()
    # todo: Experiment with different number of units in hidden layers
    model.add(LSTM(X.shape[2], return_sequences=True, input_shape=(time_steps, X.shape[2])))  # todo: try ", activation='sigmoid'" -->should be sigmoid by default i believe
    model.add(LSTM(50, return_sequences=True))  # Second LSTM layer #todo: Experiment with different number of units in hidden layers
    model.add(LSTM(50))  # Third LSTM layer, does not return sequences
    model.add(Dense(future_steps * X.shape[2]))  # Output layer for regression (use appropriate activation for classification tasks)
    model.compile(optimizer='adam', loss='mse')  # Use 'binary_crossentropy' for binary classification #standard: mse

    loss_threshold = 0.01
    loss_threshold_callback = LossThresholdCallback(threshold=loss_threshold)
    # todo: check what the 2nd reshape here does?
    model.fit(X, Y.reshape((Y.shape[0], future_steps * X.shape[2])), epochs=6000, verbose=1, callbacks=[loss_threshold_callback])

    # TODO: m (=input data dimensions) input units; dxl (d = features to be predicted, number of time steps to be predicted into future) output units

    # Prepare the most recent sequence for prediction
    rand_int = random.randint(0, 200)
    recent_sequence = np.array([data[rand_int], data[rand_int+1], data[rand_int+2]])  # insert sequence to be predicted here #np.array([[31, 32, 33, 34, 35]])
    print("Sequence chosen for prediction: " + str(recent_sequence))
    print(recent_sequence.shape[0])
    print(recent_sequence.shape[1])
    # Reshape recent_sequence to fit LSTM input shape (samples, time steps, features)
    recent_sequence = recent_sequence.reshape((1, recent_sequence.shape[0], recent_sequence.shape[1]))

    # Predict future_steps sequences
    predicted_sequences = model.predict(recent_sequence)

    # Reshape predicted sequences to match the original y shape
    predicted_sequences = predicted_sequences.reshape((future_steps, data.shape[1]))

    print(f"Predicted sequences: \n{predicted_sequences}")

    model.save('./models/stacked_LSTM.keras')

def create_sequences(data, time_steps, future_steps):
    X, Y = [], []
    for i in range(len(data) - time_steps - future_steps + 1):
        X.append(data[i:(i + time_steps), :])
        Y.append(data[(i + time_steps):(i + time_steps + future_steps), :])

    print(np.array(X))
    print("\n")
    print(np.array(Y))
    return np.array(X), np.array(Y)

class LossThresholdCallback(Callback):
    def __init__(self, threshold):
        super(LossThresholdCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        if loss is not None and loss > self.threshold:
            print(f"\nEpoch {epoch + 1}: loss {loss} exceeded threshold {self.threshold}, stopping training.")
            self.model.stop_training = True