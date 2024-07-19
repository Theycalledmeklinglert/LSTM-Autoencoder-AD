import csv
import random

import numpy as np
from keras import Sequential
from keras.src.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from keras.src.layers import LSTM, Dense, Dropout, BatchNormalization
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from raw_data_processing.data_processing import csv_files_to_dataframe_to_numpyArray, \
    convert_timestamp_to_absolute_time_diff, convert_timestamp_to_relative_time_diff


def generate_test_array(start=0, end=1000, sequence_length=5):
    sequences = []
    for i in range(start, end + 1, sequence_length):
        sequence = list(range(i, i + sequence_length))
        sequences.append(sequence)
    return np.array(sequences)


#TODO: I think this works correctly? ----> bruh
#TODO: THIS IS A HUGE POSSIBLE SOURCE OF ERROR!! THIS HAS HAS HAS TO WORK CORRECTLY!!!!!!
def create_XY_data_sequences(data, time_steps, future_steps):
    X, Y = [], []
    i = 0
    while (i*time_steps + time_steps + future_steps) < len(data):
    #for i in range(len(data) - time_steps - future_steps + 1):
        x_data_low = i*time_steps
        x_data_high = i*time_steps + time_steps
        y_data_low = x_data_high
        y_data_high = y_data_low + future_steps
        X.append(data[x_data_low:x_data_high, :])
        Y.append(data[y_data_low:y_data_high, :])
        i = i+1

    print("Created X sequence: " + str(np.array(X)))
    print("\n")
    print("Created Y sequence: " + str(np.array(Y)))
    return np.array(X), np.array(Y)

def write_XY_to_csv(X, Y, filename, row_offset=3):
    # Determine the shape of the new array
    total_rows = X.shape[0] + row_offset
    x_cols = X.shape[1] * X.shape[2]
    y_cols = Y.shape[1] * Y.shape[2]
    total_cols = x_cols + y_cols

    # Create an empty array with the desired shape
    combined_data = np.empty((total_rows, total_cols))
    combined_data[:] = np.nan  # Fill with NaN to handle the empty rows

    # Flatten X and Y to fit into the combined array
    X_flat = X.reshape(X.shape[0], -1)
    Y_flat = Y.reshape(Y.shape[0], -1)

    # Place X into the combined array starting from row_offset
    combined_data[row_offset:row_offset + X_flat.shape[0], :X_flat.shape[1]] = X_flat

    # Place Y into the combined array starting from row_offset
    combined_data[row_offset:row_offset + Y_flat.shape[0], X_flat.shape[1]:] = Y_flat

    # Write the combined array to a CSV file
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        # Create headers
        headers = [f"X{i+1}" for i in range(X_flat.shape[1])] + [f"Y{i+1}" for i in range(Y_flat.shape[1])]
        writer.writerow(headers)
        writer.writerows(combined_data)


class AccuracyThresholdCallback(Callback):
    def __init__(self, threshold):
        super(AccuracyThresholdCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        accuracy = logs.get('accuracy')
        if accuracy is not None and accuracy > self.threshold:
            print(f"\nEpoch {epoch + 1}: accuracy {accuracy} exceeded threshold {self.threshold}, stopping training.")
            self.model.stop_training = True


def test_stacked_LSTM(csv_path):
    #data = generate_test_array()
    time_steps = 1  # Use the 3 most recent value
    future_steps = 3  # Predict the next 3 values
    accuracy_threshold = 0.90
    accuracy_threshold_callback = AccuracyThresholdCallback(threshold=accuracy_threshold)

    data = csv_files_to_dataframe_to_numpyArray(csv_path)
    print(data.shape)

    #TODO: I need to standardize scaling somehow. for steering angle i.e. data can approximately go from -100 to 100 while time is very large
    # todo:experimental
    # todo:experimental

    # feature_1 = data[:, 0].reshape(-1, 1)  # The large-scale feature
    # feature_2 = data[:, 1].reshape(-1, 1)  # The smaller-scale feature
    # scaler = StandardScaler()
    # normalized_feature_1 = scaler.fit_transform(feature_1)
    # normalized_feature_2 = scaler.fit_transform(feature_2)
    #
    # data = np.hstack((normalized_feature_1, normalized_feature_2))

    #Todo: Test with this
    #data_with_time_diffs = convert_timestamp_to_absolute_time_diff(data)
    data_with_time_diffs = convert_timestamp_to_relative_time_diff(data)

    print(str(data_with_time_diffs))

    print("HERE FUCK: " + str(data))

    return

    X, Y = create_XY_data_sequences(data, time_steps, future_steps)

    print("X: " + str(X))
    print("Y: " + str(Y))
    X_sN, X_vN2, X_tN = split_data_sequence_into_datasets(X)
    Y_sN, Y_vN2, Y_tN = split_data_sequence_into_datasets(Y)
    X_sN = reshape_data_for_LSTM(X_sN, time_steps)
    X_vN2 = reshape_data_for_LSTM(X_vN2, time_steps)
    X_tN = reshape_data_for_LSTM(X_tN, time_steps)

    Y_sN = reshape_data_for_LSTM(Y_sN, future_steps)  #used to be time_steps
    Y_vN2 = reshape_data_for_LSTM(Y_vN2, future_steps)
    Y_tN = reshape_data_for_LSTM(Y_tN, future_steps)
    check_shapes_after_reshape(X_sN, X_vN2, X_tN, Y_sN, Y_vN2, Y_tN)

    model = Sequential()
    # todo: Experiment with different number of units in hidden layers
    #model.add(BatchNormalization())
    model.add(LSTM(X_sN.shape[2], return_sequences=True, input_shape=(time_steps, X_sN.shape[2]), activation='tanh', recurrent_activation='sigmoid'))
    model.add(Dropout(0.2))
    #model.add(LSTM(50, return_sequences=True, activation='tanh', recurrent_activation='sigmoid'))
    #model.add(Dropout(0.2))
    model.add(LSTM(units=X_sN.shape[2]*time_steps, activation='tanh', recurrent_activation='sigmoid'))  # Third LSTM layer, does not return sequences
    model.add(Dropout(0.2))
    #todo: not sure if sigmoid is useful here #activation='sigmoid'
    model.add(Dense(future_steps * X_sN.shape[2], activation='sigmoid'))  # Output layer for regression (use appropriate activation for classification tasks)
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])  # Use 'binary_crossentropy' for binary classification #standard: mse
    model.summary()
    #early_stopping_callback = EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True, mode='min', min_delta=0.001)
    #lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)
    # todo: check what the 2nd reshape here does? -->The target data Y needs to be reshaped to match the expected output shape of the model, which is (samples, future_steps * features).

    print(str(Y_sN.reshape((Y_sN.shape[0], future_steps * Y_sN.shape[2])).shape))
    print(str(Y_sN.reshape((Y_sN.shape[0], future_steps * Y_sN.shape[2]))))

    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, restore_best_weights=True)
    model.fit(X_sN, Y_sN.reshape((Y_sN.shape[0], future_steps * Y_sN.shape[2])), epochs=2000, batch_size=32, validation_data=(X_vN2, Y_vN2.reshape(Y_vN2.shape[0], future_steps*Y_vN2.shape[2])), verbose=1, callbacks=[early_stopping])
    model.save('./models/stacked_LSTM.keras')

    #m (=input data dimensions) input units; dxl (d = features to be predicted, number of time steps to be predicted into future) output units
    stacked_LSTM_predict_and_calculate_error(model, X_tN, Y_tN, future_steps, 10)


