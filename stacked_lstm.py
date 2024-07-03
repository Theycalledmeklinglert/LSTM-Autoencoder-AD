import csv
import random

import numpy as np
from keras import Sequential
from keras.src.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from keras.src.layers import LSTM, Dense, Dropout, BatchNormalization
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from raw_data_processing.rosbag_file_conversion import csv_file_to_dataframe_to_numpyArray


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


    #TODO: split vN1 and vN2 into X_vN1 and Y_vN1 / X_vN2 and Y_vN2 (-->i guess Y should just be X data with all indexes shifted by future_steps to the left?
def split_data_sequence_into_datasets(arr):
    train_ratio = 0.7
    val1_ratio = 0.0  #TODO: should be 0.2; haven't implemented early_stopping using val1 yet
    val2_ratio = 0.2  #TODO: temp bandaid while early_stopping is not implemented
    test_ratio = 0.1

    assert (train_ratio*10 + val1_ratio*10 + val2_ratio*10 + test_ratio*10) == 10   #due to stupid floating point assertionError

    n_total = len(arr)
    n_train = int(train_ratio * n_total)
    n_val1 = int(val1_ratio * n_total)
    n_val2 = int(val2_ratio * n_total)
    n_test = n_total - n_train - n_val1 - n_val2  # To ensure all samples are used

    print(f"Total samples: {n_total}")
    print(f"Training samples: {n_train}")
    #print(f"Validation 1 samples: {n_val1}")
    print(f"Validation 2 samples: {n_val2}")
    print(f"Test samples: {n_test}")

    # Split data sequentially
    sN = arr[:n_train]
    #vN1 = arr[n_train:n_train + n_val1]
    vN2 = arr[n_train + n_val1:n_train + n_val1 + n_val2]
    tN = arr[n_train + n_val1 + n_val2:n_train + n_val1 + n_val2 + n_test]
    print(f"Training set size: {len(sN)}")
    #print(f"Validation set 1 size: {len(vN1)}")
    print(f"Validation set 2 size: {len(vN2)}")
    print(f"Test set size: {len(tN)}")

    print("sN df: " + str(sN))
    #print("vN1 df: " + str(vN1))
    print("vN2 df: " + str(vN2))
    print("tN df: " + str(tN))

    #return sN, vN1, vN2, tN
    return sN, vN2, tN


def reshape_data_for_LSTM(data, time_steps):
    # Reshape X to fit LSTM input shape (samples, time steps, features)
    print(data.shape)
    data = data.reshape((data.shape[0], time_steps, data.shape[2]))
    print("Reshaped data for LSTM into: " + str(data))
    return data

def check_shapes_after_reshape(X_sN, X_vN2, X_tN, Y_sN, Y_vN2, Y_tN):
    shapes = [X_sN.shape, X_vN2.shape, X_tN.shape, Y_sN.shape, Y_vN2.shape, Y_tN.shape]
    print("Shapes of arrays after reshaping:")
    for i, shape in enumerate(shapes):
        print(f"Array {i+1}: {shape}")

    # Check if all arrays have the same shape in terms of time_steps and features
    try:
        shape_to_compare = (shapes[0][1], shapes[0][2])
        if not all((shape[1], shape[2]) == shape_to_compare for shape in shapes):
            raise ValueError("Shapes of reshaped arrays are not consistent in terms of time_steps and features!")
    except ValueError as e:
        print(f"Error: {str(e)}")

def test_stacked_LSTM(csv_path):
    #data = generate_test_array()
    time_steps = 1  # Use the 3 most recent value
    future_steps = 3  # Predict the next 3 values
    accuracy_threshold = 0.90
    accuracy_threshold_callback = AccuracyThresholdCallback(threshold=accuracy_threshold)

    data = csv_file_to_dataframe_to_numpyArray(csv_path)
    print(data.shape)

    #TODO: I need to standardize scaling somehow. for steering angle i.e. data can approximately go from -100 to 100 while time is very large
    # todo:experimental
    # todo:experimental

    feature_1 = data[:, 0].reshape(-1, 1)  # The large-scale feature
    feature_2 = data[:, 1].reshape(-1, 1)  # The smaller-scale feature
    scaler = StandardScaler()
    normalized_feature_1 = scaler.fit_transform(feature_1)
    normalized_feature_2 = scaler.fit_transform(feature_2)

    data = np.hstack((normalized_feature_1, normalized_feature_2))
    print("HERE FUCK: " + str(data))

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

    return

    model.fit(X_sN, Y_sN.reshape((Y_sN.shape[0], future_steps * Y_sN.shape[2])), epochs=2000, batch_size=32, validation_data=(X_vN2, Y_vN2.reshape(Y_vN2.shape[0], future_steps*Y_vN2.shape[2])), verbose=1, callbacks=[accuracy_threshold_callback])
    model.save('./models/stacked_LSTM.keras')


    #m (=input data dimensions) input units; dxl (d = features to be predicted, number of time steps to be predicted into future) output units
    rand_int = random.randint(0, X_tN.shape[0]-3)
    recent_sequence = np.array(X_tN[rand_int])  # insert sequence to be predicted here #np.array([[31, 32, 33, 34, 35]])
    print("Sequence chosen for prediction: " + str(recent_sequence))
    print(recent_sequence.shape)
    # Reshape recent_sequence to fit LSTM input shape (samples, time steps, features)
    recent_sequence = recent_sequence.reshape((1, recent_sequence.shape[0], recent_sequence.shape[1]))

    # Predict future_steps sequences
    predicted_sequences = model.predict(recent_sequence)

    # Reshape predicted sequences to match the original y shape
    predicted_sequences = predicted_sequences.reshape((future_steps, X_tN.shape[2])) #data.shape[1])

    print(f"Predicted sequences: \n{predicted_sequences}")


