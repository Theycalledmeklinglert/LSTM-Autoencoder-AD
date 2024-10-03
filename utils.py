import os
import random
import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from data_processing import reverse_normalization, get_matching_file_pairs_from_directories, \
    get_normalized_data_and_labels, reshape_data_for_autoencoder_lstm



def autoencoder_predict_and_calculate_error(model, X_tN, labels, future_steps, iterations, scaler):
    normal_err_vecs = []
    anomaly_err_vecs = []
    all_err_vecs = []

    j = 0

    #for i in range(0, iterations):
    for i in range(X_tN.shape[0]):
        #rand_int = random.randint(0, X_tN.shape[0]-1)
        chosen_sequence = np.array(X_tN[i])  # sequence to be predicted
        # Reshape chosen_sequence to fit LSTM input shape (samples, time steps, features)
        chosen_sequence = chosen_sequence.reshape((1, chosen_sequence.shape[0], chosen_sequence.shape[1]))

        predicted_sequence = model.predict([chosen_sequence, np.flip(chosen_sequence, axis=1)], verbose=0)
        # Reshape predicted sequences to match the original y shape

        chosen_sequence = reverse_normalization(np.squeeze(chosen_sequence, axis=0),
                                                scaler)  # Reverse reshaping and normalizing
        predicted_sequence = reverse_normalization(np.squeeze(predicted_sequence, axis=0),
                                                   scaler)  # Reverse reshaping and normalizing

        #print("Iteration: " + str(i))
        #print("Input sequence: \n" + str(chosen_sequence))
        #print("Predicted sequences: \n" + str(predicted_sequence))
        error_vec = np.absolute(np.subtract(chosen_sequence, predicted_sequence))   #todo: np.absolute might be really counterproductive here

        if 1 in labels[i]:
            #print("In certified anomaly")
            #print(str(error_vec))
            anomaly_err_vecs.append(error_vec)

        else:
            #print("Normal error: \n" + str(error_vec))
            normal_err_vecs.append(error_vec)

        #all_err_vecs.append(error_vec)

        # if normal_err_vecs:
        #     avg_normal_error_matrix = np.mean(np.mean(normal_err_vecs, axis=0), axis=0)
        #     avg_error_vec_this_iteration = np.mean(error_vec, axis=0)
        #     if (avg_error_vec_this_iteration > avg_normal_error_matrix*3)[0]:
        #         print("a: \n" + str(avg_normal_error_matrix))
        #         print("b: \n" + str(avg_error_vec_this_iteration))
        #
        #         print("Possible anomaly detected at: \n")
        #         print(str(chosen_sequence))
        #print("Error vec: " + str(error_vec) + "\n")

    avg_normal_error_matrix = np.mean(normal_err_vecs, axis=0)
    # print("Avg. normal error: " + str(avg_normal_error_matrix))
    # print("Avg. anomaly error: " + str(avg_anomaly_error_matrix))
    #
    print("Avg. normal error (now with 20% less cancer!): " + str(np.mean(avg_normal_error_matrix, axis=0)))
    if not anomaly_err_vecs:
        avg_anomaly_error_matrix = 0
    else:
        avg_anomaly_error_matrix = np.mean(np.mean(anomaly_err_vecs, axis=0), axis=0)
    print("Avg. anomaly error (now with 20% less cancer!): " + str(avg_anomaly_error_matrix))
    return all_err_vecs


def stacked_LSTM_predict_and_calculate_error(model, X_tN, Y_tN, future_steps, iterations):
    all_err_vecs = []
    for i in range(0, iterations):
        rand_int = random.randint(0, X_tN.shape[0] - future_steps)
        chosen_sequence = np.array(X_tN[rand_int:(rand_int + future_steps), :])  # sequence to be predicted
        print("Input sequence: " + str(chosen_sequence))
        # Reshape chosen_sequence to fit LSTM input shape (samples, time steps, features)
        chosen_sequence = chosen_sequence.reshape((1, chosen_sequence.shape[0], chosen_sequence.shape[1]))
        predicted_sequence = model.predict(chosen_sequence, verbose=0)
        # Reshape predicted sequences to match the original y shape
        predicted_sequence = predicted_sequence.reshape((future_steps, X_tN.shape[2]))
        print("Predicted sequences: " + str(predicted_sequence))

        true_sequence = np.array(Y_tN[rand_int:(rand_int + future_steps)])
        true_sequence = true_sequence.reshape((future_steps, Y_tN.shape[2]))

        error_vec = np.subtract(true_sequence, predicted_sequence)
        all_err_vecs.append(error_vec)
        print("Error vec: " + str(error_vec) + "\n")

    avg_error_matrix = np.mean(all_err_vecs, axis=0)
    print("Avg. error: " + str(avg_error_matrix))
    print("Avg. error (now with 20% less cancer!): " + str(np.mean(avg_error_matrix, axis=0)))



def add_anomaly_column_to_csv_files(directories):
    for directory in directories:
        # List all files in the directory
        for filename in os.listdir(directory):
            if filename.endswith('.csv'):
                file_path = os.path.join(directory, filename)

                # Read the CSV file
                df = pd.read_csv(file_path)

                # Add the "Anomaly" column with value 0
                df['Anomaly'] = 0

                # Save the updated CSV file back to the same location
                df.to_csv(file_path, index=False)
                print(f"Updated {file_path}")


class DataFrameContainsNaNError(Exception):
    """Exception raised when the DataFrame contains NaN values."""
    def __init__(self, message="DataFrame contains NaN values."):
        self.message = message
        super().__init__(self.message)


class SensorFileColumnsContainsOnlyZeroesError(Exception):
    """Exception raised when the DataFrame contains NaN values."""
    def __init__(self, message="Sensor File contains at least one column with nothing but zeroes."):
        self.message = message
        super().__init__(self.message)


class SensorFileColumnsOnlyContainsSameValue(Exception):
    """Exception raised when the DataFrame contains NaN values."""
    def __init__(self, message="Sensor File contains at least one column with nothing but zeroes."):
        self.message = message
        super().__init__(self.message)


class InvalidReshapeParamters(Exception):
    def __init__(self, message="Window size must be greater than or equal to 1. \n window_step must be greater than 0 \n Window size must be greater than window_step"):
        self.message = message
        super().__init__(self.message)


