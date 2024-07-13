import random

import numpy as np
import tensorflow as tf
from keras import Loss

from raw_data_processing.data_processing import reverse_normalize_data

# test
def autoencoder_predict_and_calculate_error(model, X_tN, future_steps, iterations, scaler):
    all_err_vecs = []
    for i in range(0, iterations):
        rand_int = random.randint(0, X_tN.shape[0]-1)
        chosen_sequence = np.array(X_tN[rand_int])  # sequence to be predicted
        # Reshape chosen_sequence to fit LSTM input shape (samples, time steps, features)
        chosen_sequence = chosen_sequence.reshape((1, chosen_sequence.shape[0], chosen_sequence.shape[1]))

        predicted_sequence = model.predict([chosen_sequence, np.flip(chosen_sequence, axis=1)], verbose=0)
        # Reshape predicted sequences to match the original y shape

        chosen_sequence = reverse_normalize_data(np.squeeze(chosen_sequence, axis=0),
                                                 scaler)  # Reverse reshaping and normalizing
        predicted_sequence = reverse_normalize_data(np.squeeze(predicted_sequence, axis=0),
                                                    scaler)  # Reverse reshaping and normalizing

        #predicted_sequence = predicted_sequence.reshape((future_steps, X_tN.shape[2]))
        #print("Input sequence: " + str(chosen_sequence))
        #print("Predicted sequences: " + str(predicted_sequence))
        error_vec = np.subtract(chosen_sequence, predicted_sequence)
        all_err_vecs.append(error_vec)
        #print("Error vec: " + str(error_vec) + "\n")
    avg_error_matrix = np.mean(all_err_vecs, axis=0)
    print("Avg. error: " + str(avg_error_matrix))
    print("Avg. error (now with 20% less cancer!): " + str(np.mean(avg_error_matrix, axis=0)))


class CustomL2Loss(Loss):
    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))

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










