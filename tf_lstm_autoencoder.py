import numpy as np
import tensorflow as tf
import keras
from keras.src.callbacks import EarlyStopping
from keras.src.saving import load_model
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy.linalg import inv
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
#from tensorflow import keras
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from keras import Loss

from data_processing import reshape_data_for_autoencoder_lstm, normalize_data, \
    split_data_sequence_into_datasets, reverse_normalization, csv_file_to_nparr, \
    transform_true_labels_to_window_size, get_normalized_data_and_labels
from utils import autoencoder_predict_and_calculate_error, \
    get_matching_file_pairs_from_directories


@keras.saving.register_keras_serializable(package="MyLayers")
class LSTMAutoEncoder(tf.keras.Model):
    def get_config(self):
        base_config = super().get_config()
        config = {
            "layer_dims": self.layer_dims,
            "input_dim": self.input_dim,
            "time_steps": self.time_steps,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        layer_dims = config.pop("layer_dims")
        input_dim = config.pop("input_dim")
        time_steps = config.pop("time_steps")
        num_layers = config.pop("num_layers")
        dropout = config.pop("dropout")

        return cls(input_dim, time_steps, layer_dims, num_layers, dropout, **config)

    def __init__(self, input_dim, time_steps, layer_dims, num_layers, dropout, **kwargs):
        super(LSTMAutoEncoder, self).__init__(**kwargs)
        self.layer_dims = layer_dims
        self.input_dim = input_dim
        self.time_steps = time_steps
        self.num_layers = num_layers
        self.dropout = dropout
        self.encoder_lstm = []
        self.decoder_lstm = []
        self.decoder_dense = None

        # Encoder
        #print("Im going fucking feral: " + str(layer_dims))
        for i in layer_dims:
            self.encoder_lstm.append(
                LSTM(i, activation='relu', return_sequences=True, return_state=True, dropout=dropout))
        #Note: HOlY FUCK am i retarded?
        # self.encoder_lstm.append(LSTM(layer_dims[0], activation='relu', return_sequences=True, return_state=True, dropout=dropout))
        # Decoder
        for i in reversed(layer_dims):
            self.decoder_lstm.append(
                LSTM(i, activation='relu', return_sequences=True, return_state=True, dropout=dropout))
        #self.decoder_lstm.append(LSTM(layer_dims[0], activation='relu', return_sequences=True, return_state=True, dropout=dropout))
        self.decoder_dense = Dense(input_dim)

    def call(self, inputs, training=False):
        #if training is None:
        print("Training argument is " + str(training))
        #return

        encoder_inputs, decoder_inputs = inputs
        x = encoder_inputs
        #print("This may be huge: " + str(decoder_inputs.shape))
        #print("This may be huge2: " + str(encoder_inputs))

        encoder_states = [tf.zeros((tf.shape(encoder_inputs)[0], self.layer_dims[0])),
                          tf.zeros((tf.shape(encoder_inputs)[0], self.layer_dims[0]))]
        empty_encoder_states_cond = True
        #encoder_states = [None, None]

        #I'm still not sure if this is correct. The previous function looked like this:
        #    # for lstm_layer in self.encoder_lstm[:-1]:   #all but last layer
        #         #     x, _, _ = lstm_layer(x)
        #         # encoder_outputs, state_h, state_c = self.encoder_lstm[-1](x)  # output of last layer

        #Should I pass the states inbetween layers or keep each layer's state in its corresponding layer and then only pass the states of the last layer to the decoder?
        for t in range(self.time_steps):
            x = encoder_inputs[:, t:t + 1, :]  # Select one time step
            for lstm_layer in self.encoder_lstm:
                if empty_encoder_states_cond:
                    x, state_h, state_c = lstm_layer(x)
                    empty_encoder_states_cond = False
                else:
                    x, state_h, state_c = lstm_layer(x, initial_state=encoder_states)
                encoder_states = [state_h, state_c]

        encoder_states = [state_h, state_c]
        # Use the first value of the reversed sequence as initial input for the decoder

        all_outputs = []
        inputs = decoder_inputs[:, 0:1, :]  #last entry in time-series
        for t in range(self.time_steps):
            for lstm_layer in self.decoder_lstm:
                decoder_outputs, state_h, state_c = lstm_layer(inputs, initial_state=encoder_states)
                inputs = decoder_outputs
            outputs = self.decoder_dense(decoder_outputs)
            all_outputs.append(outputs)
            if training:
                inputs = decoder_inputs[:, t:t + 1, :]  # Use the ground truth as the next input
            else:
                inputs = outputs  # Use the output as the next input
            encoder_states = [state_h, state_c]

        #print("All outputs: \n" + str(all_outputs))
        #reverse decoder output since decoder predicts target data in reverse
        all_outputs = all_outputs[::-1]                     #todo: I think this is correct now??? Not 100% sure
        #print("Reversed outputs: \n" + str(all_outputs))

        decoder_outputs = tf.concat(all_outputs, axis=1)
        return decoder_outputs

    def train_step(self, data):
        encoder_inputs, decoder_inputs = data[0]
        target = data[1]

        with tf.GradientTape() as tape:
            predictions = self([encoder_inputs, decoder_inputs], training=True)
            loss = self.compiled_loss(target, predictions, regularization_losses=self.losses)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(target, predictions)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        encoder_inputs, decoder_inputs = data[0]  # Unpacking encoder and decoder inputs
        target = data[1]

        predictions = self([encoder_inputs, decoder_inputs], training=False)
        loss = self.compiled_loss(target, predictions, regularization_losses=self.losses)
        self.compiled_metrics.update_state(target, predictions)

        return {m.name: m.result() for m in self.metrics}


def create_autoencoder(input_dim, time_steps, layer_dims, num_layers, dropout):
    keras.saving.get_custom_objects().clear()
    encoder_inputs = Input(shape=(time_steps, input_dim))
    decoder_inputs = Input(shape=(time_steps, input_dim))
    autoencoder = LSTMAutoEncoder(input_dim, time_steps, layer_dims, num_layers, dropout)
    outputs = autoencoder([encoder_inputs, decoder_inputs])
    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=outputs)
    model.compile(optimizer=Adam(), loss=CustomL2Loss(), metrics=['accuracy'])  #for prints: ", run_eagerly=True";
    return model


def test_lstm_autoencoder(time_steps, layer_dims, dropout, batch_size, epochs, remove_timestamps, directories, single_sensor_name=None, model_file_path=None):
    #scaler = MinMaxScaler(feature_range=(0, 1))  #Scales the data to a fixed range, typically [0, 1].
    #scaler = StandardScaler()           #Scales the data to have a mean of 0 and a standard deviation of 1.
    scaler = MaxAbsScaler()  #Scales each feature by its maximum absolute value, so that each feature is in the range [-1, 1]. #todo: best performance so far
    #scaler = None

    all_file_pairs = get_matching_file_pairs_from_directories(directories, single_sensor_name)
    print("all_file_pairs: " + str(all_file_pairs))

    for file_pair in all_file_pairs:
        data_with_time_diffs, true_labels_list = get_normalized_data_and_labels(file_pair, scaler, remove_timestamps)
        #plot_data(data_with_time_diffs[0])
        #if list is empty due to excluded csv file
        if not data_with_time_diffs:
            continue

        print("now reshaping")

        data_with_time_diffs = reshape_data_for_autoencoder_lstm(data_with_time_diffs, time_steps, 0)
        true_labels_list = reshape_data_for_autoencoder_lstm(true_labels_list, time_steps, 0)

        [print("reshaped data shape: \n" + str(data.shape)) for data in data_with_time_diffs]
        # print("reshaped data type: \n" + str(type(data_with_time_diffs[1])))
        # print("len: \n" + str(len(data_with_time_diffs)))
        # print("reshaped data: \n" + str(data_with_time_diffs))
        # print("reshaped labels: \n" + str(true_labels_list))
        # print("test: \n" + str(true_labels_list[0][3]))


        #shuffle data
        #data_with_time_diffs[0], true_labels_list[0] = shuffle_data(data_with_time_diffs[0], true_labels_list[0])
        #data_with_time_diffs[1], true_labels_list[1] = shuffle_data(data_with_time_diffs[1], true_labels_list[1])
        #X_sN, X_vN1, X_sN_labels, X_vN1_labels = train_test_split(data_with_time_diffs[0], true_labels_list[0], test_size=0.30, shuffle=False)
        #_, _, _, X_tN = split_data_sequence_into_datasets(data_with_time_diffs[1], 0.0, 0.0, 0.0, 1.0)
        #X_sN, X_vN1, X_vN2, _ = split_data_sequence_into_datasets(data_with_time_diffs[0], 0.8, 0.2, 0.0, 0.0)
        X_sN = data_with_time_diffs[0]

        # TODO: shuffle=True is entirely experimental. X_val is used for MLE and Fbeta so I do not think it matters but im not sure
        X_vN1, X_vN2, X_vN1_labels, X_vN2_labels = train_test_split(data_with_time_diffs[1], true_labels_list[1], test_size=0.50, shuffle=True)
        X_vNA = data_with_time_diffs[2]
        X_tN = data_with_time_diffs[3]
        true_labels_list = transform_true_labels_to_window_size(true_labels_list)
        print("true label list shape: \n" + str(true_labels_list[2].shape))
        #TODO: Shuffling is likely entirely unnecessary/bad here since it destroys the Overlapping window functionality

        input_dim = X_sN.shape[2]
        if model_file_path is None:
            model = create_autoencoder(input_dim, time_steps, layer_dims, len(layer_dims), dropout)
            model.summary()
            early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=60, restore_best_weights=True)

            #if X_sN doesnt get flipped in create_autoencoder because tensorflow hates me then I need to add it here!!!
            model.fit([X_sN, np.flip(X_sN, axis=1)], X_sN, epochs=epochs, batch_size=batch_size,
                      validation_data=([X_vN1, np.flip(X_vN1, axis=1)], X_vN1), verbose=1, callbacks=[early_stopping])

            model_name = "./models/LSTM_autoencoder_decoder_" + str(file_pair[0][file_pair[0].rfind("\\") + 1:].rstrip(".csv")) + "_timesteps" + str(time_steps) + "_layers"
            for layer in layer_dims:
                model_name = model_name + "_" + str(layer)

            print("Saved model to: " + str(model_name) + ".keras")
            model.save(model_name + ".keras")
        else:
            model = load_model(model_file_path, custom_objects={"LSTMAutoEncoder": LSTMAutoEncoder}, compile=True)

        #autoencoder_predict_and_calculate_error(model, X_tN, true_labels_list[3], 1, len(X_tN), scaler)
        # print("new: \n" + str(true_labels_list[2]))
        # print(X_tN.shape)

        X_vN1_error_vecs = np.asarray(calculate_rec_error_vecs(model, X_vN1, scaler))

        #todo: the way i flatten it might be incorrect
        flattened_X_vN1_error_vecs = X_vN1_error_vecs.reshape(X_vN1_error_vecs.shape[0], X_vN1_error_vecs.shape[1] * X_vN1_error_vecs.shape[2])
        print(X_vN1_error_vecs[0])
        print(flattened_X_vN1_error_vecs[0])

        mu, sigma = estimate_normal_error_distribution(flattened_X_vN1_error_vecs)
        X_vNA_error_vecs = calculate_rec_error_vecs(model, X_vNA, scaler)
        val_anomaly_scores = compute_anomaly_score(X_vNA_error_vecs, mu, sigma) #todo: X_vN2 needs to be appended somewhere here and put into calc febeta as well

        print("anom score shape: " + str(val_anomaly_scores.shape))
        print("true labels shape: " + str(true_labels_list[2].flatten().shape))

        flattened_X_vNA_labels = true_labels_list[2].flatten().tolist()
        for index, label in enumerate(flattened_X_vNA_labels):
            if flattened_X_vNA_labels[index] == 1 or (flattened_X_vNA_labels[index] == 1 and flattened_X_vNA_labels[index+1] == 0 and index <= len(flattened_X_vNA_labels) - 1):
                # print("index of anom: " + str(index) + " | " + "anomaly score: " + str(anomaly_scores[index]))
                # print("previous/following score: " + str(anomaly_scores[index-1]) + ", " + str(anomaly_scores[index+1]))
                print("index of anom: " + str(index) + " | " + "anomaly score: " + str(val_anomaly_scores[index]))
                print("previous score: " + str(val_anomaly_scores[index - 1]))
                print("15 following scores: " + str(val_anomaly_scores[index:index+15]))




        best_anomaly_threshold, best_fbeta = find_optimal_threshold(val_anomaly_scores, true_labels_list[2].flatten(), 0.9) #todo: optimal value of beta seems to be application knowledge
        print("Anomaly scores: \n" + str(val_anomaly_scores.tolist()))
        print("Best anomaly threshold: " + str(best_anomaly_threshold))

        plot_data_over_threshold(val_anomaly_scores, true_labels_list[2], best_anomaly_threshold, file_pair[0])

        print("\n---------------------------------------------------------------------------------\nDetection rate for valildation data: \n")
        print("detection with fbeta threshold: \n")
        calculate_detection_rate(val_anomaly_scores, true_labels_list[2], best_anomaly_threshold)
        print("detection with hack threshold: \n")
        average_anomaly_score_of_actual_anomalies = np.mean(val_anomaly_scores[true_labels_list[2].flatten() == 1])
        print("hack threshold: " + str(average_anomaly_score_of_actual_anomalies))
        calculate_detection_rate(val_anomaly_scores, true_labels_list[2], average_anomaly_score_of_actual_anomalies)
        print("\n---------------------------------------------------------------------------------\n")

        X_tN_error_vecs = calculate_rec_error_vecs(model, X_tN, scaler)
        test_anomaly_scores = compute_anomaly_score(X_tN_error_vecs, mu, sigma)

        print("\n---------------------------------------------------------------------------------\nDetection rate for Test data: \n")
        print("detection with fbeta threshold: \n")
        calculate_detection_rate(test_anomaly_scores, true_labels_list[3], best_anomaly_threshold)
        print("detection with hack threshold: \n")
        average_anomaly_score_of_actual_anomalies = np.mean(test_anomaly_scores[true_labels_list[3].flatten() == 1])
        print("hack threshold: " + str(average_anomaly_score_of_actual_anomalies))
        calculate_detection_rate(test_anomaly_scores, true_labels_list[3], average_anomaly_score_of_actual_anomalies)
        print("\n---------------------------------------------------------------------------------\n")

        plot_data_over_threshold(test_anomaly_scores, true_labels_list[3], best_anomaly_threshold, file_pair[3])

        X_vNAll_error_vecs = calculate_rec_error_vecs(model, data_with_time_diffs[1], scaler)
        X_vNAll_anomaly_scores = compute_anomaly_score(X_vNAll_error_vecs, mu, sigma)

        plot_data_over_threshold(X_vNAll_anomaly_scores, true_labels_list[1], best_anomaly_threshold, file_pair[1])

        # X_tN_error_vecs = calculate_rec_error_vecs(model, X_tN, scaler)
        # anomaly_scores = compute_anomaly_score(X_tN_error_vecs, mu, sigma)


class CustomL2Loss(Loss):
    def get_config(self):
        base_config = super().get_config()
        return base_config

    def call(self, y_true, y_pred):
        #print("y_true: \n" + str(y_true))
        #print("y_pred: \n" + str(y_pred))
        diff = y_true - y_pred
        l2_norm = tf.norm(diff, ord='euclidean', axis=-1, )
        #print("diff: \n" + str(diff))
        #print("l2_norm: \n" + str(l2_norm))
        #print("l2_norm after reduce_sum: \n" + str(tf.reduce_sum(l2_norm)))

        #return tf.reduce_sum(l2_norm)   #todo: reduce_sum or reduce_mean?
        return tf.reduce_mean(l2_norm)


def calculate_rec_error_vecs(model, X_vN1, scaler):
    error_vecs = []
    for i in range(len(X_vN1)):
        current_sequence = X_vN1[i].reshape((1, X_vN1[i].shape[0], X_vN1[i].shape[1]))
        predicted_sequence = model.predict([current_sequence, np.flip(current_sequence, axis=1)], verbose=0)
        current_sequence = reverse_normalization(np.squeeze(current_sequence, axis=0), scaler)
        predicted_sequence = reverse_normalization(np.squeeze(predicted_sequence, axis=0), scaler)
        error_vecs.append(np.subtract(current_sequence, predicted_sequence)) #took np.absolute() out because i don't think it is useful here
    return np.asarray(error_vecs)


    #todo: I think either saving or loading the model is not saving/loading the weights or internal states correctly....FUCKKKKKKK
    #todo: I think either saving or loading the model is not saving/loading the weights or internal states correctly....FUCKKKKKKK
    #todo: I think either saving or loading the model is not saving/loading the weights or internal states correctly....FUCKKKKKKK


    #todo:
    #      //add true_labels to all data
    #      //implement overlapping window separation of data
    #      //finish anomaly score and fbeta score using X_vN2 and X_vA
    #      //consider switch to loss_function='mse' and compare performance --> mse was worse (very limited sample size so grain of salt)
    #      find way to create anomalous datasets
    #      //ask Sebastian or Tamara what typical anomalies (might) look like
    #      find optimal hyperparameters for each sensor
    #      find and implement next algorithm
    #      //Schleif meeting


# possible solutions:
#       1. delete every entry of the data array that contains an empty cell ---> THIS DOESNT FKING WORK BECAUSE I WILL LOSE LIKE 9/10th OF THE DATA THAT HAVE SHORTER MEASURING INTERVALLS AND THE VALUES WONT EVEN REALLY
#          BE CORRELATED
#       2. find sensor with least data entries; downsample all other sensor data accordingly before adding them together
#       ----> I think 1. and 2. are equivalent aaaaaaaaaaaaaaaaaaaahhhhhh
#       3. average empty cells out or sth, idk that sounds retarded AND complicated, my favorite
#       3. Kill myself

def estimate_normal_error_distribution(error_vecs):
    # MLE for the mean (µ)
    mu = np.mean(error_vecs, axis=0)
    # MLE for the covariance matrix (Σ)
    sigma = np.cov(error_vecs, rowvar=False)
    return mu, sigma


#todo:
# It might be better to somehow compare each single value in the vector with a treshold instead of the entire one
# however in the paper they explicitly use the whole vector
# ----> alternatively I could modify the error function to split the predicted/choosen sequnece into four and then use each subvector, will have to see
# sigma is a matrix here. I'm not sure if this is correct but try it for now


def compute_anomaly_score(error_vecs, mu, sigma):    #todo: Need to calculate anomaly score for each error_vec; not sure if this does this but otherwise just do it with a loop
    # print("Mu: " + str(mu))
    # print("Sigma: " + str(sigma))
    # print("Error vec: " + str(error_vecs))
    # print(error_vecs.shape)
    #
    # flattened_error_vecs2 = error_vecs.reshape(error_vecs.shape[0], error_vecs.shape[1] * error_vecs.shape[2])
    # print("Mu: \n" + str(mu))
    # print("Flattened error vecs2: \n", flattened_error_vecs2)
    # print("Mu shape: " + str(mu.shape))
    # print("Flattened error vecs2: ", flattened_error_vecs2.shape)


    scores = []
    inv_sigma = np.linalg.inv(sigma)

    error_vecs = error_vecs.reshape(error_vecs.shape[0], error_vecs.shape[1] * error_vecs.shape[2]) #todo: this might be totally incorrect but idrk rn
    for error in error_vecs:
        diff = np.subtract(error, mu)
        score = np.dot(np.dot(diff, inv_sigma), diff.T)
        #print("individual score: " + str(score))
        scores.append(score)  #todo: seems to work?
                                                      # anomaly_scores = np.einsum('ij,ij->i', diff @ np.linalg.inv(Sigma), diff) ---> bruh?
    #scores = np.einsum('ij,jk,ik->i', diff, inv_sigma, diff)
    test_score_np_arr = np.asarray(scores)
    #print("scores " + str(test_score_np_arr))
    #print("scores shape: " + str(test_score_np_arr.shape))
    return test_score_np_arr


#todo: is supposed to use "from sklearn.metrics import precision_recall_curve, fbeta_score" but only uses "fbeta_scores". ChatGPT is currently down so look
#todo: into it later
def find_optimal_threshold(anomaly_scores, true_labels, beta):
    precision, recall, thresholds = precision_recall_curve(y_true=true_labels, y_score=anomaly_scores)
    fbeta_scores = (1 + beta ** 2) * (precision * recall) / ( (beta ** 2) * precision + recall)
    fbeta_scores = np.nan_to_num(fbeta_scores, nan=-np.inf, posinf=-np.inf)

    print("fbeta scores: \n" + str(fbeta_scores))
    print("best fbeta: " + str(np.max(fbeta_scores)))
    best_index = np.argmax(fbeta_scores)
    if best_index >= len(thresholds):
        best_index = len(thresholds) - 1

    print("Thresholds: " + str(thresholds))
    print("Best threshold: " + str(thresholds[best_index]))
    print(str(thresholds))
    best_threshold = thresholds[best_index]
    best_fbeta = fbeta_scores[best_index]

    plt.figure()
    plt.plot(thresholds, precision[:-1], 'b-', label='Precision')
    plt.plot(thresholds, recall[:-1], 'g-', label='Recall')
    plt.xlabel('Threshold')
    plt.ylabel('Precision/Recall')
    plt.legend(loc='best')
    plt.title('Precision-Recall Curve')
    plt.show()

    return best_threshold, best_fbeta


def calculate_detection_rate(anomaly_scores, true_labels, threshold):
    true_labels = true_labels.flatten()
    total_anoms = np.sum(true_labels == 1)
    anomaly_scores_of_anoms = []
    avg_anomaly_score_of_anoms = 0
    avg_anomaly_score_of_normals = 0

    true_pos = 0
    false_pos = 0
    false_neg = 0
    true_neg = 0

    if anomaly_scores.shape != true_labels.shape:
        print("anomaly scores shape is not equal to true labels shape")
        return

    for i in range(len(true_labels)):
        if true_labels[i] == 1:
            anomaly_scores_of_anoms.append(anomaly_scores[i])
            avg_anomaly_score_of_anoms += anomaly_scores[i]
            if anomaly_scores[i] >= threshold:
                true_pos += 1
            if anomaly_scores[i] <= threshold:
                false_neg += 1
        elif true_labels[i] == 0:
            avg_anomaly_score_of_normals += anomaly_scores[i]
            if anomaly_scores[i] >= threshold:
                false_pos += 1
            if anomaly_scores[i] <= threshold:
                true_neg += 1

    avg_anomaly_score_of_anoms = avg_anomaly_score_of_anoms / total_anoms
    avg_anomaly_score_of_normals = avg_anomaly_score_of_normals / anomaly_scores.shape[0] - total_anoms
    print("true positives: " + str(true_pos))
    print("false positives: " + str(false_pos))
    print("true negatives: " + str(true_neg))
    print("false negatives: " + str(false_neg))

    print("total anomalies: " + str(total_anoms))
    print("average anomaly score of all anomalies: " + str(avg_anomaly_score_of_anoms))
    print("scores of all anomalies: " + str(anomaly_scores_of_anoms))
    print("average score of normal data windows: " + str(avg_anomaly_score_of_normals))


def plot_data_over_threshold(anomaly_scores, true_labels, threshold, file_name):
    # """
    #    Plots the data points under the threshold as blue dots, the ones over the threshold as red crosses,
    #    and the threshold itself as a red dotted line.
    #
    #    Parameters:
    #    - data: 1D numpy array of data points
    #    - threshold: A scalar value representing the threshold
    #    - anomaly_labels: 1D numpy array of 0s and 1s indicating actual anomalies (1 for anomaly, 0 for normal)
    #    """
    # plt.figure(figsize=(10, 6))
    # plt.plot(np.where(data <= threshold)[0], data[data <= threshold], 'bo', label='Normal')  # Blue dots for "Normal"
    # plt.plot(np.where(data > threshold)[0], data[data > threshold], 'rx', label='Anomaly')  # Red crosses for "Anomaly"
    # plt.title('Prediction Result')
    # plt.xlabel('Index')
    # plt.ylabel('Anomaly Score')
    # plt.legend()
    # plt.show()

    # Plotting
    plt.figure(figsize=(12, 6))
    for i, (score, label) in enumerate(zip(anomaly_scores, true_labels)):
        color = 'red' if label == 1 else 'blue'
        marker = 'x' if label == 1 else 'o'
        plt.scatter(i, score, color=color, marker=marker, s=100, label='Anomaly' if label == 1 else 'Normal')

    # if threshold * 1.5 > anomaly_scores.max() * 1.5:
    #     plt.ylim(0, threshold * 1.5)
    # else:
    #     plt.ylim(0, anomaly_scores.max() * 1.5)

    plt.ylim(0, threshold * 4)


    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Normal', markerfacecolor='blue', markersize=10),
        Line2D([0], [0], marker='x', color='w', label='Anomaly', markerfacecolor='red', markersize=10)
    ]

    plt.axhline(y=threshold, color='red', linestyle='--', label='Anomaly Threshold', linewidth=2)  # Red dotted line for the threshold
    plt.legend(handles=legend_elements, loc='upper left')
    plt.title('Anomaly Scores with True Labels of' + file_name)
    plt.xlabel('Index')
    plt.ylabel('Anomaly Score')
    # plt.legend(['Normal', 'Anomaly'], loc='upper left')
    plt.show()

    plt.savefig(file_name + '.png')  # Save the plot to a file

    print("Number of elements above threshold: " + str(np.sum(anomaly_scores > threshold)))
    print("Number of elements below threshold: " + str(np.sum(anomaly_scores <= threshold)))

