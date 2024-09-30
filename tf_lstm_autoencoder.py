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
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler, QuantileTransformer
#from tensorflow import keras
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from keras import Loss

from data_processing import reshape_data_for_autoencoder_lstm, normalize_data, \
    split_data_sequence_into_datasets, reverse_normalization, csv_file_to_nparr, \
    get_normalized_data_and_labels, get_matching_file_pairs_from_directories
from utils import autoencoder_predict_and_calculate_error


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
        for i in layer_dims:
            self.encoder_lstm.append(
                LSTM(i, activation='relu', return_sequences=True, return_state=True, dropout=dropout))

        # Decoder
        for i in reversed(layer_dims):
            self.decoder_lstm.append(
                LSTM(i, activation='relu', return_sequences=True, return_state=True, dropout=dropout))

        self.decoder_dense = Dense(input_dim)

    def call(self, inputs, training=False):
        #if training is None:
        print("Training argument is " + str(training))

        encoder_inputs, decoder_inputs = inputs
        #x = encoder_inputs
        #print("This may be huge: " + str(decoder_inputs.shape))
        #print("This may be huge2: " + str(encoder_inputs))

        #encoder_states = [tf.zeros((tf.shape(encoder_inputs)[0], self.layer_dims[0])), tf.zeros((tf.shape(encoder_inputs)[0], self.layer_dims[0]))]

        #I'm still not sure if this is correct. The previous function looked like this:
        #    # for lstm_layer in self.encoder_lstm[:-1]:   #all but last layer
        #         #     x, _, _ = lstm_layer(x)
        #         # encoder_outputs, state_h, state_c = self.encoder_lstm[-1](x)  # output of last layer

        #Should I pass the states inbetween layers or keep each layer's state in its corresponding layer and then only pass the states of the last layer to the decoder?

        encoder_states_empty = True
        encoder_states = [None, None]
        for t in range(self.time_steps):
            x = encoder_inputs[:, t:t + 1, :]  # Select one time step
            #print("here3: " + str(x.shape))
            for lstm_layer in self.encoder_lstm:
                #if encoder_states_empty:
                #    x, state_h, state_c = lstm_layer(x)
                #    encoder_states_empty = False
                #else:
                x, state_h, state_c = lstm_layer(x) # todo: REMOVED; EXPERIMENTAL!!!, initial_state=encoder_states)
                encoder_states = [state_h, state_c]

        #encoder_states = [state_h, state_c]
        all_outputs = []

        #todo: EXPERIMANTAL!!!!!!
        if training:
            dec_input = decoder_inputs[:, 0:1, :]   # Use first value of the reversed sequence as initial input for the decoder
        else:
            #print("here: " + str(decoder_inputs[:, 0:1, :].shape))
            #dec_input = self.decoder_dense(state_h)
            dec_input = tf.expand_dims(self.decoder_dense(encoder_states[0]), axis=1) # todo: changed state_h for encoder_state[0] EPERIMENTAL!!! # use last state of encoder as decoder input make predict first prediction of sequence
            #print("here2: " + str(dec_input.shape))

        all_outputs.append(dec_input)

        for t in range(1, self.time_steps):
            #print("iteration: " + str(t))

            for lstm_layer in self.decoder_lstm:
                dec_output, state_h, state_c = lstm_layer(dec_input, initial_state=encoder_states)
                #dec_output = self.decoder_dense(dec_output)
                dec_output = tf.expand_dims(self.decoder_dense(state_h), axis=1)

                #todo: Test this
                if training:
                    dec_input = decoder_inputs[:, 0:1, :]
                else:
                    dec_input = dec_output

                encoder_states = [state_h, state_c]  # todo: not sure if state_c should be passed
                all_outputs.append(dec_output)
            #outputs = self.decoder_dense(state_h)

            #if training:
            #    inputs = decoder_inputs[:, t:t + 1, :]  # Use the ground truth as the next input
            #else:
            #    inputs = outputs  # Use the output as the next input

        #print("All outputs: \n" + str(all_outputs))

        #reverse decoder output since decoder predicts target data in reverse
        all_outputs = all_outputs[::-1]
        #print("Reversed outputs: \n" + str(all_outputs))

        dec_output = tf.concat(all_outputs, axis=1)
        return dec_output

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

    # def test_step(self, data):
    #     encoder_inputs, decoder_inputs = data[0]  # Unpacking encoder and decoder inputs
    #     target = data[1]
    #
    #     predictions = self([encoder_inputs, decoder_inputs], training=False)
    #     loss = self.compiled_loss(target, predictions, regularization_losses=self.losses)
    #     self.compiled_metrics.update_state(target, predictions)
    #
    #     return {m.name: m.result() for m in self.metrics}


def create_autoencoder(input_dim, time_steps, layer_dims, num_layers, dropout):
    keras.saving.get_custom_objects().clear()
    encoder_inputs = Input(shape=(time_steps, input_dim))
    decoder_inputs = Input(shape=(time_steps, input_dim))
    autoencoder = LSTMAutoEncoder(input_dim, time_steps, layer_dims, num_layers, dropout)
    outputs = autoencoder([encoder_inputs, decoder_inputs])
    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=outputs)
    #todo: Added clipnorm=1.0; experimental!!!
    model.compile(optimizer=Adam(clipnorm=1.0), loss=CustomL2Loss(), metrics=['mean_squared_error'])  # CustomL2Loss() ; metrics=['accuracy'] ; for prints: ", run_eagerly=True";
    return model


class CustomL2Loss(Loss):
    def get_config(self):
        base_config = super().get_config()
        return base_config

    def call(self, y_true, y_pred):
        #print("CustomL2Loss y_true: " + str(y_true.shape))
        #print("CustomL2Loss y_pred: " + str(y_pred.shape))

        diff = y_true - y_pred
        squared_diff = tf.square(diff + 1e-10) #may halp with over/underflow

        # tf.print("y_true: " + str(y_true.numpy()))
        # tf.print("y_pred: " + str(y_pred.numpy()))
        # tf.print("squared_diff: " + str(squared_diff.numpy()))

        return tf.reduce_mean(tf.reduce_sum(squared_diff, axis=-1))


        #print("y_true: \n" + str(y_true))
        #print("y_pred: \n" + str(y_pred))
        #diff = y_true - y_pred
        #l2_norm = tf.norm(diff, ord='euclidean', axis=-1, )
        #print("diff: \n" + str(diff))
        #print("l2_norm: \n" + str(l2_norm))
        #print("l2_norm after reduce_sum: \n" + str(tf.reduce_sum(l2_norm)))

        #return tf.reduce_sum(l2_norm)   #todo: reduce_sum or reduce_mean?
        #return tf.reduce_mean(l2_norm)


def test_lstm_autoencoder(time_steps, layer_dims, dropout, batch_size, epochs, beta, calc_anom_score_flag, remove_timestamps, directories,
                          single_sensor_name=None, model_file_path=None):
    scaler = MinMaxScaler(feature_range=(0, 1))  #Scales the data to a fixed range, typically [0, 1].
    #scaler = StandardScaler()                      #Scales the data to have a mean of 0 and a standard deviation of 1.
    #scaler = MaxAbsScaler()                         #Scales each feature by its maximum absolute value, so that each feature is in the range [0, 1] or [-1, 0] or [-1, 1]
    #scaler = QuantileTransformer(output_distribution='normal')
    #scaler = None

    all_file_pairs = get_matching_file_pairs_from_directories(directories, single_sensor_name)
    print("all_file_pairs: " + str(all_file_pairs))

    for file_pair in all_file_pairs:
        data_with_time_diffs, true_labels_list = get_normalized_data_and_labels(file_pair, scaler, 10, remove_timestamps)
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
        #X_vN1, X_vN2, X_vN1_labels, X_vN2_labels = train_test_split(data_with_time_diffs[1], true_labels_list[1], test_size=0.50, shuffle=False)
        X_vN1 = data_with_time_diffs[1]
        X_vNA = data_with_time_diffs[2]
        X_tN = data_with_time_diffs[3]
        #true_labels_list = transform_true_labels_to_window_size(true_labels_list)
        print("true label list shape: \n" + str(true_labels_list[2].shape))
        #TODO: Shuffling is likely entirely unnecessary/bad here since it destroys the Overlapping window functionality

        model_name = "./models/LSTM_autoencoder_decoder_" + str(
            file_pair[0][file_pair[0].rfind("\\") + 1:].rstrip(".csv")) + "_timesteps" + str(time_steps) + "_layers"
        for layer in layer_dims:
            model_name = model_name + "_" + str(layer)

        input_dim = X_sN.shape[2]
        if model_file_path is None:
            model = create_autoencoder(input_dim, time_steps, layer_dims, len(layer_dims), dropout)
            model.summary()
            early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=20,
                                           restore_best_weights=True)

            #if X_sN doesnt get flipped in create_autoencoder because tensorflow hates me then I need to add it here!!!
            model.fit([X_sN, np.flip(X_sN, axis=1)], X_sN, epochs=epochs, batch_size=batch_size,
                      validation_data=([X_vN1, np.flip(X_vN1, axis=1)], X_vN1), verbose=1, callbacks=[early_stopping])

            # todo: test if it works better using X_vN_all instead of X_vN1

            model.save(model_name + ".keras")
            print("Saved model to: " + str(model_name) + ".keras")
        else:
            model = load_model(model_file_path, custom_objects={"LSTMAutoEncoder": LSTMAutoEncoder}, compile=True)

        X_vN_error_vecs = np.asarray(calculate_rec_error_vecs(model, data_with_time_diffs[1], scaler))
        #print(X_vN_error_vecs[0:10])

        print("started calculating of mu and sigma")
        mu, sigma = calculate_mle_mu_sigma(X_vN_error_vecs) #todo: CHANGED!!!
        print("mu: " + str(mu))
        print("sigma: " + str(sigma))
        X_vNA_error_vecs = calculate_rec_error_vecs(model, X_vNA, scaler)
        X_vNA_anomaly_scores = compute_anomaly_score(X_vNA_error_vecs, mu, sigma)

        print("anom score shape: " + str(X_vNA_anomaly_scores.shape))
        print("true labels shape: " + str(true_labels_list[2].flatten().shape))

        # optimal value of beta seems to be application knowledge
        if calc_anom_score_flag:
            best_anomaly_threshold, best_fbeta = find_optimal_threshold(X_vNA_anomaly_scores, true_labels_list[2], beta)
            write_threshold_to_file(model_name.replace("./models/", ""), best_anomaly_threshold)
            print("recalculated anomaly threshold")
        else:
            best_anomaly_threshold = read_threshold_from_file(model_name.replace("./models/", ""))
            print("read anomaly threshold from file")

        #print("Anomaly scores: \n" + str(X_vNA_anomaly_scores.tolist()))
        print("Best anomaly threshold: " + str(best_anomaly_threshold))

        X_tN_error_vecs = calculate_rec_error_vecs(model, X_tN, scaler)
        X_tN_anomaly_scores = compute_anomaly_score(X_tN_error_vecs, mu, sigma)

        X_vN_anomaly_scores = compute_anomaly_score(X_vN_error_vecs, mu, sigma)

        calculate_and_plot_detection_rate("X_vN", file_pair[1], X_vN_anomaly_scores, true_labels_list[1], best_anomaly_threshold, time_steps)
        calculate_and_plot_detection_rate("X_tN", file_pair[3], X_tN_anomaly_scores, true_labels_list[3], best_anomaly_threshold, time_steps)
        calculate_and_plot_detection_rate("X_vNA", file_pair[2], X_vNA_anomaly_scores, true_labels_list[2], best_anomaly_threshold, time_steps)


def calculate_rec_error_vecs(model, reshpaed_input, scaler):
    error_vecs = []
    for i in range(len(reshpaed_input)):
        #print("shape before reshape: " + str(reshpaed_input[i].shape))
        current_sequence = reshpaed_input[i].reshape((1, reshpaed_input[i].shape[0], reshpaed_input[i].shape[1]))
        #print("shape after reshape: " + str(current_sequence.shape))

        predicted_sequence = model.predict([current_sequence, np.flip(current_sequence, axis=1)], verbose=0)

        current_sequence = reverse_normalization(np.squeeze(current_sequence, axis=0), scaler)
        predicted_sequence = reverse_normalization(np.squeeze(predicted_sequence, axis=0), scaler)

        #print("chosen seq: " + str(current_sequence))
        #print("predicted seq: " + str(predicted_sequence))
        # print("difference: " + str(np.subtract(current_sequence, predicted_sequence)))
        error_vecs.append(np.absolute(np.subtract(current_sequence, predicted_sequence)))  #todo: old: took np.absolute()out because i don't think it is useful here
    return np.asarray(error_vecs)

    #todo:
    #      //add true_labels to all data
    #      //implement overlapping window separation of data
    #      //finish anomaly score and fbeta score using X_vN2 and X_vA
    #      //consider switch to loss_function='mse' and compare performance --> mse was worse (very limited sample size so grain of salt)
    #      //find way to create anomalous datasets
    #      //ask Sebastian or Tamara what typical anomalies (might) look like
    #      find optimal hyperparameters for each sensor
    #      //find and implement next algorithm
    #      //Schleif meeting


def calculate_mle_mu_sigma(error_vecs):
    reshaped_error_vecs = error_vecs.reshape(-1, error_vecs.shape[-1])
    mu = np.mean(reshaped_error_vecs, axis=0)           #todo: changed calculation of mu!!!!!!
    sigma = np.cov(reshaped_error_vecs, rowvar=False)  # Covariance matrix

    # # MLE for the mean (µ)
    # mu = np.mean(error_vecs, axis=0)
    # # MLE for the covariance matrix (Σ)
    # sigma = np.cov(error_vecs, rowvar=False)
    # return mu, sigma

    return mu, sigma


#todo:
# (((((It might be better to somehow compare each single value in the vector with a treshold instead of the entire one
# however in the paper they explicitly use the whole vector
# ----> alternatively I could modify the error function to split the predicted/choosen sequnece into four and then use each subvector, will have to see
# sigma is a matrix here. I'm not sure if this is correct but try it for now)))))
# -----------------> pretty sure i fucked this up

def compute_anomaly_score(error_vecs, mu, sigma):  #todo: Need to calculate anomaly score for each error_vec; not sure if this does this but otherwise just do it with a loop
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

    inv_sigma = np.linalg.inv(sigma)
    #error_vecs = error_vecs.reshape(error_vecs.shape[0], error_vecs.shape[1] * error_vecs.shape[2])  #todo: this might be totally incorrect but idrk rn
    print("here mfer: " + str(error_vecs.shape))

    scores_all_windows = []
    for sequence in error_vecs:
        scores_of_window = []
        for data_point in sequence:
            #print("data_point: " + str(data_point))
            #print("shape of data_point (should just be 4 attributes): " + str(data_point.shape))

            diff = np.subtract(data_point, mu)
            score = np.dot(np.dot(diff, inv_sigma), diff.T)
            #print("individual score: " + str(score))
            scores_of_window.append(score)  #todo: seems to work?
        scores_all_windows.append(scores_of_window)

    test_score_np_arr = np.asarray(scores_all_windows)
    #print("scores " + str(test_score_np_arr))
    #print("scores shape: " + str(test_score_np_arr.shape))
    return test_score_np_arr

def find_optimal_threshold(anomaly_scores, true_labels, beta):
    anomaly_scores = anomaly_scores.flatten()
    true_labels = np.squeeze(true_labels).flatten()

    precision, recall, thresholds = precision_recall_curve(y_true=true_labels, y_score=anomaly_scores)
    fbeta_scores = (1 + beta ** 2) * (precision * recall) / ((beta ** 2) * precision + recall)
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
    true_labels = np.squeeze(true_labels).flatten()
    anomaly_scores = anomaly_scores.flatten()

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
            else:
                false_neg += 1
        elif true_labels[i] == 0:
            avg_anomaly_score_of_normals += anomaly_scores[i]
            if anomaly_scores[i] >= threshold:
                false_pos += 1
            else:
                true_neg += 1

    if total_anoms > 0:
        avg_anomaly_score_of_anoms = avg_anomaly_score_of_anoms / total_anoms
    else:
        avg_anomaly_score_of_anoms = 0

    total_normals = len(true_labels) - total_anoms
    if total_normals > 0:
        avg_anomaly_score_of_normals = avg_anomaly_score_of_normals / total_normals
    else:
        avg_anomaly_score_of_normals = 0
    print("true positives: " + str(true_pos))
    print("false positives: " + str(false_pos))
    print("true negatives: " + str(true_neg))
    print("false negatives: " + str(false_neg))

    print("total anomalies: " + str(total_anoms))
    print("average anomaly score of all anomalies: " + str(avg_anomaly_score_of_anoms))
    #print("scores of all anomalies: " + str(anomaly_scores_of_anoms))
    print("average score of normal data windows: " + str(avg_anomaly_score_of_normals))


def calculate_and_plot_detection_rate(dataset_name, file_path, anomaly_scores, true_labels, best_anomaly_threshold, time_steps):
    print(
        "\n---------------------------------------------------------------------------------\nDetection rate for " + dataset_name + ": \n")
    print("detection with fbeta threshold: \n")
    calculate_detection_rate(anomaly_scores, true_labels, best_anomaly_threshold)
    median_anomaly_score_of_actual_anomalies = np.median(anomaly_scores.flatten()[np.squeeze(true_labels).flatten() == 1])
    print("median anomaly score of anomalies: " + str(median_anomaly_score_of_actual_anomalies))
    median_anomaly_score_of_normals = np.median(anomaly_scores.flatten()[np.squeeze(true_labels).flatten() == 0])
    print("median anomaly score of normals: " + str(median_anomaly_score_of_normals))
    print("\n---------------------------------------------------------------------------------\n")
    plot_data_over_threshold(anomaly_scores, true_labels, best_anomaly_threshold, file_path, time_steps)


def plot_data_over_threshold(anomaly_scores, true_labels, threshold, file_name, time_steps):
    # plt.figure(figsize=(10, 6))
    # plt.plot(np.where(data <= threshold)[0], data[data <= threshold], 'bo', label='Normal')  # Blue dots for "Normal"
    # plt.plot(np.where(data > threshold)[0], data[data > threshold], 'rx', label='Anomaly')  # Red crosses for "Anomaly"
    # plt.title('Prediction Result')
    # plt.xlabel('Index')
    # plt.ylabel('Anomaly Score')
    # plt.legend()
    # plt.show()

    """
    Plots the data points under the threshold as blue dots, the ones over the threshold as red crosses,
    and the threshold itself as a red dotted line.

    Parameters:
    - anomaly_scores: 2D numpy array (sequences x time steps) of anomaly scores
    - true_labels: 3D numpy array (sequences x time steps x 1) of true labels (1 for anomaly, 0 for normal)
    - threshold: A scalar value representing the threshold
    - file_name: Name of the file to save the plot
    """

    true_labels = np.squeeze(true_labels).flatten()
    anomaly_scores = anomaly_scores.flatten()

    if anomaly_scores.shape != true_labels.shape:
        print("Anomaly scores shape is not equal to true labels shape")
        return

    window_anomaly_scores = []
    window_labels = []

    for i in range(0, len(anomaly_scores), time_steps):
        window_scores = anomaly_scores[i:i + time_steps]
        window_labels_slice = true_labels[i:i + time_steps]

        if np.any(window_labels_slice == 1):
            #window_anomaly_scores.append(np.max(window_scores))
            if np.mean(window_scores > threshold) > 0.4:
                window_anomaly_scores.append(np.max(window_scores))
            else:
                window_anomaly_scores.append(np.mean(window_scores))
            window_labels.append(1)
        else:
            # Check if more than 30% of the anomaly scores in window are higher than threshold
            if np.mean(window_scores > threshold) > 0.4:
                window_anomaly_scores.append(np.max(window_scores))
            else:
                window_anomaly_scores.append(np.mean(window_scores))
            window_labels.append(0)

    for k in range(2):
        plt.figure(figsize=(12, 6))

        for idx, (score, label) in enumerate(zip(window_anomaly_scores, window_labels)):
            if label == 1:  # True anomaly (window)
                if score >= threshold:
                    color = 'red'  # Anomalous window above threshold
                else:
                    color = 'yellow'  # Anomalous window below threshold
                marker = 'x'
            else:  # Normal window
                if score >= threshold:
                    color = 'purple'  # Normal window above threshold
                else:
                    color = 'blue'  # Normal window below threshold
                marker = 'o'

            plt.scatter(idx, score, color=color, marker=marker, s=20, label='Anomaly' if label == 1 else 'Normal')

        if k == 0:
            plt.ylim(0, threshold * 1.1)
        else:
            plt.ylim(0, max(window_anomaly_scores) * 1.1)

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Normal', markerfacecolor='blue', markersize=10),
            Line2D([0], [0], marker='x', color='w', label='Anomaly', markerfacecolor='red', markersize=10)
        ]

        plt.axhline(y=threshold, color='red', linestyle='--', label='Anomaly Threshold', linewidth=2)

        plt.text(-0.05 * len(window_anomaly_scores), threshold, f'{threshold}', va='center', ha='right', color='red',
                 fontsize=12, fontweight='bold')

        plt.legend(handles=legend_elements, loc='upper left')
        plt.title('Anomaly Scores with True Labels of ' + file_name)
        plt.xlabel('Window Index')
        plt.ylabel('Anomaly Score')

        plt.savefig(file_name + str(k) + '.png')
        plt.show()

    print("Number of windows above threshold: " + str(np.sum(np.array(window_anomaly_scores) > threshold)))
    print("Number of windows below threshold: " + str(np.sum(np.array(window_anomaly_scores) <= threshold)))


def write_threshold_to_file(model_name, threshold):
    file_path = "./anomaly_thresholds.txt"
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        lines = []

    model_entry = f"{model_name}={threshold}\n"
    updated = False

    for i, line in enumerate(lines):
        if line.startswith(f"{model_name}="):
            lines[i] = model_entry
            updated = True
            break
    if not updated:
        lines.append(model_entry)

    with open(file_path, 'w') as file:
        file.writelines(lines)

def read_threshold_from_file(model_name):
    file_path = "./anomaly_thresholds.txt"
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print("No anomaly threshold file found.")
        return None

    for line in lines:
        if line.startswith(f"{model_name}="):
            try:
                threshold = float(line.split('=')[1].strip())
                return threshold
            except ValueError:
                print(f"Invalid threshold value for model '{model_name}' in file.")
                return None

    print(f"Model '{model_name}' not found in file.")
    return None