import shutil

import numpy as np
import tensorflow as tf
import keras
from keras import Loss
from keras.src.callbacks import EarlyStopping
from keras_tuner import RandomSearch
from scipy.linalg import inv
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import MaxAbsScaler
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from kerastuner import HyperModel

from data_processing import reshape_data_for_autoencoder_lstm, normalize_data, \
    split_data_sequence_into_datasets, reverse_normalization, directory_csv_files_to_dataframe_to_numpyArray
from utils import autoencoder_predict_and_calculate_error, get_matching_file_pairs_from_directory


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

        print("Und Keras so: Fick dich!")
        print(layer_dims)
        print(input_dim)
        print(time_steps)
        print(num_layers)
        print(dropout)

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
        print("Im going fucking feral: " + str(layer_dims))
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
        print("This may be huge: " + str(decoder_inputs.shape))
        print("This may be huge2: " + str(encoder_inputs))

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
        #reverse decoder output since it predicts target data in reverse
        all_outputs = all_outputs[::-1]  #todo: I think this is correct now??? Not 100% sure
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


# def create_autoencoder(input_dim, time_steps, layer_dims, num_layers, dropout):
#     keras.saving.get_custom_objects().clear()
#     keras.saving.get_custom_objects().update({"customL2Loss": CustomL2Loss})  # Register the custom loss
#     encoder_inputs = Input(shape=(time_steps, input_dim))
#     decoder_inputs = Input(shape=(time_steps, input_dim))
#     autoencoder = LSTMAutoEncoder(input_dim, time_steps, layer_dims, num_layers, dropout)
#     outputs = autoencoder([encoder_inputs, decoder_inputs])
#     model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=outputs)
#     model.compile(optimizer=Adam(), loss=CustomL2Loss(), metrics=['accuracy'])  #for prints: ", run_eagerly=True"
#     return model


class LSTMAutoEncoderHyperModel(HyperModel):
    def __init__(self, input_dim, time_steps):
        self.input_dim = input_dim
        self.time_steps = time_steps

    def build(self, hp):
        units = hp.Int('units', min_value=8, max_value=256, step=10)
        layer_amount = hp.Int('layer_amount', min_value=1, max_value=2, step=1)
        layer_dims = []
        for i in range(layer_amount):
            layer_dims.append(units)
        dropout = hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-1, sampling='LOG')

        encoder_inputs = Input(shape=(self.time_steps, self.input_dim))
        decoder_inputs = Input(shape=(self.time_steps, self.input_dim))

        autoencoder = LSTMAutoEncoder(self.input_dim, self.time_steps, layer_dims, len(layer_dims), dropout)
        outputs = autoencoder([encoder_inputs, decoder_inputs])

        model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss=CustomL2Loss(), metrics=['accuracy'])
        return model



@keras.saving.register_keras_serializable(package="my_package", name="CustomL2Loss")
class CustomL2Loss(Loss):
    def get_config(self):
        base_config = super().get_config()
        return base_config

    def call(self, y_true, y_pred):
        # print("y_true: \n" + str(y_true))
        # print("y_pred: \n" + str(y_pred))
        diff = y_true - y_pred
        l2_norm = tf.norm(diff, ord='euclidean', axis=-1, )
        # print("diff: \n" + str(diff))
        # print("l2_norm: \n" + str(l2_norm))
        # print("l2_norm after reduce_sum: \n" + str(tf.reduce_sum(l2_norm)))

        # return tf.reduce_sum(l2_norm)   #todo: reduce_sum or reduce_mean?
        return tf.reduce_mean(l2_norm)




def tune_lstm_autoencoder(time_steps, directories):
    shutil.rmtree('tuner_results')

    # ((((todo: data shuffling may be advisable (think i saw it in the other guys code) --> i dont think so but worth a try ))))

    #scaler = MinMaxScaler(feature_range=(-1, 1))  #Scales the data to a fixed range, typically [0, 1].
    #scaler = StandardScaler()           #Scales the data to have a mean of 0 and a standard deviation of 1.
    scaler = MaxAbsScaler()  #Scales each feature by its maximum absolute value, so that each feature is in the range [-1, 1]. #todo: best performance so far

    all_file_pairs = get_matching_file_pairs_from_directory(directories[0], directories[1])
    print("all_file_pairs: " + str(all_file_pairs))

    for file_pair in all_file_pairs:
        data_with_time_diffs = []
        true_labels_list = []  #specify whether a point is classified as anomaly
        print("Now training model on: " + str(file_pair[0][file_pair[0].rfind("\\") + 1:].rstrip(".csv")))

        #data is automatically scaled to relative timestamps
        for single_file in file_pair:
            data, true_labels = directory_csv_files_to_dataframe_to_numpyArray(single_file)
            if data is None:
                break
            print("unnormalized_data_with_time_diffs: \n" + str(data))
            normalized_data = normalize_data(data, scaler)
            print("normalized_data_with_time_diffs: \n" + str(normalized_data))
            data_with_time_diffs.append(normalized_data)
            true_labels_list.append(true_labels)

        #if list is empty due to exception csv
        if not data_with_time_diffs:
            continue

        data_with_time_diffs = reshape_data_for_autoencoder_lstm(data_with_time_diffs, time_steps)
        true_labels_list = reshape_data_for_autoencoder_lstm(true_labels_list, time_steps)

        #np.random.shuffle(data_with_time_diffs[1]) #todo: EXPERIMENTAL!!!!!!!!!!!!!!!!!!!!!!!!!!

        indices = np.arange(data_with_time_diffs[1].shape[0])
        np.random.shuffle(indices)
        data_with_time_diffs[1] = data_with_time_diffs[1][indices]
        true_labels_list[1] = true_labels_list[1][indices]

        X_sN, X_vN1, X_vN2, _ = split_data_sequence_into_datasets(data_with_time_diffs[0], 0.8, 0.2, 0.0, 0.0)
        _, _, _, X_tN = split_data_sequence_into_datasets(data_with_time_diffs[1], 0.0, 0.0, 0.0, 1.0)
        #_, X_vN1, X_vN2, X_tN = split_data_sequence_into_datasets(data_with_time_diffs[1], 0.0, 0.6, 0.0, 0.4)

        print("reached after data split")

        input_dim = X_sN.shape[2]
        hypermodel = LSTMAutoEncoderHyperModel(input_dim, time_steps)

        print("reached after hypermodel def")

        tuner = RandomSearch(
            hypermodel,
            objective='val_loss',
            max_trials=8,  #10
            executions_per_trial=1,  #3
            directory='tuner_results',
            project_name='lstm_autoencoder_tuning'
        )

        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=30, restore_best_weights=True)

        print("before tuner search")

        tuner.search(x=[X_sN, np.flip(X_sN, axis=1)], y=X_sN,
                     epochs=150,
                     batch_size=32,
                     validation_data=([X_vN1, np.flip(X_vN1, axis=1)], X_vN1),
                     callbacks=[early_stopping],
                     verbose=2)

        print("after tuner search")


        best_model = tuner.get_best_models(num_models=1)[0]
        best_model.summary()
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_hps_dict = best_hps.values

        print("Best hyperparameters:")
        for param, value in best_hps_dict.items():
            print(f"{param}: {value}")

        model_name = "./models/" + "keras_tuned_" + "LSTM_autoencoder_decoder_" + str(file_pair[0][file_pair[0].rfind("\\") + 1:].rstrip(".csv")) + "_timesteps" + str(time_steps) + ".keras"

        print("Saved model to: " + str(model_name))
        best_model.save(model_name)

        print("before predict test")

        autoencoder_predict_and_calculate_error(best_model, X_sN, 1, len(X_sN), scaler)

        print("after predict test")

def calculate_rec_error_vecs(model, X_vN1, scaler):
    error_vecs = []
    for i in range(len(X_vN1)):
        current_sequence = X_vN1[i].reshape((1, X_vN1[i].shape[0], X_vN1[i].shape[1]))
        predicted_sequence = model.predict([current_sequence, np.flip(current_sequence, axis=1)], verbose=0)
        current_sequence = reverse_normalization(np.squeeze(current_sequence, axis=0),
                                                 scaler)  # Reverse reshaping and normalizing
        predicted_sequence = reverse_normalization(np.squeeze(predicted_sequence, axis=0),
                                                   scaler)  # Reverse reshaping and normalizing
        # print("Chosen sequence: " + str(current_sequence))
        # print("Predicted sequences: " + str(predicted_sequence))
        # print("Result: " + str(np.absolute(np.subtract(current_sequence, predicted_sequence))))
        error_vecs.append(np.absolute(np.subtract(current_sequence, predicted_sequence)))
    print("Error vecs: \n" + str(error_vecs))
    return error_vecs

    #todo:
    #      finish evaluation of anomaly score.
    #      //add true_labels to all data
    #      implement overlapping window separation of data
    #      find optimal hyperparameters for each sensor
    #      find a way to create authentic anomalous data and test
    #      find and implement next algorithm
    #      write eMail to Schleif

    #todo: -----------------------------> Try using KerasTuner   <-----------------------------


#the data arrays will have null elements due to the sensors having different measuring intervalls!
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


#todo: sigma is a matrix here. I'm not sure if this is correct but try it for now
def compute_anomaly_score(error, mu, sigma):
    diff = error - mu
    inv_sigma = inv(sigma)  #todo: is this correct?
    score = np.dot(np.dot(diff, inv_sigma), diff.T)  #todo: is this correct?
    return score


#todo: is supposed to use "from sklearn.metrics import precision_recall_curve, fbeta_score" but only uses "fbeta_scores". ChatGPT is currently down so look
#todo: into it later
def find_optimal_threshold(anomaly_scores, true_labels, beta=1.0):
    precision, recall, thresholds = precision_recall_curve(true_labels, anomaly_scores)
    fbeta_scores = (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall)
    best_index = np.argmax(fbeta_scores)
    best_threshold = thresholds[best_index]
    best_fbeta = fbeta_scores[best_index]
    return best_threshold, best_fbeta
