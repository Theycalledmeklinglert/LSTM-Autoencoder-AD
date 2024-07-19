import numpy as np
import tensorflow as tf
import keras
from keras.src.callbacks import EarlyStopping
from keras.src.saving import load_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, MaxAbsScaler
#from tensorflow import keras
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model

from raw_data_processing.data_processing import csv_files_to_dataframe_to_numpyArray, \
    convert_timestamp_to_relative_time_diff, reshape_data_for_autoencoder_lstm, normalize_data, \
    split_data_sequence_into_datasets, reverse_normalize_data
from utils import CustomL2Loss, autoencoder_predict_and_calculate_error


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
        for i in reversed(layer_dims):
            if i == layer_dims[0]:
                break
            self.encoder_lstm.append(LSTM(i, activation='relu', return_sequences=True, return_state=True, dropout=dropout))
        self.encoder_lstm.append(LSTM(layer_dims[0], activation='relu', return_sequences=False, return_state=True, dropout=dropout))
        # Decoder
        for i in reversed(layer_dims):
            if i == layer_dims[0]:
                break
            self.decoder_lstm.append(LSTM(i, activation='relu', return_sequences=True, return_state=True, dropout=dropout))
        self.decoder_lstm.append(LSTM(layer_dims[0], activation='relu', return_sequences=True, return_state=True, dropout=dropout))
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


        #TODO: Should I pass the states inbetween layers or keep each layer's state in its corresponding layer and then only pass the states of the last layer to the decoder?
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
        inputs = decoder_inputs[:, 0:1, :] #last entry in time-series
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
        encoder_inputs, decoder_inputs = data[0]    # Unpacking encoder and decoder inputs
        target = data[1]                            # Target values for current batch

        predictions = self([encoder_inputs, decoder_inputs], training=False)
        loss = self.compiled_loss(target, predictions, regularization_losses=self.losses)
        self.compiled_metrics.update_state(target, predictions)

        return {m.name: m.result() for m in self.metrics}

def calculate_rec_error_vecs(model, X_vN1, scaler):
    error_vecs = []
    for i in range(len(X_vN1)):
        current_sequence = X_vN1[i].reshape((1, X_vN1[i].shape[0], X_vN1[i].shape[1]))
        predicted_sequence = model.predict([current_sequence, np.flip(current_sequence, axis=1)], verbose=0)
        current_sequence = reverse_normalize_data(np.squeeze(current_sequence, axis=0), scaler)  # Reverse reshaping and normalizing
        predicted_sequence = reverse_normalize_data(np.squeeze(predicted_sequence, axis=0), scaler)  # Reverse reshaping and normalizing
        #print("Chosen sequence: " + str(current_sequence))
        #print("Predicted sequences: " + str(predicted_sequence))
        error_vecs.append(np.subtract(current_sequence, predicted_sequence))
    print(error_vecs)
    return error_vecs


def estimate_error_distribution(error_vecs):
    print("dipshit")
    return



def create_autoencoder(input_dim, time_steps, layer_dims, num_layers, dropout):
    keras.saving.get_custom_objects().clear()
    encoder_inputs = Input(shape=(time_steps, input_dim))
    decoder_inputs = Input(shape=(time_steps, input_dim))
    autoencoder = LSTMAutoEncoder(input_dim, time_steps, layer_dims, num_layers, dropout)
    outputs = autoencoder([encoder_inputs, decoder_inputs])
    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=outputs)
    model.compile(optimizer=Adam(), loss=CustomL2Loss(), metrics=['accuracy'])
    return model


def test_lstm_autoencoder(time_steps, layer_dims, num_layers, dropout, batch_size, epochs, directories, model_file_path=None):
    #todo: implement overlapping window separation of data
    # ((((todo: maybe data shuffling maybe advisable (think i saw it in the other guys code) --> i dont think so but worth a try ))))

    scaler = MinMaxScaler(feature_range=(-1, 1))            #Scales the data to a fixed range, typically [0, 1].
    #scaler = StandardScaler()           #Scales the data to have a mean of 0 and a standard deviation of 1.
    #scaler = MaxAbsScaler()            #Scales each feature by its maximum absolute value, so that each feature is in the range [-1, 1]. #todo: best performance so far

    data = []
    #data is automatically scaled to relative timestamps
    for directory in directories:
        data.append(csv_files_to_dataframe_to_numpyArray(directory))

    print("converted csv('s) to numpy array: " + str(data))

    return


    # todo: the data arrays will have null elements due to the sensors having different measuring intervalls!
    # possible solutions:
    #       1. delete every entry of the data array that contains an empty cell ---> THIS DOESNT FKING WORK BECAUSE I WILL LOSE LIKE 9/10th OF THE DATA THAT HAVE SHORTER MEASURING INTERVALLS AND THE VALUES WONT EVEN REALLY
    #          BE CORRELATED
    #       2. find sensor with least data entries; downsample all other sensor data accordingly before adding them together
    #       ----> I think 1. and 2. are equivalent aaaaaaaaaaaaaaaaaaaahhhhhh
    #       3. average empty cells out or sth, idk that sounds retarded AND complicated, my favorite
    #       3. Kill myself

    data_with_time_diffs = []
    for samples in data:
        print("unscaled_data_with_time_diffs: \n" + str(samples))
        normalized_data_with_time_diffs = normalize_data(samples, scaler)
        print("normalized_data_with_time_diffs: \n" + str(normalized_data_with_time_diffs))
        data_with_time_diffs.append(normalized_data_with_time_diffs)

    data_with_time_diffs = reshape_data_for_autoencoder_lstm(data_with_time_diffs, time_steps)
    X_sN = data_with_time_diffs[0]  #ideally/eventually I would use completely seperate datasets/csv for all of them
    _, X_vN1, X_vN2, _ = split_data_sequence_into_datasets(data_with_time_diffs[1], 0.0, 1.0, 0.0, 0.0)
    _, _, _, X_tN = split_data_sequence_into_datasets(data_with_time_diffs[1], 0.0, 0.0, 0.0, 1.0)

    input_dim = X_sN.shape[2]
    if model_file_path is None:
        model = create_autoencoder(input_dim, time_steps, layer_dims, num_layers, dropout)
        model.summary()
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=30, restore_best_weights=True)
        #if X_sN doesnt get flipped in create_autoencoder because tensorflow hates me then I need to add it here!!!
        model.fit([X_sN, np.flip(X_sN, axis=1)], X_sN, epochs=epochs, batch_size=batch_size, validation_data=([X_vN1, np.flip(X_vN1, axis=1)], X_vN1), verbose=1, callbacks=[early_stopping])
        #autoencoder_predict_and_calculate_error(model, X_tN, 1, 1, scaler)
        model.save('./models/LSTM_autoencoder_decoder_30_30.keras')
    else:
        model = load_model(model_file_path, custom_objects={"LSTMAutoEncoder" : LSTMAutoEncoder}, compile=True)

    #autoencoder_predict_and_calculate_error(model, X_tN, 1, 100, scaler)
    calculate_rec_error_vecs(model, X_vN1, scaler)


