import numpy as np
import tensorflow as tf
from keras.src.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, MaxAbsScaler
from tensorflow import keras
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras import backend
from tensorflow.keras import Model

from raw_data_processing.data_processing import csv_file_to_dataframe_to_numpyArray, \
    convert_timestamp_to_relative_time_diff, reshape_data_for_autoencoder_lstm, normalize_data, \
    split_data_sequence_into_datasets
from utils import CustomL2Loss, autoencoder_predict_and_calculate_error


class LSTMAutoEncoder(tf.keras.Model):
    def __init__(self, input_dim, time_steps, layer_dims, num_layers, dropout):
        super(LSTMAutoEncoder, self).__init__()
        self.layer_dims = layer_dims
        self.input_dim = input_dim
        self.time_steps = time_steps
        self.num_layers = num_layers
        self.dropout = dropout

        # Encoder
        self.encoder_lstm = []
        for i in reversed(layer_dims):
            if i == layer_dims[0]:
                break
            self.encoder_lstm.append(LSTM(i, activation='relu', return_sequences=True, return_state=True, dropout=dropout))
        self.encoder_lstm.append(LSTM(layer_dims[0], activation='relu', return_sequences=False, return_state=True, dropout=dropout))
        #self.encoder_lstm = LSTM(layer_dims[0], activation='relu', return_sequences=False, return_state=True, dropout=dropout)

        # Decoder
        self.decoder_lstm = []
        for i in reversed(layer_dims):
            if i == layer_dims[0]:
                break
            self.decoder_lstm.append(LSTM(i, activation='relu', return_sequences=True, return_state=True, dropout=dropout))
        self.decoder_lstm.append(LSTM(layer_dims[0], activation='relu', return_sequences=True, return_state=True, dropout=dropout))
        #self.decoder_lstm = LSTM(layer_dims[0], activation='relu', return_sequences=True, return_state=True, dropout=dropout)

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
        #todo: unsure if this is correct
        #todo: what the fuck am i using for as first input to the decoder????????
        #todo: ---> I'm either using the encoder_output, an empty vector or the last time_step entry in encoder_inputs
        #todo: ---> during training I (((probably use the the last time_step entry in encoder_inputs))) --> more likely it's an empty vector as well
        # todo: ---> during inference I use an empty vector according to the paper x'(3) = wTh'(3)^D + b
        # todo: ---> in github repo by other guy he just uses the last time_step entry in encoder_inputs

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
        encoder_inputs, decoder_inputs = data[0]    #todo: wtf is happening here?
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

def create_autoencoder(input_dim, time_steps, layer_dims, num_layers, dropout):
    encoder_inputs = Input(shape=(time_steps, input_dim))
    decoder_inputs = Input(shape=(time_steps, input_dim))
    autoencoder = LSTMAutoEncoder(input_dim, time_steps, layer_dims, num_layers, dropout)
    outputs = autoencoder([encoder_inputs, decoder_inputs])    #np.flip(encoder_inputs, axis=1)
    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=outputs)
    model.compile(optimizer=Adam(), loss=CustomL2Loss(), metrics=['accuracy'])
    return model


def test_lstm_autoencoder(time_steps, layer_dims, num_layers, dropout, batch_size, epochs, csv_path):
    #todo: implement overlapping window separation of data
    # ((((todo: maybe data shuffling maybe advisable (think i saw it in the other guys code) --> i dont think so but worth a try ))))

    #scaler = MinMaxScaler()            #Scales the data to a fixed range, typically [0, 1].
    #scaler = StandardScaler()           #Scales the data to have a mean of 0 and a standard deviation of 1.
    scaler = MaxAbsScaler()            #Scales each feature by its maximum absolute value, so that each feature is in the range [-1, 1]. #todo: best performance so far

    data = csv_file_to_dataframe_to_numpyArray(csv_path)
    data_with_time_diffs = []
    for sample in data:
        print("Sample shape: " + str(sample.shape))
        unscaled_data_with_time_diffs = convert_timestamp_to_relative_time_diff(sample)
        print("unscaled_data_with_time_diffs: \n" + str(unscaled_data_with_time_diffs))
        normalized_data_with_time_diffs = normalize_data(unscaled_data_with_time_diffs, scaler)
        print("normalized_data_with_time_diffs: \n" + str(normalized_data_with_time_diffs))
        data_with_time_diffs.append(normalized_data_with_time_diffs)

    data_with_time_diffs = reshape_data_for_autoencoder_lstm(data_with_time_diffs, time_steps)
    X_sN = data_with_time_diffs[0]  #todo: ideally/eventually I would use completely seperate datasets/csv for all of them
    _, X_vN1, X_vN2, _ = split_data_sequence_into_datasets(data_with_time_diffs[1], 0.0, 1.0, 0.0, 0.0)  #todo: ideally/eventually I would use completely seperate datasets/csv for all of them
    _, _, _, X_tN = split_data_sequence_into_datasets(data_with_time_diffs[1], 0.0, 0.0, 0.0, 1.0)  #todo: ideally/eventually I would use completely seperate datasets/csv for all of them

    input_dim = X_sN.shape[2]

    model = create_autoencoder(input_dim, time_steps, layer_dims, num_layers, dropout)
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=30, restore_best_weights=True)

    #if X_sN doesnt get flipped in create_autoencoder because tensorflow hates me then I need to add it here!!!
    model.fit([X_sN, np.flip(X_sN, axis=1)], X_sN, epochs=epochs, batch_size=batch_size, validation_data=([X_vN1, np.flip(X_vN1, axis=1)], X_vN1), verbose=1, callbacks=[early_stopping])

    autoencoder_predict_and_calculate_error(model, X_tN, 1, 100, scaler)

