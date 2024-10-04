from keras.src.layers import LSTM, Dense
import tensorflow as tf
import keras


@keras.saving.register_keras_serializable(package="MyLayers")
class NEWLSTMAutoEncoder(tf.keras.Model):
    def __init__(self, input_dim, time_steps, layer_dims, dropout, **kwargs):
        super(NEWLSTMAutoEncoder, self).__init__(**kwargs)
        self.layer_dims = layer_dims
        self.input_dim = input_dim
        self.time_steps = time_steps
        #self.num_layers = num_layers
        self.dropout = dropout
        self.encoder_lstm = LSTM(layer_dims, activation='relu', return_sequences=True, return_state=True, dropout=dropout)
        self.decoder_lstm = LSTM(layer_dims, activation='relu', return_sequences=True, return_state=True, dropout=dropout)
        self.decoder_dense = Dense(input_dim)    #, kernel_regularizer=regularizers.L2(0.01))     #todo: the regularizer is EXPERIMENTAL;dont really know if its beneficial here

    def call(self, inputs, training=False):
        print("Training argument is " + str(training))

        encoder_inputs, decoder_inputs = inputs
        # encoder_states_empty = True
        # encoder_states = [None, None]
        batch_size = tf.shape(encoder_inputs)[0]
        hidden_size = self.encoder_lstm.units
        encoder_states = [tf.zeros((batch_size, hidden_size)), tf.zeros((batch_size, hidden_size))]
        # for t in range(self.time_steps):
        #     x = encoder_inputs[:, t:t + 1, :]  # Select one time step
        #     x, state_h, state_c = self.encoder_lstm(x) #, initial_state=encoder_states)  # todo: REMOVED; EXPERIMENTAL!!!, initial_state=encoder_states)
        #     encoder_states = [state_h, state_c]

        x, state_h, state_c = self.encoder_lstm(encoder_inputs)
        encoder_states = [state_h, state_c]

        all_outputs = []
        decoder_states = encoder_states
        if training:
            dec_input = decoder_inputs[:, 0:1,
                        :]  # Use first value of the reversed sequence as initial input for the decoder
        else:
            dec_input = tf.expand_dims(self.decoder_dense(decoder_states[0]),
                                       axis=1)  # use last state of encoder as decoder input make predict first prediction of sequence
        all_outputs.append(dec_input)

        for t in range(1, self.time_steps):
            # print("iteration: " + str(t))
            dec_output, state_h, state_c = self.decoder_lstm(dec_input, initial_state=decoder_states)
            decoder_states = [state_h, state_c]  # todo: not sure if state_c should be passed
            dec_output = tf.expand_dims(self.decoder_dense(state_h), axis=1)
            # print("please end me")
            # print("state_h: " + str(state_h))
            # print("dec_output: " + str(dec_output))
            #dec_output = self.decoder_dense(dec_output)

            if training:
                dec_input = decoder_inputs[:, t:t + 1, :]  # Use ground truth as the next input #todo: try here
            else:
                dec_input = dec_output
            all_outputs.append(dec_output)

        # reverse decoder output since decoder predicts target data in reverse
        all_outputs = all_outputs[::-1]
        dec_output = tf.concat(all_outputs, axis=1)
        return dec_output

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

