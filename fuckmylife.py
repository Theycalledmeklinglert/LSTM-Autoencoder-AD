import time
import os

from keras.src.layers import LSTM, Dense
import tensorflow as tf
import keras
import numpy as np
from torch.utils.data import DataLoader

from NEWLSTMAutoencoder import NEWLSTMAutoEncoder
from torch_utils import get_data_as_list_of_single_batches_of_subseqs

# This guide can only be run with the TensorFlow backend.
os.environ["KERAS_BACKEND"] = "tensorflow"

batch_size = 32
size_window = 0
step_window = 0
scaler = None
all_data = get_data_as_list_of_single_batches_of_subseqs(size_window, step_window, True, scaler=scaler, directories=["./aufnahmen/csv/skidpad_valid_fast2_17_47_28", "./aufnahmen/csv/skidpad_valid_fast3_17_58_41", "./aufnahmen/csv/anomalous data", "./aufnahmen/csv/test data/skidpad_falscher_lenkungsoffset"], single_sensor_name="can_interface-current_steering_angle.csv")

optimizer = keras.optimizers.Adam(learning_rate=1e-3)
loss_fn = keras.losses.MeanSquaredError()               #loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model = NEWLSTMAutoEncoder(0, 0, 0, 0)

train_batches = all_data[0][0]      # because get_data_as_single_batches_of_subseqs return [data_with_time_diffs, true_label_list]
valid_batches = all_data[0][1]
anomaly_batches = all_data[0][2]
x_T_batches = all_data[0][3]

train_loader = DataLoader(train_batches, batch_size=batch_size, shuffle=False)      # shuffle needs to be off;
valid_loader = DataLoader(valid_batches, batch_size=batch_size, shuffle=False)       # shuffle needs to be off;
anomaly_loader = DataLoader(anomaly_batches, batch_size=batch_size, shuffle=False)  # shuffle needs to be off;
test_loader = DataLoader(x_T_batches, batch_size=batch_size, shuffle=False)         # shuffle needs to be off;

def train(epochs):
    for epoch in range(epochs):
        print(f"\nStart of epoch {epoch}")

        # Iterate over the batches of the dataset.
        for id_batch, train_batch in enumerate(train_loader):
            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:
                # Run the forward pass of the layer.
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.
                predictions = model.call(train_batch, training=True)

                # Compute the loss value for this minibatch.
                loss_value = loss_fn(y_batch_train, predictions)    #todo: predictions has to be (batch_size, timesteps, features)

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, model.trainable_weights)
            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply(grads, model.trainable_weights)

            with tf.GradientTape() as tape:
                predictions = self.call([encoder_inputs, decoder_inputs], is_training=True)
                loss = self.compiled_loss(target, predictions, regularization_losses=self.losses)
                # loss = self.compute_loss(target, predictions)
                # loss += tf.add_n(self.losses)
                print("does this work?")
                tf.print("does this work?")
                if loss is None:
                    print("Loss is None!")

            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            self.compiled_metrics.update_state(target, predictions)

            return {m.name: m.result() for m in self.metrics}

            # Log every 100 batches.
            if id_batch % 100 == 0:
                print(
                    f"Training loss (for 1 batch) at step {id_batch}: {float(loss_value):.4f}"
                )
                print(f"Seen so far: {(id_batch + 1) * batch_size} samples")

