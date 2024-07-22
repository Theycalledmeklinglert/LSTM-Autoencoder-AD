import random
from functools import partial

import numpy as np
from keras import Sequential, Loss, Input, Model
from keras.src.callbacks import EarlyStopping
from keras.src.layers import LSTM, Dense, Reshape, RepeatVector, TimeDistributed, Lambda
from keras.src.optimizers import Adam
from scikeras.wrappers import KerasClassifier
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler

from raw_data_processing.data_processing import old_directory_csv_files_to_dataframe_to_numpyArray, \
    convert_timestamp_to_absolute_time_diff, convert_timestamp_to_relative_time_diff, normalize_data, \
    reshape_data_for_autoencoder_lstm
from stacked_lstm import create_XY_data_sequences, split_data_sequence_into_datasets, check_shapes_after_reshape
from utils import autoencoder_predict_and_calculate_error, CustomL2Loss
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.python.keras import backend


global time_steps  # Use the time_steps most recent value
global future_steps  # Predict the next future_steps values
global scaler



def old_LSTM_autoencoder(csv_path):
    global time_steps
    global future_steps
    global scaler
    time_steps = 1
    future_steps = 1
    scaler = MinMaxScaler()

    data = old_directory_csv_files_to_dataframe_to_numpyArray(csv_path)
    print(data.shape)

    #data_with_time_diffs = convert_timestamp_to_absolute_time_diff(data)
    data_with_time_diffs = convert_timestamp_to_relative_time_diff(data)
    print("data_with_time_diffs: \n" + str(data_with_time_diffs))

    #data_with_time_diffs = normalize_data(data_with_time_diffs, scaler)
    #print("Normalized data: \n" + str(data_with_time_diffs))

    #TODO: I need to standardize scaling somehow. for steering angle i.e. can approximately go from -100 to 100 while time is very large

    X_sN, X_vN2, X_tN = split_data_sequence_into_datasets(data_with_time_diffs)

    X_sN = reshape_data_for_autoencoder_lstm(X_sN, time_steps)
    X_vN2 = reshape_data_for_autoencoder_lstm(X_vN2, time_steps)
    X_tN = reshape_data_for_autoencoder_lstm(X_tN, time_steps)

    #todo:
    #1. Test while changing parameters
    #2. TODO:try GridSearch in encoder and stacked LSTM
    #3. Think about changing time_steps and test
    #4. Try to adapt model to work with sequences  i.e. take a sequence as input and reconstruct sequence out of it or sth. to be able to handle anomalous sequences
    #5. Find a way to define a threshold for detecting anomaly out of prediction error
    #  ----> Doesnt this just work for point anomalies?
    #6. Fuse multiple datasets somehow and test performance
    #8. TODO: follow the process dscribed here!!!::file:///C:/Users/Luca/Downloads/1607.00148v2.pdf;file:///C:/Users/Luca/Downloads/1607.00148v2.pdf;file:///C:/Users/Luca/Downloads/1607.00148v2.pdf;file:///C:/Users/Luca/Downloads/1607.00148v2.pdf
            #todo: -----> Do I have to let the encoder process the entire sequence and then shove that into the decoder but in reverse order and the decoder is trained on the reconstruction error?
            #todo: -----> then how do I train the encoder on it
            #todo: -----> it uses windows too lmao??

    # # Run grid search
    # param_grid = {'classifier__input_shape': [X.shape[1]],
    #               'classifier__batch_size': [50],
    #               'classifier__learn_rate': [0.001],
    #               'classifier__epochs': [5, 10]}
    # cv = KFold(n_splits=5, shuffle=False)
    # grid = GridSearchCV(estimator=clf, param_grid=param_grid,
    #                     scoring='neg_mean_squared_error', verbose=1, cv=cv)
    # grid_result = grid.fit(X, X)
    #
    # print('Best: %f using %s' % (grid_result.best_score_, grid_result.best_params_))

    # Encoder
    encoder = Sequential()
    encoder.add(LSTM(16, activation='relu', input_shape=(time_steps, X_sN.shape[2]),
                     return_sequences=False))  #maybe change number of units
    print("1: " + str(encoder.output_shape))  # Print output shape after first LSTM layer

    #encoder.add(LSTM(4, activation='relu', return_sequences=True, recurrent_dropout=0.2)) #try with return_sequneces=false
    encoder.add(Dense(8, activation='relu'))
    print("2: " + str(encoder.output_shape))  # Print output shape after first LSTM layer

    # Decoder
    decoder = Sequential()
    decoder.add(RepeatVector(time_steps, input_shape=(8,)))  #ChatGPT
    decoder.add(
        LSTM(8, activation='relu', return_sequences=True, recurrent_dropout=0.2))  #try with return_sequneces=false
    print("3: " + str(decoder.output_shape))  # Print output shape after first LSTM layer
    decoder.add(
        LSTM(16, activation='relu', return_sequences=True, recurrent_dropout=0.2))  #try with return_sequneces=false
    print("4: " + str(decoder.output_shape))  # Print output shape after first LSTM layer
    decoder.add(Dense(X_sN.shape[2]))
    print("5: " + str(decoder.output_shape))  # Print output shape after first LSTM layer

    # Autoencoder
    autoencoder = Sequential([encoder, decoder])
    autoencoder.compile(optimizer='adam', loss=CustomL2Loss(), metrics=['accuracy'])# was < loss='mse' >  before

    autoencoder.summary()
    #early_stopping_callback = EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True, mode='min', min_delta=0.001)
    #lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)

    print(str(X_sN.shape))
    print(str(X_vN2.shape))

    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=10, restore_best_weights=True)
    autoencoder.fit(X_sN, X_sN, epochs=100, batch_size=32, validation_data=(X_vN2, X_vN2), verbose=1, callbacks=[early_stopping])

    #autoencoder_train = autoencoder.fit(train_X, train_ground, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(valid_X, valid_ground))

    autoencoder.save('./models/LSTM_autoencoder_decoder.keras')
    autoencoder_predict_and_calculate_error(autoencoder, X_tN, future_steps, 10, scaler)




def grid_search_LSTM_autoencoder(csv_path):
    time_steps = 1
    future_steps = 1  # Predict the next 3 values
    data = old_directory_csv_files_to_dataframe_to_numpyArray(csv_path)
    print(data.shape)

    data_with_time_diffs = convert_timestamp_to_absolute_time_diff(data)
    print(str(data_with_time_diffs))
    X_sN, X_vN2, X_tN = split_data_sequence_into_datasets(data_with_time_diffs)
    X_sN = reshape_data_for_autoencoder_lstm(X_sN, time_steps)
    X_vN2 = reshape_data_for_autoencoder_lstm(X_vN2, time_steps)
    X_tN = reshape_data_for_autoencoder_lstm(X_tN, time_steps)

    input_dim = X_sN.shape[2]  # Number of features

    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=10, restore_best_weights=True)

    create_model_partial = partial(create_model, time_steps=time_steps, input_dim=input_dim)
    model = KerasClassifier(build_fn=create_model_partial, epochs=100, batch_size=32, verbose=1)
    param_grid = {
        'lstm_units_1': [16, 32, 64],
        'lstm_units_2': [4, 8, 16],
        'dense_units': [8, 16],
        'learning_rate': [0.001, 0.01],
        'batch_size': [32, 64, 128]
    }
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    grid_result = grid.fit(X_sN, X_sN, validation_data=(X_vN2, X_vN2), callbacks=[early_stopping])

    print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, std, param in zip(means, stds, params):
        print(f"{mean} ({std}) with: {param}")


def create_model(time_steps, input_dim, lstm_units_1=16, lstm_units_2=8, dense_units=8, learning_rate=0.001):
    encoder = Sequential()
    encoder.add(LSTM(lstm_units_1, activation='relu', input_shape=(time_steps, input_dim), return_sequences=False))
    encoder.add(Dense(dense_units, activation='relu'))

    decoder = Sequential()
    decoder.add(RepeatVector(time_steps, input_shape=(dense_units,)))
    decoder.add(LSTM(lstm_units_2, activation='relu', return_sequences=True))
    decoder.add(LSTM(lstm_units_1, activation='relu', return_sequences=True))
    decoder.add(Dense(input_dim))

    autoencoder = Sequential([encoder, decoder])
    autoencoder.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['accuracy'])

    return autoencoder
