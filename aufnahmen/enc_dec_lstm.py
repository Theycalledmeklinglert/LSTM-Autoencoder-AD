import random

import numpy as np
from keras import Sequential
from keras.src.callbacks import EarlyStopping
from keras.src.layers import LSTM, Dense, Reshape, RepeatVector
from sklearn.preprocessing import StandardScaler

from raw_data_processing.data_processing import csv_file_to_dataframe_to_numpyArray, convert_timestamp_to_time_diff
from stacked_lstm import AccuracyThresholdCallback, create_XY_data_sequences, split_data_sequence_into_datasets, \
    reshape_data_for_LSTM, check_shapes_after_reshape


def reshape_data_for_autencoder_LSTM(data, time_steps):
    # Reshape X to fit LSTM input shape (samples, time steps, features)
    print(data.shape)
    data = data.reshape((data.shape[0], time_steps, data.shape[1]))
    print("Reshaped data for LSTM into: " + str(data))
    return data

def test_LSTM_autoencoder(csv_path):
    #data_with_time_diffs = generate_test_array()
    time_steps = 1  # Use the 3 most recent value
    future_steps = 1  # Predict the next 3 values
    accuracy_threshold = 0.98
    accuracy_threshold_callback = AccuracyThresholdCallback(threshold=accuracy_threshold)

    data = csv_file_to_dataframe_to_numpyArray(csv_path)
    print(data.shape)

    data_with_time_diffs = convert_timestamp_to_time_diff(data)
    print(str(data_with_time_diffs))

    #data_with_time_diffs = data

    #TODO: I need to standardize scaling somehow. for steering angle i.e. can approximately go from -100 to 100 while time is very large
    # feature_1 = data_with_time_diffs[:, 0].reshape(-1, 1)  # The large-scale feature
    # feature_2 = data_with_time_diffs[:, 1].reshape(-1, 1)  # The smaller-scale feature
    # scaler = StandardScaler()
    # normalized_feature_1 = scaler.fit_transform(feature_1)
    # normalized_feature_2 = scaler.fit_transform(feature_2)
    #data_with_time_diffs = np.hstack((normalized_feature_1, normalized_feature_2))

    #X, Y = create_XY_data_sequences(data_with_time_diffs, time_steps, future_steps)
    #print("X: " + str(X))
    #print("Y: " + str(Y))
    X_sN, X_vN2, X_tN = split_data_sequence_into_datasets(data_with_time_diffs)
    #Y_sN, Y_vN2, Y_tN = split_data_sequence_into_datasets(Y)

    X_sN = reshape_data_for_autencoder_LSTM(X_sN, time_steps)
    X_vN2 = reshape_data_for_autencoder_LSTM(X_vN2, time_steps)
    X_tN = reshape_data_for_autencoder_LSTM(X_tN, time_steps)

    #todo: Macht es Sinn das Zeitfeld mitzuverarbeiten?
    #todo:
    #1.test while changing parameters
    #2. TODO:try GridSearch in encoder and stacked LSTM
    #3. Figure out if scaling data_with_time_diffs is suitable for autoencoder and how to
    #4. Think about changing time_steps and test
    #5. The loss function is not suitable when using not time_diff data. It overfits haaaaard and stops immediately

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
    encoder.add(LSTM(8, activation='relu', input_shape=(time_steps, X_sN.shape[2]), return_sequences=False)) #maybe change number of units
    print("1: " + str(encoder.output_shape))  # Print output shape after first LSTM layer
    #encoder.add(Reshape((1, 32)))  # Reshape output to match expected input of next layer

    #32*time_steps?
    #encoder.add(LSTM(4, activation='relu', return_sequences=True, recurrent_dropout=0.2)) #try with return_sequneces=false
    encoder.add(Dense(4, activation='relu'))
    print("2: " + str(encoder.output_shape))  # Print output shape after first LSTM layer

    # Decoder
    decoder = Sequential()
    decoder.add(RepeatVector(time_steps, input_shape=(4,))) #ChatGPT
    #                               input_shape=(time_steps, 32),
    decoder.add(LSTM(4, activation='relu', return_sequences=True, recurrent_dropout=0.2)) #try with return_sequneces=false
    print("3: " + str(decoder.output_shape))  # Print output shape after first LSTM layer

    decoder.add(LSTM(8, activation='relu', return_sequences=True, recurrent_dropout=0.2)) #try with return_sequneces=false
    print("4: " + str(decoder.output_shape))  # Print output shape after first LSTM layer
    decoder.add(Dense(X_sN.shape[2]))
    print("5: " + str(decoder.output_shape))  # Print output shape after first LSTM layer

    # Autoencoder
    autoencoder = Sequential([encoder, decoder])
    autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    autoencoder.summary()
    #early_stopping_callback = EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True, mode='min', min_delta=0.001)
    #lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)

    print(str(X_sN.shape))
    print(str(X_vN2.shape))

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    autoencoder.fit(X_sN, X_sN, epochs=100, batch_size=32, validation_data=(X_vN2, X_vN2), verbose=1, callbacks=[early_stopping])

    #autoencoder_train = autoencoder.fit(train_X, train_ground, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(valid_X, valid_ground))

    #model.save('./models/stacked_LSTM.keras')
    autoencoder.save('./models/LSTM_autoencoder_decoder.keras')
    #m (=input data_with_time_diffs dimensions) input units; dxl (d = features to be predicted, number of time steps to be predicted into future) output units

    for i in range(0, 10):
        rand_int = random.randint(0, X_tN.shape[0])
        recent_sequence = np.array(X_tN[rand_int])  # sequence to be predicted
        print("Sequence chosen for prediction: " + str(recent_sequence))
        print(recent_sequence.shape)
        # Reshape recent_sequence to fit LSTM input shape (samples, time steps, features)
        recent_sequence = recent_sequence.reshape((1, recent_sequence.shape[0], recent_sequence.shape[1]))

        # Predict future_steps sequences
        predicted_sequences = autoencoder.predict(recent_sequence)

        # Reshape predicted sequences to match the original y shape
        predicted_sequences = predicted_sequences.reshape((future_steps, X_tN.shape[2])) #data_with_time_diffs.shape[1])

        print(f"Predicted sequences: \n{predicted_sequences}")

