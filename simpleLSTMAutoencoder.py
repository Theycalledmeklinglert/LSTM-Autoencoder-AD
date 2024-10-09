import numpy as np
# from keras.models import Sequential
# from keras.layers import LSTM, Input, Dropout
# from keras.layers import Dense
# from keras.layers import RepeatVector
# from keras.layers import TimeDistributed
import pandas as pd
from keras import Sequential
from keras.src.layers import LSTM, Dropout, RepeatVector, TimeDistributed, Dense
from keras.src.saving import load_model
from matplotlib import pyplot as plt
from pmdarima.arima import ADFTest
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer
#from keras.models import Model
import seaborn as sns

from data_processing import clean_csv, plot_data_standalone, filter_df_by_start_and_end_time_of_activity_phase


def get_trained_LSTM_Autoencder(trainX=None, trainY=None, validX=None, validY=None, batch_size=None, epochs=None, dropout=None, file_path=None):
    if file_path is not None:
        model = load_model(file_path, compile=True)
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.summary()

    else:

        #todo:
        # note "validation_split=0.1" and batch_size; shuffle is true per default
        # batch_size = 32 too large; 4 works well
        # history = model.fit(trainX, trainY, epochs=20, batch_size=4, validation_split=0.1, shuffle=False, verbose=1)
        # todo: changed trainY to trainX


        # model = Sequential()
        # model.add(LSTM(128, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
        # model.add(LSTM(64, activation='relu', return_sequences=False))
        # model.add(Dropout(rate=dropout))
        # model.add(RepeatVector(trainX.shape[1]))
        # #model.add(RepeatVector(future_steps))            #todo: changed to this for possible seq2seq; likely wrong
        #
        # model.add(LSTM(64, activation='relu', return_sequences=True))
        # model.add(LSTM(128, activation='relu', return_sequences=True))
        # model.add(Dropout(rate=dropout))
        # model.add(TimeDistributed(Dense(trainX.shape[2])))
        #model.add(TimeDistributed(Dense(future_steps)))


        model = Sequential()
        model.add(LSTM(128, input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(Dropout(rate=dropout))
        model.add(RepeatVector(trainX.shape[1]))
        model.add(LSTM(128, return_sequences=True))
        model.add(Dropout(rate=dropout))
        model.add(TimeDistributed(Dense(trainX.shape[2])))

        model.compile(optimizer='adam', loss='mean_squared_error')
        #model.compile(optimizer='adam', loss='mae')                #todo: try this one

        model.summary()

        history = model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size,  validation_data=(validX, validY), shuffle=False, verbose=1)  # todo: probably need trainX istead of trainY
        #history = model.fit(trainX, trainY, epochs=5, batch_size=4, validation_split=0.1, shuffle=False, verbose=1)  # todo: probably need trainX istead of trainY

        model.save("./models/SimpleLSTMAutoenc" + ".keras")
        print("saved model to ./models/SimpleLSTMAutoenc.keras")

        plt.plot(history.history['loss'], label='Training loss')
        plt.plot(history.history['val_loss'], label='Validation loss')
        plt.legend()
        plt.show()

    return model

def df_to_sequences(x, y, seq_size):
    x_values = []
    y_values = []

    for i in range(len(x) - seq_size):
        x_values.append(x.iloc[i:(i + seq_size)].values)
        y_values.append(y.iloc[i + seq_size])

        # if future_steps == 1:
        #     y_values.append(y.iloc[i + seq_size])
        # else:
        #     y_values.append(y.iloc[i + seq_size:(i + seq_size + future_steps)].values)

    print("x", np.array(x_values).shape)
    print("y", np.array(y_values).shape)

    return np.array(x_values), np.array(y_values)



if __name__ == '__main__':

    #dataframe = pd.read_csv('data/GE.csv')
    #df = dataframe[['Date', 'Close']]
    #df['Date'] = pd.to_datetime(df['Date'])

    directories = ["./aufnahmen/csv/skidpad_valid_fast2_17_47_28/", "./aufnahmen/csv/autocross_valid_16_05_23/", "./aufnahmen/csv/anomalous data/"]
    sensor_name = "can_interface-current_steering_angle.csv"
    #df0 = clean_csv(directories[0] + sensor_name, False)
    #df1 = clean_csv("./aufnahmen/csv/test data/ebs_test_steering_motor_encoder_damage/can_interface-current_steering_angle.csv", False)
    #df1 = clean_csv(directories[1] + sensor_name, False)
    #df0['Time'] = range(len(df0))
    #df1['Time'] = range(len(df1))

    df0_no_cut = clean_csv(directories[0] + sensor_name, False)
    df1_no_cut = clean_csv(directories[1] + sensor_name, False)
    df2_no_cut = clean_csv(directories[2] + sensor_name, False)

    #todo:first return ins controll_acc df
    _, df0 = filter_df_by_start_and_end_time_of_activity_phase(directories[0], remove_time_col=False,
                                                               control_acc_filename="control-acceleration.csv",
                                                               target_df_filename="can_interface-current_steering_angle.csv")
    _, df1 = filter_df_by_start_and_end_time_of_activity_phase(directories[1], remove_time_col=False, control_acc_filename="control-acceleration.csv", target_df_filename="can_interface-current_steering_angle.csv")

    _, df2 = filter_df_by_start_and_end_time_of_activity_phase(directories[2], remove_time_col=False, control_acc_filename="control-acceleration.csv", target_df_filename="can_interface-current_steering_angle.csv")


    print("Train Orig length: ", len(df0_no_cut))
    print("Test Orig length: ", len(df2_no_cut))
    print("Train Cut length: ", len(df0))
    print("Test Cut length: ", len(df2))

    # df0.drop(columns=["Anomaly", "FR.data", "RL.data", "RR.data"], inplace=True)      #todo: for wheelspeed
    # df1.drop(columns=["Anomaly", "FR.data", "RL.data", "RR.data"], inplace=True)
    df0.drop(columns=["Anomaly"], inplace=True)
    df1.drop(columns=["Anomaly"], inplace=True)
    df2.drop(columns=["Anomaly"], inplace=True)
    attr_1_col_name = df0.columns[1]

    train, valid, test = df0, df1, df2

    print("train: ", train)
    print("test: ", test)

    #print(df0.head())
    sns.lineplot(x=train.index, y=train[attr_1_col_name])
    plt.title('Orig Train')
    plt.show()
    #print(df2.head())
    sns.lineplot(x=valid.index, y=valid[attr_1_col_name])
    plt.title('Orig Valid')
    plt.show()
    #print(df2.head())
    sns.lineplot(x=test.index, y=test[attr_1_col_name])
    plt.title('Orig Test')
    plt.show()

    print("Start train data date is: ", df0['Time'].min())
    print("End train data date is: ", df0['Time'].max())
    print("Start test data date is: ", df2['Time'].min())
    print("End test data date is: ", df2['Time'].max())


    #todo: Diff is very important
    train[attr_1_col_name] = train[attr_1_col_name].diff().fillna(0)
    valid[attr_1_col_name] = valid[attr_1_col_name].diff().fillna(0)
    test[attr_1_col_name] = test[attr_1_col_name].diff().fillna(0)

    sns.lineplot(x=train.index, y=train[attr_1_col_name])
    plt.title('Diffed Train')
    plt.show()
    #print(df2.head())
    sns.lineplot(x=valid.index, y=valid[attr_1_col_name])
    plt.title('Diffed Valid')
    plt.show()
    sns.lineplot(x=test.index, y=test[attr_1_col_name])
    plt.title('Diffed Test')
    plt.show()

    adf_test = ADFTest(alpha=0.05)
    p_val, should_diff = adf_test.should_diff(test[attr_1_col_name].to_numpy().flatten())
    print("For train data ---> p_val: " + str(p_val) + " should_diff: " + str(should_diff))

    # if should_diff:
    #todo:
    # Doesnt work for ebs test steering motor encoder damage.bag
    # Entweder vlt checken ob mean oder Varianz der Diffs übr oder unter irgendeinem Wert liegt
    # ODER: Steering angle imemr multivariate zusammen mit steering_angle_data aus Controll acceleration predicted lassen
    #     test[attr_1_col_name] = test[attr_1_col_name].diff().fillna(0)
    #     print("Test was diffed as determined by ADF p_values > 0.05")
    #     print(df2.head())
    #     sns.lineplot(x=test['Time'], y=test[attr_1_col_name])
    #     plt.title('Diffed Test')
    #     plt.show()
    # else:
    #     print("Test was NOT diffed as determined by ADF p_values < 0.05")
    #     print(df2.head())
    #     sns.lineplot(x=test['Time'], y=test[attr_1_col_name])
    #     plt.title('Not Diffed Test')
    #     plt.show()

    # LSTM uses sigmoid and tanh that are sensitive to magnitude so data needs to be normalized
    # scaler = StandardScaler()           #todo: MaxAbsScaler() might be smart for large val ranges at end of the series'
    # scaler = MaxAbsScaler()
    # scaler = RobustScaler()
    # scaler = QuantileTransformer()

    # TODO:
    # Experiment with using 2 different Scalers!!!! THIS MAY NOT WORK! THEN CHANGE IT BACK!!
    # ----------------------------------------------------------------------------------------------------------

    # scaler_train = MinMaxScaler()  # may work better for steering_angle, and acceleration due to negativ values
    # scaler_test = MinMaxScaler()
    # train[attr_1_col_name] = scaler_train.fit_transform(train[[attr_1_col_name]])
    # test[attr_1_col_name] = scaler_test.fit_transform(test[[attr_1_col_name]])

    scaler = MinMaxScaler()  # may work better for steering_angle, and acceleration due to negativ values
    scaler = scaler.fit(train[[attr_1_col_name]])
    train[attr_1_col_name] = scaler.transform(train[[attr_1_col_name]])
    valid[attr_1_col_name] = scaler.transform(valid[[attr_1_col_name]])
    test[attr_1_col_name] = scaler.transform(test[[attr_1_col_name]])

    # scaler = scaler.fit(train[[attr_1_col_name]])
    # train[attr_1_col_name] = scaler.transform(train[[attr_1_col_name]])
    # test[attr_1_col_name] = scaler.transform(test[[attr_1_col_name]])

    # TODO:
    # ----------------------------------------------------------------------------------------------------------

    print("scaled and diffed train: \n", train)
    print("scaled and diffed valid :  \n", valid)
    print("scaled and diffed test :  \n", test)

    seq_size = 50  # Number of time steps to look back
    #future_steps = 3

    trainX, trainY = df_to_sequences(train[[attr_1_col_name]], train[attr_1_col_name], seq_size)  #todo: double [[]] to return dataframe instead of sequence

    validX, validY = df_to_sequences(valid[[attr_1_col_name]], valid[attr_1_col_name], seq_size)

    testX, testY = df_to_sequences(test[[attr_1_col_name]], test[attr_1_col_name], seq_size)

    #todo:
    # Ich glaube auch dass mein Train Datensatz einfach scheisse gewählt ist wegen der langen linear Section
    # try dropout 0.4
    # also try the other architecture

    model = get_trained_LSTM_Autoencder(trainX, trainY, validX, validY, batch_size=4, epochs=20, dropout=0.4, file_path=None) #, file_path="./models/fuckingSimpleLSTMAutoenc.keras")  #, "./models/pretty good performance on steering angle - fuckingSimpleLSTMAutoenc.keras")

    testPredict = model.predict(testX)
    testMAE = np.mean(np.abs(testPredict - testX), axis=1)
    # plt.hist(testMAE, bins=30)
    # plt.show()

    testMSE = np.mean(np.square(testPredict - testX), axis=1)
    plt.hist(testMSE, bins=30)
    plt.show()

    trainPredict = model.predict(trainX)
    trainMAE = np.mean(np.abs(trainPredict - trainX), axis=1)
    # plt.hist(trainMAE, bins=30)
    # plt.show()

    trainMSE = np.mean(np.square(trainPredict - trainX), axis=1)
    plt.hist(trainMSE, bins=30)
    plt.show()

    max_trainMAE = 1.0 * np.max(trainMAE)  # or Define 90% value of max as threshold.
    max_trainMSE = 1.0 * np.max(trainMSE)  # or Define 90% value of max as threshold.

    test_pred_error_mae = pd.DataFrame(test[seq_size:])
    test_pred_error_mae['testMAE'] = testMAE
    test_pred_error_mae['max_trainMAE'] = max_trainMAE
    test_pred_error_mae['anomaly'] = test_pred_error_mae['testMAE'] > test_pred_error_mae['max_trainMAE']
    test_pred_error_mae[attr_1_col_name] = test[seq_size:][attr_1_col_name]

    # Plot testMAE vs max_trainMAE
    sns.lineplot(x=test_pred_error_mae.index, y=test_pred_error_mae['testMAE'])
    sns.lineplot(x=test_pred_error_mae.index, y=test_pred_error_mae['max_trainMAE'])
    plt.title('Test MAE')
    plt.show()

    # Capture all details in a DataFrame for easy plotting
    test_pred_error_mse = pd.DataFrame(test[seq_size:])
    test_pred_error_mse['testMSE'] = testMSE
    test_pred_error_mse['max_trainMSE'] = max_trainMSE
    test_pred_error_mse['anomaly'] = test_pred_error_mse['testMSE'] > test_pred_error_mse['max_trainMSE']
    test_pred_error_mse[attr_1_col_name] = test[seq_size:][attr_1_col_name]

    # Plot testMSE vs max_trainMAE
    sns.lineplot(x=test_pred_error_mse.index, y=test_pred_error_mse['testMSE'])
    sns.lineplot(x=test_pred_error_mse.index, y=test_pred_error_mse['max_trainMSE'])
    plt.title('Test MSE')
    plt.show()

    detected_anomalies_test_mse = test_pred_error_mse.loc[test_pred_error_mse['anomaly'] == True]

    #anomaly_data_numpy = anomaly_df[attr_1_col_name].to_numpy().reshape(-1, 1)
    #rescaled_anomaly_data = scaler_test.inverse_transform(anomaly_data_numpy).flatten()

    #acutal anomaly plot
    sns.lineplot(x=test_pred_error_mse.index, y=scaler.inverse_transform(test_pred_error_mse[[attr_1_col_name]].to_numpy()).flatten())
    test_pred_errors_over_threshold = detected_anomalies_test_mse[[attr_1_col_name]].to_numpy().flatten()


    if test_pred_errors_over_threshold.shape[0] > 0:
        sns.scatterplot(x=detected_anomalies_test_mse.index, y=scaler.inverse_transform(test_pred_errors_over_threshold.reshape(-1, 1)).flatten(), color='r')
    else:
        print("No pred errors over threshold for test")
    plt.show()

    #todo: do Train data stats here


    train_pred_error_mae = pd.DataFrame(train[seq_size:])
    train_pred_error_mae['trainMAE'] = trainMAE
    train_pred_error_mae['max_trainMAE'] = max_trainMAE
    train_pred_error_mae['anomaly'] = train_pred_error_mae['trainMAE'] > train_pred_error_mae['max_trainMAE']
    train_pred_error_mae[attr_1_col_name] = train[seq_size:][attr_1_col_name]

    # Plot testMAE vs max_trainMAE
    sns.lineplot(x=train_pred_error_mae.index, y=train_pred_error_mae['trainMAE'])
    sns.lineplot(x=train_pred_error_mae.index, y=train_pred_error_mae['max_trainMAE'])
    plt.title('Train MAE')
    plt.show()

    # Capture all details in a DataFrame for easy plotting
    train_pred_error_mse = pd.DataFrame(train[seq_size:])
    train_pred_error_mse['trainMSE'] = trainMSE
    train_pred_error_mse['max_trainMSE'] = max_trainMSE
    train_pred_error_mse['anomaly'] = train_pred_error_mse['trainMSE'] > train_pred_error_mse[
        'max_trainMSE']  # todo: im mixing max_trainMAE with MSE so this might need to be adapted
    train_pred_error_mse[attr_1_col_name] = train[seq_size:][attr_1_col_name]

    # Plot testMSE vs max_trainMAE
    sns.lineplot(x=train_pred_error_mse.index, y=train_pred_error_mse['trainMSE'])
    sns.lineplot(x=train_pred_error_mse.index, y=train_pred_error_mse['max_trainMSE'])
    plt.title('Train MSE')
    plt.show()

    # Plot anomalies
    # sns.lineplot(x=anomaly_df['Time'], y=scaler.inverse_transform(anomaly_df['data']))
    # sns.scatterplot(x=anomalies['Time'], y=scaler.inverse_transform(anomalies['data']), color='r')
