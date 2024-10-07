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

from data_processing import clean_csv


def get_trained_LSTM_Autoencder(trainX=None, trainY=None, file_path=None):
    if file_path is not None:
        model = load_model(file_path, compile=True)
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.summary()

    else:
        # define Autoencoder model
        # Input shape is seq_size, nb_features
        # batch_size = trainX.shape[0] (Keras handles this), seq_size = trainX.shape[1], nb_features = trainX.shape[2]

        model = Sequential()
        model.add(LSTM(128, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
        model.add(LSTM(64, activation='relu', return_sequences=False))
        model.add(Dropout(rate=0.2))
        model.add(RepeatVector(trainX.shape[1]))
        model.add(LSTM(64, activation='relu', return_sequences=True))
        model.add(LSTM(128, activation='relu', return_sequences=True))
        model.add(Dropout(rate=0.2))
        model.add(TimeDistributed(Dense(trainX.shape[2])))

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.summary()

        history = model.fit(trainX, trainY, epochs=20, batch_size=4, validation_split=0.1, shuffle=False, verbose=1)  #todo: probably need trainX istead of trainY
        # todo: changed trainY to trainX

        model.save("./models/fuckingSimpleLSTMAutoenc" + ".keras")

        plt.plot(history.history['loss'], label='Training loss')
        plt.plot(history.history['val_loss'], label='Validation loss')
        plt.legend()
        plt.show()

        # model.compile(optimizer='adam', loss='mse')
        # model.summary()

        # Try another model
        # model = Sequential()
        # model.add(LSTM(128, input_shape=(trainX.shape[1], trainX.shape[2])))    #todo: went from 128 to 256
        # model.add(Dropout(rate=0.2))
        # model.add(RepeatVector(trainX.shape[1]))
        # model.add(LSTM(128, return_sequences=True))                        #todo: went from 128 to 256
        # model.add(Dropout(rate=0.2))
        # model.add(TimeDistributed(Dense(trainX.shape[2])))                           #todo: went from 128 to 256



    return model

if __name__ == '__main__':


    #dataframe = pd.read_csv('data/GE.csv')
    #df = dataframe[['Date', 'Close']]
    #df['Date'] = pd.to_datetime(df['Date'])

    # df0 = clean_csv("./aufnahmen/csv/skidpad_valid_fast2_17_47_28/can_interface-current_steering_angle.csv", False)
    # df1 = clean_csv("./aufnahmen/csv/anomalous data/can_interface-current_steering_angle.csv", False)

    df0 = clean_csv("./aufnahmen/csv/autocross_valid_run/can_interface-current_steering_angle.csv", False)
    df1 = clean_csv("./aufnahmen/csv/test data/ebs_test_steering_motor_encoder_damage/can_interface-current_steering_angle.csv", False)
    #df1 = clean_csv("./aufnahmen/csv/anomalous data/can_interface-wheelspeed.csv", False)

    # df0.drop(columns=["Anomaly", "FR.data", "RL.data", "RR.data"], inplace=True)      #todo: for wheelspeed
    # df1.drop(columns=["Anomaly", "FR.data", "RL.data", "RR.data"], inplace=True)
    df0.drop(columns=["Anomaly"], inplace=True)
    #df1.drop(columns=["Anomaly"], inplace=True)


    attr_1_col_name = df0.columns[1]

    print(df0.head())
    sns.lineplot(x=df0['Time'], y=df0[attr_1_col_name])
    plt.title('Orig Train')
    plt.show()
    #df1.drop(columns=["Anomaly"], inplace=True)
    print(df1.head())
    sns.lineplot(x=df1['Time'], y=df1[attr_1_col_name])
    plt.title('Orig Test')
    plt.show()

    print("Start train data date is: ", df0['Time'].min())
    print("End train data date is: ", df0['Time'].max())

    # Change train data from Mid 2017 to 2019.... seems to be a jump early 2017
    #train, test = df.loc[df['Date'] <= '2003-12-31'], df.loc[df['Date'] > '2003-12-31'] #todo: original. Uses trainX to predict last part of itself (testX)
    train, test = df0, df1          #todo: original. Uses trainX to predict last part of itself (testX)

    train[attr_1_col_name] = train[attr_1_col_name].diff().fillna(0)
    #test[attr_1_col_name] = test[attr_1_col_name].diff().fillna(0)

    adf_test = ADFTest(alpha=0.05)
    p_val, should_diff = adf_test.should_diff(test[attr_1_col_name].to_numpy().flatten())
    print("For train data ---> p_val: " + str(p_val) + " should_diff: " + str(should_diff))

    if should_diff:
        #todo:
        # Doesnt work for ebs test steering motor encoder damage.bag
        # Entweder vlt checken ob mean oder Varianz der Diffs übr oder unter irgendeinem Wert liegt
        # ODER: Steering angle imemr multivariate zusammen mit steering_angle_data aus Controll acceleration predicted lassen
        test[attr_1_col_name] = test[attr_1_col_name].diff().fillna(0)
        print("Test was diffed as determined by ADF p_values > 0.05")
        print(df1.head())
        sns.lineplot(x=test['Time'], y=test[attr_1_col_name])
        plt.title('Diffed Test')
        plt.show()
    else:
        print("Test was NOT diffed as determined by ADF p_values < 0.05")
        print(df1.head())
        sns.lineplot(x=test['Time'], y=test[attr_1_col_name])
        plt.title('Not Diffed Test')
        plt.show()


    sns.lineplot(x=train['Time'], y=train[attr_1_col_name])
    plt.title('Diffed Train')
    plt.show()

    print("train: ", train)
    print("test: ", test)

    # LSTM use sigmoid and tanh that are sensitive to magnitude so values need to be normalized
    # scaler = MinMaxScaler()
    scaler = StandardScaler()           #todo: MaxAbsScaler() might be smart for large val ranges at end of the series'
    #scaler = MaxAbsScaler()
    #scaler = RobustScaler()
    #scaler = QuantileTransformer()



    scaler = scaler.fit(train[[attr_1_col_name]])

    train[attr_1_col_name] = scaler.transform(train[[attr_1_col_name]]) #todo: NOTE that he doesnt use fit_transform as "Time" is still in df
    test[attr_1_col_name] = scaler.transform(test[[attr_1_col_name]])

    print("scaled train: ", train)
    print("scaled test: ", test)

    seq_size = 30  # Number of time steps to look back

    def to_sequences(x, y, seq_size=1):
        x_values = []
        y_values = []

        for i in range(len(x) - seq_size):
            # print(i)
            x_values.append(x.iloc[i:(i + seq_size)].values)
            y_values.append(y.iloc[i + seq_size])

        return np.array(x_values), np.array(y_values)

    trainX, trainY = to_sequences(train[[attr_1_col_name]], train[attr_1_col_name], seq_size)   #todo: double [[]] to return dataframe instead of sequence
    testX, testY = to_sequences(test[[attr_1_col_name]], test[attr_1_col_name], seq_size)

    model = get_trained_LSTM_Autoencder(trainX, trainY, "./models/pretty good performance on steering angle - fuckingSimpleLSTMAutoenc.keras")
    #model.compile(optimizer='adam', loss='mae')         #loss='mean_squared_error'
    #model.compile(optimizer='adam', loss='mean_squared_error')

    #todo: try adding val data as separate series: validation_data=(X_vN1, X_vN1)

    #model.summary()

    # fit model
    #todo:
    # note "validation_split=0.1" and batch_size; shuffle is true per default
    # batch_size = 32 too large; 4 works well
    #history = model.fit(trainX, trainY, epochs=20, batch_size=4, validation_split=0.1, shuffle=False, verbose=1)  #todo: probably need trainX istead of trainY
    #todo: changed trainY to trainX

    model.save("./models/fuckingSimpleLSTMAutoenc" + ".keras")

    # plt.plot(history.history['loss'], label='Training loss')
    # plt.plot(history.history['val_loss'], label='Validation loss')
    # plt.legend()
    # plt.show()

    # model.evaluate(testX, testY)

    ###########################
    # Anomaly is where reconstruction error is large.
    # We can define this value beyond which we call anomaly.
    # Let us look at MAE in training prediction

    trainPredict = model.predict(trainX)
    trainMAE = np.mean(np.abs(trainPredict - trainX), axis=1)
    plt.hist(trainMAE, bins=30)
    plt.show()

    max_trainMAE = 0.3  # or Define 90% value of max as threshold.


    testPredict = model.predict(testX)
    testMAE = np.mean(np.abs(testPredict - testX), axis=1)
    plt.hist(testMAE, bins=30)
    plt.show()

    # Capture all details in a DataFrame for easy plotting
    anomaly_df = pd.DataFrame(test[seq_size:])
    anomaly_df['testMAE'] = testMAE
    anomaly_df['max_trainMAE'] = max_trainMAE
    anomaly_df['anomaly'] = anomaly_df['testMAE'] > anomaly_df['max_trainMAE']
    anomaly_df[attr_1_col_name] = test[seq_size:][attr_1_col_name]

    # Plot testMAE vs max_trainMAE
    sns.lineplot(x=anomaly_df['Time'], y=anomaly_df['testMAE'])
    sns.lineplot(x=anomaly_df['Time'], y=anomaly_df['max_trainMAE'])
    plt.title('Test MAE')
    plt.show()

    #testPredict = model.predict(testX)
    testMSE = np.mean(np.square(testPredict - testX), axis=1)
    plt.hist(testMSE, bins=30)
    plt.show()

    # Capture all details in a DataFrame for easy plotting
    anomaly_df = pd.DataFrame(test[seq_size:])
    anomaly_df['testMSE'] = testMSE
    anomaly_df['max_trainMAE'] = max_trainMAE
    anomaly_df['anomaly'] = anomaly_df['testMSE'] > anomaly_df['max_trainMAE']  #todo: im mixing max_trainMAE with MSE so this might need to be adapted
    anomaly_df[attr_1_col_name] = test[seq_size:][attr_1_col_name]

    # Plot testMSE vs max_trainMAE
    sns.lineplot(x=anomaly_df['Time'], y=anomaly_df['testMSE'])
    sns.lineplot(x=anomaly_df['Time'], y=anomaly_df['max_trainMAE'])
    plt.title('Test MSE')
    plt.show()



    #todo: do Train data stats here

    trainPredict = model.predict(trainX)
    trainMAE = np.mean(np.abs(trainPredict - trainX), axis=1)
    plt.hist(trainMAE, bins=30)
    plt.show()

    anomaly_df = pd.DataFrame(train[seq_size:])
    anomaly_df['trainMAE'] = trainMAE
    anomaly_df['max_trainMAE'] = max_trainMAE
    anomaly_df['anomaly'] = anomaly_df['trainMAE'] > anomaly_df['max_trainMAE']
    anomaly_df[attr_1_col_name] = test[seq_size:][attr_1_col_name]

    # Plot testMAE vs max_trainMAE
    sns.lineplot(x=anomaly_df['Time'], y=anomaly_df['trainMAE'])
    sns.lineplot(x=anomaly_df['Time'], y=anomaly_df['max_trainMAE'])
    plt.title('Train MAE')
    plt.show()

    # testPredict = model.predict(testX)
    trainMSE = np.mean(np.square(trainPredict - trainX), axis=1)
    plt.hist(trainMSE, bins=30)
    plt.show()

    # Capture all details in a DataFrame for easy plotting
    anomaly_df = pd.DataFrame(train[seq_size:])
    anomaly_df['trainMSE'] = trainMSE
    anomaly_df['max_trainMAE'] = max_trainMAE
    anomaly_df['anomaly'] = anomaly_df['trainMSE'] > anomaly_df['max_trainMAE']  # todo: im mixing max_trainMAE with MSE so this might need to be adapted
    anomaly_df[attr_1_col_name] = test[seq_size:][attr_1_col_name]

    # Plot testMSE vs max_trainMAE
    sns.lineplot(x=anomaly_df['Time'], y=anomaly_df['trainMSE'])
    sns.lineplot(x=anomaly_df['Time'], y=anomaly_df['max_trainMAE'])
    plt.title('Train MSE')
    plt.show()


    anomalies = anomaly_df.loc[anomaly_df['anomaly'] == True]

    # Plot anomalies
    # sns.lineplot(x=anomaly_df['Time'], y=scaler.inverse_transform(anomaly_df['data']))
    # sns.scatterplot(x=anomalies['Time'], y=scaler.inverse_transform(anomalies['data']), color='r')


    sns.lineplot(x=anomaly_df['Time'], y=scaler.inverse_transform(anomaly_df[attr_1_col_name].drop(columns=['Time'], inplace=True)))
    sns.scatterplot(x=anomalies['Time'], y=scaler.inverse_transform(anomalies[attr_1_col_name].drop(columns=['Time'], inplace=True)), color='r')
    plt.show()


