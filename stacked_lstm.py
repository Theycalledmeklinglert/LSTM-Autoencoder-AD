import numpy as np
# from keras.models import Sequential
# from keras.layers import LSTM, Input, Dropout
# from keras.layers import Dense
# from keras.layers import RepeatVector
# from keras.layers import TimeDistributed
import pandas as pd
from keras import Sequential
from keras.src.layers import LSTM, Dropout, RepeatVector, TimeDistributed, Dense
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
#from keras.models import Model
import seaborn as sns

from data_processing import clean_csv

if __name__ == '__main__':


    #dataframe = pd.read_csv('data/GE.csv')
    #df = dataframe[['Date', 'Close']]
    #df['Date'] = pd.to_datetime(df['Date'])

    df0 = clean_csv("./aufnahmen/csv/skidpad_valid_fast2_17_47_28/can_interface-current_steering_angle.csv", False)
    df1 = clean_csv("./aufnahmen/csv/anomalous data/can_interface-current_steering_angle.csv", False)

    df0.drop(columns=["Anomaly"], inplace=True)
    print(df0.head())
    sns.lineplot(x=df0['Time'], y=df0['data'])
    plt.title('Orig Train')
    plt.show()
    df1.drop(columns=["Anomaly"], inplace=True)
    print(df1.head())
    sns.lineplot(x=df1['Time'], y=df1['data'])
    plt.title('Orig Test')
    plt.show()

    print("Start train data date is: ", df0['Time'].min())
    print("End train data date is: ", df0['Time'].max())

    # Change train data from Mid 2017 to 2019.... seems to be a jump early 2017
    #train, test = df.loc[df['Date'] <= '2003-12-31'], df.loc[df['Date'] > '2003-12-31'] #todo: original. Uses trainX to predict last part of itself (testX)
    train, test = df0, df1          #todo: original. Uses trainX to predict last part of itself (testX)


    train['data'] = train['data'].diff().fillna(0)          #todo: EXPERIMENTAL
    test['data'] = test['data'].diff().fillna(0)            #todo: EXPERIMENTAL

    sns.lineplot(x=train['Time'], y=train['data'])
    plt.title('Diffed Train')
    plt.show()
    print(df1.head())
    sns.lineplot(x=test['Time'], y=test['data'])
    plt.title('Diffed Test')
    plt.show()

    print("train: ", train)
    print("test: ", test)

    # Convert pandas dataframe to numpy array
    # dataset = dataframe.values
    # dataset = dataset.astype('float32') #COnvert values to float

    # LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
    # normalize the dataset
    # scaler = MinMaxScaler() #Also try QuantileTransformer
    scaler = StandardScaler()           #todo: MaxAbsScaler() might be smart for large val ranges at end of the series'
    #scaler = MaxAbsScaler()
    #scaler = RobustScaler()
    # todo: could also try robust scaler


    scaler = scaler.fit(train[['data']])

    train['data'] = scaler.transform(train[['data']]) #todo: NOTE that he doesnt use fit_transform as "Time" is still in df
    test['data'] = scaler.transform(test[['data']])

    print("scaled train: ", train)
    print("scaled test: ", test)

    # As required for LSTM networks, we require to reshape an input data into n_samples x timesteps x n_features.
    # In this example, the n_features is 2. We will make timesteps = 3.
    # With this, the resultant n_samples is 5 (as the input data has 9 rows).

    seq_size = 30  # Number of time steps to look back


    # Larger sequences (look further back) may improve forecasting.


    def to_sequences(x, y, seq_size=1):
        x_values = []
        y_values = []

        for i in range(len(x) - seq_size):
            # print(i)
            x_values.append(x.iloc[i:(i + seq_size)].values)
            y_values.append(y.iloc[i + seq_size])

        return np.array(x_values), np.array(y_values)


    trainX, trainY = to_sequences(train[['data']], train['data'], seq_size)   #todo: double [[]] to return dataframe instead of sequence
    testX, testY = to_sequences(test[['data']], test['data'], seq_size)

    # define Autoencoder model
    # Input shape would be seq_size, 1 - 1 beacuse we have 1 feature.
    # seq_size = trainX.shape[1]

    model = Sequential()
    model.add(LSTM(128, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
    model.add(LSTM(64, activation='relu', return_sequences=False))

    model.add(Dropout(rate=0.2))

    model.add(RepeatVector(trainX.shape[1]))
    model.add(LSTM(64, activation='relu', return_sequences=True))
    model.add(LSTM(128, activation='relu', return_sequences=True))

    model.add(Dropout(rate=0.2))

    model.add(TimeDistributed(Dense(trainX.shape[2])))

    # model.compile(optimizer='adam', loss='mse')
    # model.summary()

    # Try another model
    # model = Sequential()
    # model.add(LSTM(128, input_shape=(trainX.shape[1], trainX.shape[2])))    #todo: went from 128 to 256
    # model.add(Dropout(rate=0.2))
    # model.add(RepeatVector(trainX.shape[1]))
    # model.add(LSTM(128, return_sequences=True))
    # model.add(Dropout(rate=0.2))
    # model.add(TimeDistributed(Dense(trainX.shape[2])))                           #todo: went from 128 to 256

    #model.compile(optimizer='adam', loss='mae')         #loss='mean_squared_error'
    model.compile(optimizer='adam', loss='mean_squared_error')

    #todo: try adding val data as separate series: validation_data=(X_vN1, X_vN1)

    model.summary()

    # fit model
    #todo: note "validation_split=0.1" and batch_size; shuffle is true per default
    history = model.fit(trainX, trainY, epochs=20, batch_size=4, validation_split=0.1, shuffle=False ,verbose=1)  #todo: probably need trainX istead of trainY
    #todo: changed trainY to trainX

    model.save("./models/fuckingSimpleLSTMAutoenc" + ".keras")

    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.legend()
    plt.show()

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
    anomaly_df['data'] = test[seq_size:]['data']

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
    anomaly_df['data'] = test[seq_size:]['data']

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
    anomaly_df['data'] = test[seq_size:]['data']

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
    anomaly_df['data'] = test[seq_size:]['data']

    # Plot testMSE vs max_trainMAE
    sns.lineplot(x=anomaly_df['Time'], y=anomaly_df['trainMSE'])
    sns.lineplot(x=anomaly_df['Time'], y=anomaly_df['max_trainMAE'])
    plt.title('Train MSE')
    plt.show()




    anomalies = anomaly_df.loc[anomaly_df['anomaly'] == True]

    # Plot anomalies
    # sns.lineplot(x=anomaly_df['Time'], y=scaler.inverse_transform(anomaly_df['data']))
    # sns.scatterplot(x=anomalies['Time'], y=scaler.inverse_transform(anomalies['data']), color='r')

    sns.lineplot(x=anomaly_df['Time'], y=scaler.inverse_transform(anomaly_df['data']).values.reshape(-1, 1).flatten())
    sns.scatterplot(x=anomalies['Time'], y=scaler.inverse_transform(anomalies['data'].values.reshape(-1, 1).flatten()), color='r')

    plt.show()