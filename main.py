# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import tensorflow as tf
from pmdarima.datasets import load_sunspots
from tensorflow import keras

from auto_arima import run_auto_arima
from data_processing import csv_file_to_nparr, \
    old_directory_csv_files_to_dataframe_to_numpyArray, read_file_to_csv_bagpy
from tf_lstm_autoencoder import test_lstm_autoencoder
from utils import add_anomaly_column_to_csv_files
import pmdarima as pm


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

if __name__ == '__main__':

    #test_stacked_LSTM("./aufnahmen/csv/autocross_valid_16_05_23/can_interface-current_steering_angle.csv")
    #old_LSTM_autoencoder("./aufnahmen/csv/autocross_valid_16_05_23/can_interface-wheelspeed.csv")
    #grid_search_LSTM_autoencoder("./aufnahmen/csv/autocross_valid_16_05_23/can_interface-wheelspeed.csv")


    print(tf.__version__)
    print(keras.__version__)

    wineind = pm.datasets.load_wineind()

    print(type(wineind))

    y = load_sunspots(True)
    print(y.head())
    print(type(y))

    #run_auto_arima(["./aufnahmen/csv/autocross_valid_16_05_23", "./aufnahmen/csv/autocross_valid_run"])

    #todo:
    # tsdisplay(y_train, lag_max=100)
    # fit2 = Pipeline([
    #     ('boxcox', BoxCoxEndogTransformer(lmbda2=1e-6)),
    #     ('arima', pm.AutoARIMA(trace=True,
    #                            suppress_warnings=True,
    #                            m=12))
    # ])
    # fit2.fit(y_train)


    #get_sample_time("./aufnahmen/csv/autocross_valid_16_05_23")

    #test_lstm_autoencoder(10, [50, 50], 0.0, 32, 200, ["./aufnahmen/csv/csv test 1", "./aufnahmen/csv/csv test 2"]) #"C:\\Users\\Luca\\PycharmProjects\\AnoamlydetectionInFormulaStudent\\models\\LSTM_autoencoder_decoder_can_interface-wheelspeed_100_100.keras"

    #add_anomaly_column_to_csv_files(["./aufnahmen/csv/autocross_valid_16_05_23", "./aufnahmen/csv/autocross_valid_run", "./aufnahmen/csv/anomalous data", "./aufnahmen/csv/autocross_valid2_17_23_44"])

    #todo: I think either saving or loading the model is not saving/loading the weights or internal states correctly....FUCKKKKKKK
    #test_lstm_autoencoder(150, [20], 0.0, 64, 500, ["./aufnahmen/csv/autocross_valid_16_05_23", "./aufnahmen/csv/autocross_valid_run", "./aufnahmen/csv/anomalous data", "./aufnahmen/csv/autocross_valid2_17_23_44"], "can_interface-current_steering_angle.csv", "./models/LSTM_autoencoder_decoder_can_interface-current_steering_angle_timesteps150_layers_20.keras")
    #test_lstm_autoencoder(20, [60], 0.0, 32, 500, ["./aufnahmen/csv/autocross_valid_16_05_23", "./aufnahmen/csv/autocross_valid_run", "./aufnahmen/csv/anomalous data", "./aufnahmen/csv/autocross_valid2_17_23_44"], "can_interface-wheelspeed.csv", "./models/LSTM_autoencoder_decoder_can_interface-wheelspeed_timesteps20_layers_60.keras")
    test_lstm_autoencoder(20, [40], 0.0, 32, 500, ["./aufnahmen/csv/autocross_valid_16_05_23", "./aufnahmen/csv/autocross_valid_run", "./aufnahmen/csv/anomalous data", "./aufnahmen/csv/autocross_valid2_17_23_44"], "control-acceleration.csv", "./models/LSTM_autoencoder_decoder_control-acceleration_timesteps20_layers_40.keras")

    #df = clean_csv("C:\\Users\\Luca\\PycharmProjects\AnoamlydetectionInFormulaStudent\\aufnahmen\csv\\autocross_valid_16_05_23\\diagnostics.csv")
    #print_unique_values(df, "status")

    #read_file_to_csv_bagpy("./aufnahmen/tmp/autocross_valid_run.bag")
    #error_vecs = np.random.rand(300, 100, 5)  # 300 samples, 100 timesteps, 5 features each
    #print(str(error_vecs))
    # Flatten the error_vecs to shape (samples * timesteps, features)
    # flattened_error_vecs = error_vecs.reshape(-1, 5)
    # print("Flattened error vecs1: \n", flattened_error_vecs)
    # flattened_error_vecs2 = error_vecs.reshape(error_vecs.shape[0], error_vecs.shape[1] * error_vecs.shape[2])
    # print("Flattened error vecs2: \n", flattened_error_vecs2)

    print_hi('PyCharm')

    #todo: Note if I want to change the amount of timesteps, a new model has to be trained on it

