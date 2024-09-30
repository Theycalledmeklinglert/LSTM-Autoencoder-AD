# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import tensorflow as tf
from pmdarima.datasets import load_sunspots
from tensorflow import keras

from anomaly_and_CD_injection import add_anomalies_and_drift
from auto_arima import run_auto_arima
from data_processing import csv_file_to_nparr, \
    old_directory_csv_files_to_dataframe_to_numpyArray, read_file_from_bagpy_to_csv, plot_data_integrated, \
    plot_data_standalone, plot_acf, plot_acf_standalone
from exampleGraphs.example_plot_generator import plot_contextual_anomaly, \
    plot_collective_anomaly_similar, plot_point_anomaly
from exampleGraphs.normal_vs_noisy_data import plot_normal_vs_noisy_data, plot_scatter_normal_vs_noisy, \
    plot_clusters_with_noise
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

    #run_auto_arima(["./aufnahmen/csv/autocross_valid_16_05_23", "./aufnahmen/csv/autocross_valid_run"], "can_interface-current_steering_angle.csv") # ,"can_interface-wheelspeed.csv")

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

    #add_anomaly_column_to_csv_files(["./aufnahmen/csv/skidpad_valid_fast2_17_47_28", "./aufnahmen/csv/skidpad_valid_fast3_17_58_41", "./aufnahmen/csv/test data/skidpad_falscher_lenkungsoffset"])

    #test_lstm_autoencoder(20, [60], 0.0, 32, 500, True, ["./aufnahmen/csv/autocross_valid_16_05_23", "./aufnahmen/csv/autocross_valid_run", "./aufnahmen/csv/anomalous data", "./aufnahmen/csv/autocross_valid2_17_23_44"], "can_interface-wheelspeed.csv") # , "./models/LSTM_autoencoder_decoder_can_interface-wheelspeed_timesteps20_layers_60.keras")

    #before_injection_for_plot, true_labels = csv_file_to_nparr('./aufnahmen/csv/anomalous data/normal_control-acceleration.csv', True)

    #TODO: idea for add_anomalies_and_drift: Only label row as anomaly if the resulting value of new value is greating than 3*sigma of all data in column or sth.
    # oviously not all data that comes from concept drift is supposed to be labels as anomalies and my current add_anomalies often changes values way too litlle

    #add_anomalies_and_drift(100,'./aufnahmen/csv/anomalous data/normal_control-acceleration.csv', './aufnahmen/csv/anomalous data/control-acceleration.csv')
    #after_injection_for_plot, true_labels = csv_file_to_nparr('./aufnahmen/csv/anomalous data/control-acceleration.csv', True)
    #plot_data(before_injection_for_plot, str("before_injection_for_plot"))
    #plot_data(after_injection_for_plot, str("after_injection_for_plot"))

    #todo: this was commented in:
    # try droput 0.001; try with remove_timestamps and without; try with higher timesteps

    #plot_acf_standalone(["./aufnahmen/csv/autocross_valid_16_05_23", "./aufnahmen/csv/autocross_valid_run", "./aufnahmen/csv/skidpad_valid_fast2_17_47_28", "./aufnahmen/csv/test data/skidpad_falscher_lenkungsoffset"], "control-acceleration.csv")
    #plot_data_standalone(["./aufnahmen/csv/autocross_valid_16_05_23", "./aufnahmen/csv/autocross_valid_run", "./aufnahmen/csv/skidpad_valid_fast2_17_47_28", "./aufnahmen/csv/test data/skidpad_falscher_lenkungsoffset"], "can_interface-current_steering_angle.csv")

    test_lstm_autoencoder(40, [150], 0.000, 8, 200, 0.4, True, True, ["./aufnahmen/csv/autocross_valid_16_05_23", "./aufnahmen/csv/autocross_valid_run", "./aufnahmen/csv/anomalous data", "./aufnahmen/csv/test data/autocross_unbekannter_kommunikationsfehler"], "can_interface-wheelspeed.csv") #, "./models/LSTM_autoencoder_decoder_can_interface-wheelspeed_timesteps40_layers_60.keras")
    #test_lstm_autoencoder(20, [80], 0.0, 32, 200, 0.9, True, True, ["./aufnahmen/csv/skidpad_valid_fast2_17_47_28", "./aufnahmen/csv/skidpad_valid_fast3_17_58_41", "./aufnahmen/csv/anomalous data", "./aufnahmen/csv/test data/skidpad_falscher_lenkungsoffset"], "can_interface-current_steering_angle.csv") #, "./models/LSTM_autoencoder_decoder_can_interface-current_steering_angle_timesteps40_layers_50.keras")
    #test_lstm_autoencoder(20, [60], 0.0, 64, 500, 0.9,  True, False, ["./aufnahmen/csv/autocross_valid_16_05_23", "./aufnahmen/csv/autocross_valid_run", "./aufnahmen/csv/anomalous data", "./aufnahmen/csv/autocross_valid2_17_23_44"], "control-acceleration.csv", "./models/LSTM_autoencoder_decoder_control-acceleration_timesteps20_layers_60.keras")
    #test_lstm_autoencoder(20, [30], 0.0, 64, 500, 0.9,  True, True, ["./aufnahmen/csv/autocross_valid_16_05_23", "./aufnahmen/csv/autocross_valid_run", "./aufnahmen/csv/anomalous data", "./aufnahmen/csv/autocross_valid2_17_23_44"], "estimation-velocity.csv") #, "./models/LSTM_autoencoder_decoder_control-acceleration_timesteps20_layers_30.keras")

    #plot_point_anomaly()
    #plot_collective_anomaly_similar()
    #plot_clusters_with_noise()
    #plot_contextual_anomaly()


    #df = clean_csv("C:\\Users\\Luca\\PycharmProjects\AnoamlydetectionInFormulaStudent\\aufnahmen\csv\\autocross_valid_16_05_23\\diagnostics.csv")
    #print_unique_values(df, "status")

    #read_file_from_bagpy_to_csv("./aufnahmen/aufnahmen2/skidpad_valid_fast3_17_58_41.bag")
    #read_file_from_bagpy_to_csv("./aufnahmen/error_zusammenstellung/error_zusammenstellung/autocross_unbekannter_kommunikationsfehler.bag")



    print_hi('PyCharm')

    #todo: Note if I want to change the amount of timesteps, a new model has to be trained on it

