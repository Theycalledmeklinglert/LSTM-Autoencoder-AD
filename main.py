# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import tensorflow as tf
from tensorflow import keras

from raw_data_processing.data_processing import read_file_to_csv_bagpy
from tf_lstm_autoencoder import test_lstm_autoencoder


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

if __name__ == '__main__':
    # stupid_encoding_error('./aufnahmen/tmp/autocross_valid_16_05_23.bag')
    #read_file_to_csv_bagpy('./aufnahmen/tmp/autocross_valid_16_05_23.bag')
    #test_stacked_LSTM()
    #samples_arr = csv_file_to_dataframe_to_numpyArray("./aufnahmen/csv/autocross_valid_16_05_23/can_interface-current_steering_angle.csv")
    #X, Y = create_XY_data_sequences(samples_arr, 3, 3)
    #X, Y = create_XY_data_sequences(generate_test_array(), 3, 3)
    #print("X: " + str(X[:3:]))
    #print("Y: " + str(Y[:3:]))

    #test_stacked_LSTM("./aufnahmen/csv/autocross_valid_16_05_23/can_interface-current_steering_angle.csv")
    #old_LSTM_autoencoder("./aufnahmen/csv/autocross_valid_16_05_23/can_interface-wheelspeed.csv")
    #grid_search_LSTM_autoencoder("./aufnahmen/csv/autocross_valid_16_05_23/can_interface-wheelspeed.csv")


    print(tf.__version__)
    print(keras.__version__)

    #test_lstm_autoencoder(10, [30, 30], 2, 0.0, 32, 120, ["./aufnahmen/csv/autocross_valid_16_05_23/can_interface-wheelspeed.csv", "./aufnahmen/csv/autocross_valid2_17_23_44/can_interface-wheelspeed.csv"])
    test_lstm_autoencoder(10, [30, 30], 2, 0.0, 32, 120, ["./aufnahmen/csv/autocross_valid_16_05_23/can_interface-wheelspeed.csv", "./aufnahmen/csv/autocross_valid2_17_23_44/can_interface-wheelspeed.csv"], './models/pretty good autoencoder for wheel speed/LSTM_autoencoder_decoder_30_30.keras')
    print_hi('PyCharm')

    #todo: Note if I want to change the amount of timesteps, a new model has to be trained on it

