# This is a sample Python script.
from aufnahmen.enc_dec_lstm import test_LSTM_autoencoder, grid_search_LSTM_autoencoder
from raw_data_processing import data_processing
from raw_data_processing.data_processing import read_file_to_csv_bagpy, csv_file_to_dataframe_to_numpyArray
from stacked_lstm import test_stacked_LSTM, create_XY_data_sequences, generate_test_array


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

if __name__ == '__main__':
    #rosbag_file_conversion.read_File('./aufnahmen/tmp/skidpad_geschoben.bag')
    #rosbag_file_conversion.read_File('./aufnahmen/tmp/autocross_valid_16_05_23.bag')
    # stupid_encoding_error('./aufnahmen/tmp/autocross_valid_16_05_23.bag')
    #read_file_to_csv_bagpy('./aufnahmen/tmp/autocross_valid_16_05_23.bag')
    #test_stacked_LSTM()
    #samples_arr = csv_file_to_dataframe_to_numpyArray("./aufnahmen/csv/autocross_valid_16_05_23/can_interface-current_steering_angle.csv")
    #X, Y = create_XY_data_sequences(samples_arr, 3, 3)
    #X, Y = create_XY_data_sequences(generate_test_array(), 3, 3)
    #print("X: " + str(X[:3:]))
    #print("Y: " + str(Y[:3:]))

    #test_stacked_LSTM("./aufnahmen/csv/autocross_valid_16_05_23/can_interface-current_steering_angle.csv")
    #test_LSTM_autoencoder("./aufnahmen/csv/autocross_valid_16_05_23/can_interface-wheelspeed.csv")
    grid_search_LSTM_autoencoder("./aufnahmen/csv/autocross_valid_16_05_23/can_interface-wheelspeed.csv")
    print_hi('PyCharm')

