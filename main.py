# This is a sample Python script.
from raw_data_processing import rosbag_file_conversion
from raw_data_processing.rosbag_file_conversion import stupid_encoding_error, read_file_to_csv_bagpy
from stacked_lstm import test_stacked_LSTM


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

if __name__ == '__main__':
    #rosbag_file_conversion.read_bag_to_csv('./aufnahmen/tmp/autocross_valid_16_05_23.bag', 'rosbagFileCSV.csv')
    #rosbag_file_conversion.read_File('./aufnahmen/tmp/skidpad_geschoben.bag')
    #rosbag_file_conversion.read_File('./aufnahmen/tmp/autocross_valid_16_05_23.bag')
    # stupid_encoding_error('./aufnahmen/tmp/autocross_valid_16_05_23.bag')
    #read_file_to_csv_bagpy('./aufnahmen/tmp/autocross_valid_16_05_23.bag')
    test_stacked_LSTM()
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
