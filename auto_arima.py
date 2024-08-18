from data_processing import get_normalized_data_and_labels, get_flattened_single_column_from_nd_nparray, csv_files_to_df
from utils import get_matching_file_pairs_from_directories
from pmdarima.utils import tsdisplay


def run_auto_arima(directories):
    scaler = None

    all_file_pairs = get_matching_file_pairs_from_directories(directories, "can_interface-wheelspeed.csv")
    print("all_file_pairs: " + str(all_file_pairs))

    dfs = []
    for file_pair in all_file_pairs:
        for single_file in file_pair:
            df = csv_files_to_df(single_file, remove_timestamps=False)
            print("df: " + str(df))
            print("df head: " + str(df.head()))
            dfs.append(df)

        # data_with_time_diffs, true_labels_list = get_normalized_data_and_labels(file_pair, scaler, remove_timestamps=False)
        # #plot_data(data_with_time_diffs[0])
        # #if list is empty due to excluded csv file
        # if not data_with_time_diffs:
        #     continue
        # data_with_time_diffs = get_flattened_single_column_from_nd_nparray(data_with_time_diffs, 1)

    # print("data_with_time_diffs: " + str(data_with_time_diffs))
    # print("data_with_time_diffs[0]: " + str(data_with_time_diffs[0]))
    # print("data_with_time_diffs[1]: " + str(data_with_time_diffs[1]))
    #
    # y_train = data_with_time_diffs[0]
    # tsdisplay(y_train, lag_max=100)
    # print(y_train.head())
