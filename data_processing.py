import csv
import glob
import os
import traceback

import numpy
import rosbag
import genpy
import pandas as pd
from bagpy import bagreader
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MaxAbsScaler
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox


def get_normalized_data_and_labels(file_pair, scaler, factor, remove_timestamps):
    data_with_time_diffs = []
    true_labels_list = []
    print("Now training model on: " + str(file_pair[0][file_pair[0].rfind("\\") + 1:].rstrip(".csv")))

    for single_file in file_pair:
        data, true_labels = csv_file_to_nparr(single_file, remove_timestamps, factor) # timestamps are transformed to relative timestamps
        if data is None:
            break

        print("here: " + str(data.shape))

        data = data[:, -1:]

        print("here: " + str(data.shape))

        plot_data_integrated(data, single_file, not remove_timestamps)
        print("unnormalized_data_with_time_diffs: \n" + str(data))
        normalized_data = normalize_data(data, scaler)
        print("normalized_data_with_time_diffs: \n" + str(normalized_data))
        data_with_time_diffs.append(normalized_data)
        print("Anoamaly labels: \n" + str(true_labels))
        true_labels_list.append(true_labels)
    return data_with_time_diffs, true_labels_list


# def transform_true_labels_to_window_size(true_labels_list):
#     windowed_true_labels = []
#     for true_labels in true_labels_list:
#         cur_window = []
#         for labels_of_window in true_labels:
#             if 1 in labels_of_window:
#                 cur_window.append(1)
#             else:
#                 cur_window.append(0)
#         windowed_true_labels.append(np.asarray(cur_window))
#     return windowed_true_labels


def reshape_data_for_autoencoder_lstm(data_list, window_size, window_step):
    # Reshape X to fit LSTM input shape (samples, time steps, features)
    if window_size < 1 or window_step < 0 or window_step >= window_size:
        from utils import InvalidReshapeParamters
        raise InvalidReshapeParamters()
        return

    if window_step == 0:
        window_step = window_size
    windowed_data = []
    for i in range(len(data_list)):
        data = data_list[i]
        temp = []

        for window_start in range(0, data.shape[0] - window_size, window_step):
            window_end = window_start + window_size
            cur_window = data[window_start:window_end]
            temp.append(cur_window)

        windowed_data.append(numpy.array(temp))

        # print("Reshaped data of single file into: " + str(windowed_data[0]))
        # print("Reshaped data of single file into: " + str(windowed_data[1]))
        # print("First single: " + str(windowed_data[0].shape))
        # print("Second single: " + str(windowed_data[0].shape))
        #
        # for k in range(len(data_list[i])):
        #     print("Length at " + str(k) + ": " + str(data_list[i][k].shape))
        #     if k == len(data_list[i])-1:
        #         print(str(data_list[i][k]))
        #
        # print("All: " + str(numpy.array(windowed_data).shape))
        # print("Reshaped data for LSTM into: " + str(numpy.array(windowed_data)))

    return windowed_data


def split_data_sequence_into_datasets(arr, train_ratio, val1_ratio, val2_ratio, test_ratio):
    assert (train_ratio * 10 + val1_ratio * 10 + val2_ratio * 10 + test_ratio * 10) == 10  # due to float round error

    n_total = len(arr)
    n_train = int(train_ratio * n_total)
    n_val1 = int(val1_ratio * n_total)
    n_val2 = int(val2_ratio * n_total)
    n_test = n_total - n_train - n_val1 - n_val2  # To ensure all samples are used

    # Split data sequentially
    sN = arr[:n_train]
    vN1 = arr[n_train:n_train + n_val1]
    vN2 = arr[n_train + n_val1:n_train + n_val1 + n_val2]
    tN = arr[n_train + n_val1 + n_val2:n_train + n_val1 + n_val2 + n_test]
    print(f"Training set size: {len(sN)}")
    print(f"Validation set 1 size: {len(vN1)}")
    print(f"Validation set 2 size: {len(vN2)}")
    print(f"Test set size: {len(tN)}")

    # print("sN df: " + str(sN))
    # print("vN1 df: " + str(vN1))
    # print("vN2 df: " + str(vN2))
    # print("tN df: " + str(tN))

    # return sN, vN1, vN2, tN
    return sN, vN1, vN2, tN


def reshape_data_for_LSTM(data, steps):
    # Reshape X to fit LSTM input shape (samples, time steps, features)
    # print(data.shape)
    data = data.reshape((data.shape[0], steps, data.shape[2]))
    print("Reshaped data for LSTM into: " + str(data))
    return data


def check_shapes_after_reshape(X_sN, X_vN2, X_tN, Y_sN, Y_vN2, Y_tN):
    shapes = [X_sN.shape, X_vN2.shape, X_tN.shape, Y_sN.shape, Y_vN2.shape, Y_tN.shape]
    print("Shapes of arrays after reshaping:")
    for i, shape in enumerate(shapes):
        print(f"Array {i + 1}: {shape}")

    # Check if all arrays have the same shape in terms of time_steps and features
    try:
        shape_to_compare = (shapes[0][1], shapes[0][2])
        if not all((shape[1], shape[2]) == shape_to_compare for shape in shapes):
            raise ValueError("Shapes of reshaped arrays are not consistent in terms of time_steps and features!")
    except ValueError as e:
        print(f"Error: {str(e)}")


def normalize_data(data, scaler):
    if scaler is None:
        return data
    return scaler.fit_transform(data)


def reverse_normalization(scaled_data, scaler):
    if scaler is None:
        return scaled_data
    return scaler.inverse_transform(scaled_data)


def shuffle_data(data, true_labels=None):
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    if true_labels is not None:
        true_labels = true_labels[indices]

    # idxsss = np.where(true_labels_list[1] == 1)
    # print("?" + str(idxsss))
    # [print("hey1: " + str(reverse_normalize_data(seq, scaler)) + "\n") for seq in
    #  data_with_time_diffs[1][np.unique(idxsss[0])]]
    # print("hey4: \n" + str(data_with_time_diffs[1][np.unique(idxsss[0])]))

    return data, true_labels


def convert_timestamp_to_relative_time_diff(df):
    time_columns = [col for col in df.columns if col == "Time"]
    # Subtract each value in "Time" columns from the first row's value

    for time_col in time_columns:
        #start_timestamp = df[time_col].iloc[0]
        #df[time_col] = df[time_col] - start_timestamp
        # Replace the "Time" column with the row index
        df[time_col] = df.index     #todo: Experimental

    return df


def csv_file_to_nparr(file_path, remove_timestamps, factor):
    print("Getting data from: " + str(file_path))

    offset = 0
    df = clean_csv(file_path, remove_timestamps)
    if df is None:
        return

    print("cleaned csv")

    df = convert_timestamp_to_relative_time_diff(df)
    if "Anomaly" in df.columns:
        offset = 1

    print("converted timestamps")
    print(df.shape)

    samples = np.zeros((df.shape[0], df.shape[1] - offset))
    print(samples.shape)
    true_labels = np.zeros((df.shape[0], 1))
    for row_index, row in df.iterrows():
        #print("going through row: " + str(row_index))
        for col_index, column in enumerate(df.columns):
            if column == "Anomaly":
                true_labels[row_index] = row[column].astype('int')  # 0 or 1; 1=anomaly
            else:
                samples[row_index, col_index] = row[column] * factor
                #print("changed from: " + str(row[column]) + " to: " + str(row[column] * 10))
            # print("row_index: " + str(row_index))
            # print("column: " + str(column) + " + col_index: " + str(col_index))
            # print("grabbed: " + str(row[column]))
    return samples, true_labels


def csv_files_to_df(file_path, remove_timestamps):
    df = clean_csv(file_path, remove_timestamps)
    if df is None:
        return
    df = convert_timestamp_to_relative_time_diff(df)
    return df

def get_flattened_single_column_from_nd_nparray(list_of_data, col_idx):
    for idx, data in enumerate(list_of_data):
        list_of_data[idx] = data[:, :col_idx].flatten()
    return list_of_data


def old_directory_csv_files_to_dataframe_to_numpyArray(directory):
    print("Reading files from directory: " + directory)
    dims = [0, 0]
    dfs = []

    for file in get_csv_file_paths(directory):
        df = clean_csv(file)

        if df is not None:
            print("now processing: " + str(file))
            print("cleaned df: \n" + str(df))
            print("shape: " + str(df.shape))
            dims[0] = dims[0] + df.shape[0]
            dims[1] = dims[1] + df.shape[1]
            df = convert_timestamp_to_relative_time_diff(df)
            dfs.append(df)

    samples = np.zeros((dims[0], dims[1]))

    # todo: csv's have different number of rows --> need to fill the rest of them with 0s i guess?
    # todo: OR: downsample my data to equal length but that's needlessly complicated too
    # todo: might be the only viable option cause otherwise I will fuck up my training
    col_offset = 0
    for df in dfs:
        for row_index, row in df.iterrows():
            for col_index, column in enumerate(df.columns):
                samples[row_index, col_offset + col_index] = row[column]
                # print("row_index: " + str(row_index))
                # print("column: " + str(column) + " + col_index: " + str(col_index))
                # print("grabbed: " + str(row[column]))
        col_offset = col_offset + df.shape[1]

    return samples


# removes empty columns, columns containing only one and the same value through and through and columns that do not offer useful information
def clean_csv(file_path, remove_timestamps=False, nrows=None):
    # list of known incompatible csv files due to incompatible/incorrect formatting or files that are simply are not sensor data
    exceptions = [
        "slam-car_state", "slam-landmark_info", "slam-map",
        "slam-state", "stereo_cone_perception-cones", "tf", "estimation-velocity", "lidar-cone_position_cloud",
        "map_matching-driving_path", "sbg-gps_raw",
        "map_matching-reference_track", "diagnostics"
    ]
    if any(keyword in file_path for keyword in exceptions):
        print("Intentionally did not process: " + str(file_path))
        return

    df = pd.read_csv(file_path, nrows=nrows, ) #, header=None)

    columns_to_remove = ['header.stamp.secs', 'header.stamp.nsecs', 'header.frame_id', 'child_frame_id',
                         'twist.covariance',
                         'header.seq']  # ", 'header.seq'"

    if remove_timestamps:
        columns_to_remove.append('Time')

    for col in columns_to_remove:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Remove columns that contain only the column name
    if df.isnull().values.any():  # True if there are NaNs, False otherwise
        from utils import DataFrameContainsNaNError
        raise DataFrameContainsNaNError()

    cols_to_check = [col for col in df.columns if col != "Anomaly"]
    #print("Columns in DataFrame:", df.columns.tolist())
    #print("Columns to check:", cols_to_check)
    #print("df: " + str(df))

    if not cols_to_check:
        print("No columns to check after filtering.")
    else:
        number_of_valid_cols = len(df.columns)
        # Drop columns with all missing values, excluding "Anomaly"
        print(type(cols_to_check))
        df.dropna(axis=0, how='all', subset=cols_to_check, inplace=True) # i don't know why this works since axis=0 is supposed to be deleting rows and axis=1 to delete columns. However axis=1 crashes for some reason
                                                                         # and axis=0 works and removes only the 0 columns so I guess I'll just leave it here for now if it doesnt cause any issues
        if number_of_valid_cols != len(df.columns):
            from utils import SensorFileColumnsContainsOnlyZeroesError
            print("Sensor file contains NaN. Check for malfunction in " + str(file_path))
            raise SensorFileColumnsContainsOnlyZeroesError()

        # Remove columns that only contain one and the same value, excluding "Anomaly"
        for col in cols_to_check:
            if df[col].nunique() == 1:
                df.drop(columns=[col], inplace=True)

        try:
            if number_of_valid_cols != len(df.columns):
                from utils import SensorFileColumnsOnlyContainsSameValue
                print("Sensor file contains NaN. Check for malfunction in ")
                raise SensorFileColumnsOnlyContainsSameValue()
        except SensorFileColumnsOnlyContainsSameValue as e:
            print("A column in the sensor file contains only one and the same value. Check for malfunction in " + str(file_path))


    print("df after removing nan and 0 cols: \n" + str(df))

    #print("filepath: " + str(file_path))

    return df


def get_csv_file_paths(directory):
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    return csv_files


def get_matching_file_pairs_from_directories(directories, file_name=None):
    # Ensure there are at least two directories to compare
    if len(directories) < 2:
        raise ValueError("At least two directories are required.")

    if file_name:
        for directory in directories:
            if file_name not in os.listdir(directory):
                return []

        matching_files = [[os.path.join(directory, file_name) for directory in directories]]
        print(str(matching_files))
    else:
        # List all files in the first directory
        common_files = set(os.listdir(directories[0]))

        # Find the intersection of files across all directories
        for directory in directories[1:]:
            common_files.intersection_update(os.listdir(directory))

        # Collect the full paths of matching files from each directory
        matching_files = []
        for file in common_files:
            file_paths = [os.path.join(directory, file) for directory in directories]
            matching_files.append(file_paths)

    return matching_files


def print_unique_values(df, column_name):
    if column_name in df.columns:
        unique_values = df[column_name].unique()
        print(f"Unique values in column '{column_name}':")
        for value in unique_values:
            print(value)
    else:
        print(f"Column '{column_name}' does not exist in the dataframe.")


# Sample time of approximately 10 milliseconds for wheelspeed

def get_sample_time(directory):
    all_time_diffs = {}
    for file in get_csv_file_paths(directory):
        time_diffs = []
        prev_timestamp = 0
        df = clean_csv(file)

        if df is None:
            continue

        for i in df.index:
            curr_timestamp = df.iloc[i]["Time"]
            time_diffs.append(curr_timestamp - prev_timestamp)
            prev_timestamp = curr_timestamp
            # if i == 100:
            #    break

        time_diffs = [float(value) / 1e6 for value in time_diffs]  # convert to msecs
        all_time_diffs[file] = time_diffs

    for entry in all_time_diffs.keys():
        print("Sample time differences for " + entry[entry.rfind("/") + 1:] + ": " + str(
            sum(all_time_diffs[entry]) / len(all_time_diffs[entry])) + " msecs")

    for entry in all_time_diffs.keys():
        print("length of " + entry[entry.rfind("/") + 1:] + " :" + str(len(all_time_diffs[entry])))

    # t is more accurate
    # print("t: " + str(t))
    # combined_nanoseconds = t.secs * 1e9 + t.nsecs
    # print("Combined secs and nsecs: " + str(combined_nanoseconds))


def read_file_from_bagpy_to_csv(path):
    b = bagreader(path)
    csvfiles = []
    for topic in b.topics:
        if (topic =="/sbg/imu_data"):   #causes UTF-8 encoding error
            continue
        print(topic)
        data = b.message_by_topic(topic)
        print(data)
        csvfiles.append(data)

    print(csvfiles[0])
    data = pd.read_csv(csvfiles[0])

    print(csvfiles[1])
    data = pd.read_csv(csvfiles[1])

    print(csvfiles[2])
    data = pd.read_csv(csvfiles[2])


def plot_data_integrated(data, file_name, contains_timestamps):
    if contains_timestamps:
        timestamps = data[:, 0]
        values = data[:, 1:]
    else:
        timestamps = np.arange(data.shape[0])  # Create a sequence of indices as timestamps
        values = data

        # Plotting
    plt.figure(figsize=(10, 6))
    if values.shape[1] == 1:
        # If there's only one column of values
        plt.plot(timestamps, values, label='Feature 1')
    else:
        # If there are multiple columns of values
        for i in range(values.shape[1]):
            plt.plot(timestamps, values[:, i], label=f'Feature {i + 1}')

    file_name = shorten_file_name(file_name)
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.title('Plot of ' + file_name)
    plt.legend()
    plt.grid(True)
    plt.savefig("./exampleGraphs/normalPlots/" + file_name + ".png", format='png', dpi=200)
    plt.show()

def plot_data_standalone(directories, single_sensor_name):
    all_file_pairs = get_matching_file_pairs_from_directories(directories, single_sensor_name)
    for file_pair in all_file_pairs:
        for single_file in file_pair:
            data, true_labels = csv_file_to_nparr(single_file, True)
            plot_data_integrated(data, single_file, False)


def plot_acf_standalone(directories, single_sensor_name):
    #diff_series = np.diff(values)
    #print("Here: " + str(values.shape))

    all_file_pairs = get_matching_file_pairs_from_directories(directories, single_sensor_name)
    for file_pair in all_file_pairs:
        for single_file in file_pair:
            data, true_labels = csv_file_to_nparr(single_file, True)
            values = data
            file_name = shorten_file_name(single_file)

            # Create ACF plot
            print("Hier könnte ihr ACF Plot stehen!")
            for i in range(values.shape[1]):
                series = values[:, i]
                plt.figure(figsize=(10, 6))
                plot_acf(series, lags=20, alpha=0.05)
                plt.title('ACF of series ' + str(i) + " in " + file_name)
                plt.xlabel('Lag')
                plt.ylabel('Autocorrelation')
                plt.savefig("./exampleGraphs/ACFPlots/" + file_name + ".png", format='png', dpi=200)
                plt.show()


def shorten_file_name(file_name):
    return file_name[file_name.rfind("/") + 1:].rstrip(".csv").replace("\\", "_")


    # series = data[:, 1]
    #
    # # Step 1: Compute the differences
    # diff_series = np.diff(series)
    #
    # # Step 2: Perform the Ljung-Box test
    # ljung_box_result = acorr_ljungbox(diff_series, lags=[10], return_df=True)
    #
    # # Step 3: Plot the differenced series
    # plt.figure(figsize=(14, 6))
    # plt.subplot(1, 2, 1)
    # plt.plot(diff_series)
    # plt.title('Differenced Series')
    # plt.xlabel('Time')
    # plt.ylabel('Differenced Value')
    #
    # # Step 4: Plot the p-values from the Ljung-Box test
    # plt.subplot(1, 2, 2)
    # plt.bar(ljung_box_result.index, ljung_box_result['lb_pvalue'])
    # #plt.yscale('log')
    # plt.title('Ljung-Box Test p-values')
    # plt.xlabel('Lag')
    # plt.ylabel('p-value')
    # plt.axhline(y=0.05, color='r', linestyle='--')  # Significance level line
    #
    # plt.tight_layout()
    # plt.show()
