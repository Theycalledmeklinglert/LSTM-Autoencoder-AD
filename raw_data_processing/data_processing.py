import csv
import glob
import os
import traceback
import rosbag
import genpy
import pandas as pd
from bagpy import bagreader
import numpy as np
from sklearn.preprocessing import MaxAbsScaler


def reshape_data_for_autoencoder_lstm(data_list, time_steps):
    # Reshape X to fit LSTM input shape (samples, time steps, features)
    for i in range(len(data_list)):
        data = data_list[i]
        if (time_steps > 1):
            data = data[:(data.shape[0] // time_steps) * time_steps]
            data = data.reshape((data.shape[0] // time_steps, time_steps, data.shape[1]))
        else:
            data = data.reshape((data.shape[0], time_steps, data.shape[1]))
        data_list[i] = data
        #print("Reshaped data for LSTM into: " + str(data))
    return data_list


def split_data_sequence_into_datasets(arr, train_ratio, val1_ratio, val2_ratio, test_ratio):
    # train_ratio = 0.7
    # val1_ratio = 0.1  #TODO: should be 0.2; haven't implemented early_stopping using val1 yet
    # val2_ratio = 0.1  #TODO: temp bandaid while early_stopping is not implemented
    # test_ratio = 0.1

    assert (train_ratio * 10 + val1_ratio * 10 + val2_ratio * 10 + test_ratio * 10) == 10  #due to float round error

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

    #return sN, vN1, vN2, tN
    return sN, vN1, vN2, tN


def reshape_data_for_LSTM(data, steps):
    # Reshape X to fit LSTM input shape (samples, time steps, features)
    #print(data.shape)
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
    return scaler.fit_transform(data)


def reverse_normalize_data(scaled_data, scaler):
    if scaler is None:
        return scaled_data
    return scaler.inverse_transform(scaled_data)


def convert_timestamp_to_absolute_time_diff(data):
    # time_diffs = np.diff(data[:, 0], prepend=data[0, 0])
    # return np.column_stack((time_diffs, data[:, 1:]))
    print("this is stupid")
    return


def convert_timestamp_to_relative_time_diff(df):
    time_columns = [col for col in df.columns if col == "Time"]
    # Subtract each value in "Time" columns from the first row's value

    for time_col in time_columns:
        start_timestamp = df[time_col].iloc[0]
        df[time_col] = df[time_col] - start_timestamp

    # start_timestamp = data[0, 0]
    # for i in range(0, len(data)):
    #     data[i][0] = data[i][0] - start_timestamp

    return df


def directory_csv_files_to_dataframe_to_numpyArray(file_path):
    df = clean_csv(file_path)
    if df is None:
        return
    df = convert_timestamp_to_relative_time_diff(df)
    samples = np.zeros((df.shape[0], df.shape[1]))
    for row_index, row in df.iterrows():
        for col_index, column in enumerate(df.columns):
                samples[row_index, col_index] = row[column]
                # print("row_index: " + str(row_index))
                # print("column: " + str(column) + " + col_index: " + str(col_index))
                # print("grabbed: " + str(row[column]))
    return samples

def old_directory_csv_files_to_dataframe_to_numpyArray(directory):
    print("Reading files from directory: " + directory)
    dims = [0, 0]
    dfs = []

    return

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


#removes empty columns, columns containing only one and the same value through and through and columns that do not offer useful information
def clean_csv(file_path):
    # list of known incompatible csv files due to incompatible/incorrect formatting or files that are simply are not sensor data
    exceptions = [
        "slam-car_state", "slam-landmark_info", "slam-map",
        "slam-state", "stereo_cone_perception-cones", "tf", "estimation-velocity", "lidar-cone_position_cloud",
        "map_matching-driving_path", "sbg-gps_raw",
        "map_matching-reference_track", "diagnostics"
        #todo: diagnostics is temporary here for now. might be useful to scan and process it separetly first and not include it in anomaly detection itself
        #todo: also ask Sebastian if "map_matching-reference_track" is of any use for AD; for now it's out
        #todo: "sbg-ekf_euler" also looks weird; not sure if useful
    ]
    if any(keyword in file_path for keyword in exceptions):
        print("did not process: " + str(file_path))
        return

    df = pd.read_csv(file_path)
    columns_to_remove = ['header.stamp.secs', 'header.stamp.nsecs', 'header.frame_id', 'child_frame_id',
                         'twist.covariance']  #todo: test if removing this was good or bad: ", 'header.seq'"

    for col in columns_to_remove:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Remove columns that contain only the column name
    df.dropna(axis=1, how='all', inplace=True)

    # Remove columns that only contain one and the same value
    for col in df.columns:
        if df[col].nunique() == 1:
            df.drop(columns=[col], inplace=True)

    return df


def get_csv_file_paths(directory):
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    return csv_files


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
            #if i == 100:
            #    break

        time_diffs = [float(value) / 1e6 for value in time_diffs]   #convert to msecs
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

