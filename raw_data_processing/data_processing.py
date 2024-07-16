import csv
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
        if(time_steps > 1):
            data = data[:(data.shape[0]//time_steps) * time_steps]
            data = data.reshape((data.shape[0]//time_steps, time_steps, data.shape[1]))
        else:
            data = data.reshape((data.shape[0], time_steps, data.shape[1]))
        data_list[i] = data
        print("Reshaped data for LSTM into: " + str(data))
    return data_list


def split_data_sequence_into_datasets(arr, train_ratio, val1_ratio, val2_ratio, test_ratio):
    # train_ratio = 0.7
    # val1_ratio = 0.1  #TODO: should be 0.2; haven't implemented early_stopping using val1 yet
    # val2_ratio = 0.1  #TODO: temp bandaid while early_stopping is not implemented
    # test_ratio = 0.1

    assert (train_ratio*10 + val1_ratio*10 + val2_ratio*10 + test_ratio*10) == 10   #due to stupid floating point assertionError

    n_total = len(arr)
    n_train = int(train_ratio * n_total)
    n_val1 = int(val1_ratio * n_total)
    n_val2 = int(val2_ratio * n_total)
    n_test = n_total - n_train - n_val1 - n_val2  # To ensure all samples are used

    print(f"Total samples: {n_total}")
    print(f"Training samples: {n_train}")
    print(f"Validation 1 samples: {n_val1}")
    print(f"Validation 2 samples: {n_val2}")
    print(f"Test samples: {n_test}")

    # Split data sequentially
    sN = arr[:n_train]
    vN1 = arr[n_train:n_train + n_val1]
    vN2 = arr[n_train + n_val1:n_train + n_val1 + n_val2]
    tN = arr[n_train + n_val1 + n_val2:n_train + n_val1 + n_val2 + n_test]
    print(f"Training set size: {len(sN)}")
    print(f"Validation set 1 size: {len(vN1)}")
    print(f"Validation set 2 size: {len(vN2)}")
    print(f"Test set size: {len(tN)}")

    print("sN df: " + str(sN))
    print("vN1 df: " + str(vN1))
    print("vN2 df: " + str(vN2))
    print("tN df: " + str(tN))

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
        print(f"Array {i+1}: {shape}")

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
    time_diffs = np.diff(data[:, 0], prepend=data[0, 0])
    return np.column_stack((time_diffs, data[:, 1:]))

def convert_timestamp_to_relative_time_diff(data):
    start_timestamp = data[0, 0]
    for i in range(0, len(data)):
        data[i][0] = data[i][0] - start_timestamp
    return data

def csv_file_to_dataframe_to_numpyArray(file_path):
    samples = []
    for file_name in file_path:
        df = clean_csv(file_name)
        curr_sample = np.zeros((df.shape[0], df.shape[1]))
        for row_index, row in df.iterrows():
            for col_index, column in enumerate(df.columns):
                curr_sample[row_index, col_index] = row[column]
                # print("row_index: " + str(row_index))
                # print("column: " + str(column) + " + col_index: " + str(col_index))
                # print("grabbed: " + str(row[column]))

        samples.append(curr_sample)
        print("converted csv to numpy array: " + str(curr_sample))
    return samples


def clean_csv(file_path):
    df = pd.read_csv(file_path)
    columns_to_remove = ['header.stamp.secs', 'header.stamp.nsecs', 'header.frame_id', 'header.seq']

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


def read_file_to_csv_bagpy(path):
    b = bagreader(path)

    csvfiles = []
    for topic in b.topics:
        if (topic =="/sbg/imu_data"):   #causes UTF-8 encoding error
            continue
        print(topic)
        f_name = b.message_by_topic(topic)
        print(f_name)
        csvfiles.append(f_name)

    print(csvfiles[0])
    f_name = pd.read_csv(csvfiles[0])

    print(csvfiles[1])
    f_name = pd.read_csv(csvfiles[1])

    print(csvfiles[2])
    f_name = pd.read_csv(csvfiles[2])

# Sample time of approximately 10 milliseconds for wheelspeed

def get_sample_time(bag, topicName):
    i = 0
    time_diffs = []
    prev_timestamp = 0

    for topic, msg, t in bag.read_messages(topics=[topicName]):
        # print(msg)

        if i > 9:
            break

        #print(msg.__slots__)

        # Sample time of approximately 10 milliseconds for wheelspeed
        curr_timestamp = t.secs * 1e9 + t.nsecs
        time_diffs.append(str(curr_timestamp - prev_timestamp))
        prev_timestamp = (t.secs * 1e9 + t.nsecs)
        i += 1

    # t is more accurate
    # print("t: " + str(t))
    # combined_nanoseconds = t.secs * 1e9 + t.nsecs
    # print("Combined secs and nsecs: " + str(combined_nanoseconds))

    print("Sample time differences for " + str(topicName)[str(topicName).rfind("/") + 1:] + ": " + str(time_diffs))
