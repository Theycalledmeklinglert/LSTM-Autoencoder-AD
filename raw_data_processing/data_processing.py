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
        print(data.shape)
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
        data = b.message_by_topic(topic)
        print(data)
        csvfiles.append(data)

    print(csvfiles[0])
    data = pd.read_csv(csvfiles[0])

    print(csvfiles[1])
    data = pd.read_csv(csvfiles[1])

    print(csvfiles[2])
    data = pd.read_csv(csvfiles[2])



# Sample time of approximately 10 milliseconds for wheelspeed
def read_file_rosbag(fName):
    # rosbag = bagreader(fName)
    # print(rosbag.topic_table)
    # rosbag.get

    bag = rosbag.Bag(fName)
    topic_info = bag.get_type_and_topic_info()[1]
    topics = list(topic_info.keys())
    types = [val[0] for val in topic_info.values()]

    print("Types:", types)
    print("Topics:", topics)


    #i = 0
    processed_topics = {topic: False for topic in topics}

    #for topic, msg, t in bag.read_messages(topics=['/can_interface/wheelspeed']):
    try:
        for topic, msg, t in bag.read_messages():

            #print(msg)

            #if i > 9:
            #    break
            #print(msg.__slots__)

            if not processed_topics[topic]:
                print(f"Processing topic: {topic}")
                print(f"Message slots: {msg.__slots__}")

                msg_data = {}
                for slot_name in msg.__slots__:
                    attr = getattr(msg, slot_name)
                    if isinstance(attr, bytes):
                        try:
                            attr = attr.decode('utf-8')
                        except (SyntaxError, UnicodeError):
                            try:
                                attr = attr.decode('latin1')  # or another encoding
                            except (SyntaxError, UnicodeError) as e:
                                print(f"Failed to decode attribute {slot_name}: {e}")
                                continue
                    msg_data[slot_name] = attr
                print("Message data:", msg_data)

            processed_topics[topic] = True

            # Check if all topics have been processed
            if all(processed_topics.values()):
                break

            #i += 1

    except ((SyntaxError, UnicodeError), genpy.message.MessageException) as e:
            print(f"An error occurred: {e}")

    #print(msg_data.get("header").stamp.secs)
    #print(msg_data.get("FL").data)
    #print(type(msg_data.get("FL").data))

    #print(type(msg_data.get("FL")))

    get_sample_time(bag, '/can_interface/wheelspeed')
    get_sample_time(bag, '/can_interface/current_steering_angle')
    get_sample_time(bag, '/lidar/cone_position_cloud')

    bag.close()


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

def read_bag_to_csv(bag_file, csv_file):
    try:
        bag = rosbag.Bag(bag_file)
    except Exception as e:
        print(f"Failed to open bag file: {e}")
        return

    # Open CSV file for writing
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write the header
        header_written = False
        try:
            # Iterate over all topics and messages
            for topic, msg, t in bag.read_messages():
                    if not header_written:
                        # Write header based on message fields
                        header = ['topic', 'timestamp'] + msg.__slots__
                        writer.writerow(header)
                        header_written = True

                    # Extract message data
                    row = [topic, t.to_sec()]
                    for slot in msg.__slots__:
                        attr = getattr(msg, slot)
                        if isinstance(attr, (int, float, str, bool)):
                            row.append(attr)
                        elif hasattr(attr, '__slots__'):
                            for sub_slot in attr.__slots__:
                                sub_attr = getattr(attr, sub_slot)
                                if isinstance(sub_attr, (int, float, str, bool)):
                                    row.append(sub_attr)
                                else:
                                    row.append(str(sub_attr))
                        else:
                            row.append(str(attr))

                    # Write message data to CSV
                    writer.writerow(row)

        except (SyntaxError, UnicodeError) as e:
            print(f"An error occurred while processing topic {topic}: {e}")
            print("FUCK V2: NEW AND IMPROVED")
            print(traceback.format_exc())

    bag.close() #redundant i think