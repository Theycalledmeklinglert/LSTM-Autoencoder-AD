import torch
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset
import numpy as np

from data_processing import clean_csv, convert_timestamp_to_relative_time_diff, reshape_data_for_autoencoder_lstm, \
    get_normalized_data_and_labels, get_matching_file_pairs_from_directories


class FormulaStudentDataset(Dataset):
    def __init__(self, path, type, nrows=None):
        """
        Args:
            path (string): Path to file with annotations.
            type (string): 'csv' or 'pytorch'
        """
        if type == 'csv':
            self.data = pd.read_csv(path, delimiter=' ', nrows=nrows, header=None)
            print("old Datasetloader df: \n" + str(self.data.head))
            self.data = read_Formul_Stud_csv_to_nmpy_arr(path, True, nrows=nrows)
            print("my fucked up Datasetloader df: \n" + str(self.data.head))

        elif type == 'pytorch':
            self.data = torch.load(path)
        else:
            raise ValueError('type value is wrong: ', type)
        self.type = type

    def __len__(self):
        if self.type == 'csv':
            return len(self.data)
        if self.type == 'pytorch':
            return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.type == 'csv':
            return torch.from_numpy(np.array(self.data.iloc[idx, :], dtype=np.float))
        if self.type == 'pytorch':
            return self.data[idx,:,:]



class EarlyStopping:
    def __init__(self, patience=0):
        self.last_metrics = 10**8
        self.patience = patience
        self.patience_count = 0

    def check_training(self, metric):
        if metric < self.last_metrics:
            self.last_metrics = metric
            self.patience_count = 0
            return False
        elif (metric > self.last_metrics) & (self.patience_count < self.patience):
            self.patience_count += 1
            return False
        else:
            return True


class ModelManagement:
    def __init__(self, path, name_model):
        self.path = path
        self.last_metrics = 10**8
        self.name_model = name_model
        self.dict_model = None

    def save(self, model):
        torch.save(model.state_dict(), self.path + '%s' % self.name_model)

    def checkpoint(self, epoch, model, optimizer, loss):
        if self.last_metrics > loss:
            self.dict_model = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }
            self.last_metrics = loss

    def save_best_model(self):
        torch.save(self.dict_model, self.path + '%s_epoch_%d' % (self.name_model, self.dict_model['epoch']))


class LossCheckpoint:
    def __init__(self):
        self.losses = []

    def plot(self, log=False):
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(self.losses)), self.losses)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        if log:
            plt.yscale('log')
        plt.show()


def read_Formul_Stud_csv_to_nmpy_arr(file_path, remove_timestamps, nrows):
    print("Getting data from: " + str(file_path))
    df = clean_csv(file_path, remove_timestamps)
    if df is None:
        return
    print("cleaned csv")
    df = convert_timestamp_to_relative_time_diff(df)

    #if "Anomaly" in df.columns:
    print("converted timestamps")
    print(df.shape)
    return df


def get_data_as_single_batches_of_subseqs(time_steps, window_step, remove_timestamps, scaler=None, directories=None, single_sensor_name=None):
    #scaler = MinMaxScaler(feature_range=(0, 1))  # Scales the data to a fixed range, typically [0, 1].
    # scaler = StandardScaler()                      #Scales the data to have a mean of 0 and a standard deviation of 1.
    # scaler = MaxAbsScaler()                         #Scales each feature by its maximum absolute value, so that each feature is in the range [0, 1] or [-1, 0] or [-1, 1]
    # scaler = QuantileTransformer(output_distribution='normal')
    #scaler = None

    all_file_pairs = get_matching_file_pairs_from_directories(directories, single_sensor_name)
    print("all_file_pairs: " + str(all_file_pairs))

    for file_pair in all_file_pairs:
        #TODO: CHANGED FOR TESTING!!!!!!!
        data_with_time_diffs, true_labels_list = get_normalized_data_and_labels(file_pair, scaler, 1.0, remove_timestamps)

        [print("original data shape: \n" + str(data.shape)) for data in data_with_time_diffs]

        # if list is empty due to excluded csv file
        if not data_with_time_diffs:
            continue

        print("now reshaping")

        data_with_time_diffs = reshape_data_for_autoencoder_lstm(data_with_time_diffs, time_steps,
                                                                 window_step)  #TODO: Consider WINDOW STEP
        true_labels_list = reshape_data_for_autoencoder_lstm(true_labels_list, time_steps, window_step)

        [print("reshaped data shape: \n" + str(data.shape)) for data in data_with_time_diffs]

        # TODO: Shuffling is likely entirely unnecessary/bad here since it destroys the Overlapping window functionality
        # shuffle=True is entirely experimental. X_val is used for MLE and Fbeta so I do not think it matters but im not sure
        # X_vN1, X_vN2, X_vN1_labels, X_vN2_labels = train_test_split(data_with_time_diffs[1], true_labels_list[1], test_size=0.50, shuffle=False)
        X_sN = data_with_time_diffs[0]
        X_vN1 = data_with_time_diffs[1]
        X_vNA = data_with_time_diffs[2]
        X_tN = data_with_time_diffs[3]
        # true_labels_list = transform_true_labels_to_window_size(true_labels_list)
        print("true label list shape: \n" + str(true_labels_list[2].shape))

        return [data_with_time_diffs, true_labels_list]


def batched_tensor_to_numpy_and_invert_scaling(tensor, scaler):
    '''
        :param tensor: tensor or numpy array of shape [batch_size, time_steps, nb_features]
        :param scaler: scaler used to scale the data
        :return: numpy array of shape (batch_size * time_steps, nb_features) and rescaled if scaler is provided
        '''
    # Check if the input is a PyTorch tensor
    if torch.is_tensor(tensor):
        # Reshape and move to CPU if necessary, then convert to NumPy
        numpy_arr = tensor.reshape(-1, tensor.shape[-1]).cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        # If it's already a NumPy array, reshape it directly
        numpy_arr = tensor.reshape(-1, tensor.shape[-1])
    else:
        raise TypeError("Input should be either a PyTorch tensor or a NumPy array")

    # Apply inverse scaling if a scaler is provided
    if scaler is not None:
        numpy_arr = scaler.inverse_transform(numpy_arr)

    return numpy_arr

def plot_time_series(data, title):
    print("shape of data to plot: " + str(data.shape))
    num_samples, num_features = data.shape

    # Flatten the data to plot it as a continuous time series
    #flattened_data = data.reshape(num_samples * timesteps, num_features)


    # Create a time axis (indices for the total number of timesteps)
    time_axis = range(data.shape[0])

    # Plot each feature separately
    plt.figure(figsize=(12, 6))

    for i in range(num_features):
        plt.plot(time_axis, data[:, i], label=f'Feature {i + 1}')

    plt.title("Predicted time Series for: " + title)
    plt.xlabel("Time")
    plt.ylabel("Feature Value")
    plt.legend()
    plt.grid(True)
    plt.show()
