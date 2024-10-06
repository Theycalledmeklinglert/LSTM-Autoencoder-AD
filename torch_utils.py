import torch
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import precision_recall_curve
from torch.utils.data import Dataset
import numpy as np

from data_processing import clean_csv, convert_timestamp_to_relative_time_diff, reshape_data_for_autoencoder_lstm, \
    get_normalized_data_and_labels, get_matching_file_pairs_from_directories, get_noShift_andShift_data_windows_for_lstm


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


def get_data_as_list_of_single_batches_of_subseqs(time_steps, window_step, remove_timestamps, scaler=None, directories=None, single_sensor_name=None):
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
        data_with_time_diffs = reshape_data_for_autoencoder_lstm(data_with_time_diffs, time_steps, window_step)
        true_labels_list = reshape_data_for_autoencoder_lstm(true_labels_list, time_steps, window_step)
        [print("reshaped data shape: \n" + str(data.shape)) for data in data_with_time_diffs]

        return [data_with_time_diffs, true_labels_list]


def get_data_as_shifted_batches_seqs(time_steps, remove_timestamps, window_step=0, scaler=None, directories=None, single_sensor_name=None):
    all_file_pairs = get_matching_file_pairs_from_directories(directories, single_sensor_name)
    print("all_file_pairs: " + str(all_file_pairs))
    for file_pair in all_file_pairs:
        #TODO: CHANGED FOR TESTING!!!!!!!
        data_with_time_diffs, true_labels_list = get_normalized_data_and_labels(file_pair, scaler, 1.0, remove_timestamps)
        [print("original data shape: \n" + str(data.shape)) for data in data_with_time_diffs]
        # if list is empty due to excluded csv file
        if not data_with_time_diffs:
            continue

        print("now reshaping and shifting")
        no_shift_data_with_time_diffs, shift_data_with_time_diffs = get_noShift_andShift_data_windows_for_lstm(data_with_time_diffs, time_steps, window_step)
        no_shift_true_labels_list, shift_true_labels_list = get_noShift_andShift_data_windows_for_lstm(true_labels_list, time_steps, window_step)
        [print("reshaped and shifted data shape: \n" + str(data.shape)) for data in data_with_time_diffs]
        print("reshaped and shifted true label list shape: \n" + str(true_labels_list[2].shape))

        return no_shift_data_with_time_diffs, shift_data_with_time_diffs, no_shift_true_labels_list, shift_true_labels_list


def batched_tensor_to_numpy_and_invert_scaling(tensor, scaler):
    '''
        :param tensor: tensor or numpy array of shape [batch_size, time_steps, nb_features]
        :param scaler: scaler used to scale the data
        :return: numpy array of shape (batch_size * time_steps, nb_features) and rescaled if scaler is provided
        '''
    data_with_time_diffs = None
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

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Feature Value")
    plt.legend()
    plt.grid(True)
    plt.show()


def calculate_mle_mu_sigma(error_vecs):
    mu = np.mean(error_vecs, axis=0)
    sigma = None
    if error_vecs.shape[1] == 1:
        sigma = np.var(error_vecs)  # variance for 1D vectors
    else:
        sigma = np.cov(error_vecs, rowvar=False)  # Covariance matrix for mD vectors

    return mu, sigma


def compute_anomaly_score(error_vecs, mu, sigma):
    print("error_vecs shape: " + str(error_vecs.shape))
    scores_of_seq = None

    if error_vecs.shape[1] == 1:
        # scores_of_window.append(np.square(data_point - mu) / sigma) #z_score for univariate data -->doesnt work
        scores_of_seq = np.sqrt(np.square(np.subtract(error_vecs, mu)))     #todo: this might need improvement
    else:
        # Mahalanobis distance for multivariate data
        inv_cov_matr = np.linalg.inv(sigma)
        diff = error_vecs - mu
        #score = np.dot(np.dot(diff, inv_cov_matr), diff.T) #todo: old one for data point one
        # print("score manually: " + str(score))
        # score = mahalanobis(data_point, mu, inv_cov_matr)
        scores_of_seq = np.einsum('ij,jk,ik->i', diff, inv_cov_matr, diff)

    return scores_of_seq


def find_optimal_threshold(anomaly_scores, true_labels, beta):
    anomaly_scores = anomaly_scores.flatten()
    true_labels = np.squeeze(true_labels).flatten()

    print("anomaly_scores shape: " + str(anomaly_scores.shape))
    print("true_labels shape: " + str(true_labels.shape))

    #true_labels = np.where(true_labels > 0.5, 1, 0)

    precision, recall, thresholds = precision_recall_curve(y_true=true_labels, y_score=anomaly_scores)
    fbeta_scores = (1 + beta ** 2) * (precision * recall) / ((beta ** 2) * precision + recall)
    fbeta_scores = np.nan_to_num(fbeta_scores, nan=-np.inf, posinf=-np.inf)

    print("fbeta scores: \n" + str(fbeta_scores))
    print("best fbeta: " + str(np.max(fbeta_scores)))
    best_index = np.argmax(fbeta_scores)
    if best_index >= len(thresholds):
        best_index = len(thresholds) - 1

    print("Thresholds: " + str(thresholds))
    print("Best threshold: " + str(thresholds[best_index]))
    print(str(thresholds))
    best_threshold = thresholds[best_index]
    best_fbeta = fbeta_scores[best_index]

    plt.figure()
    plt.plot(thresholds, precision[:-1], 'b-', label='Precision')
    plt.plot(thresholds, recall[:-1], 'g-', label='Recall')
    plt.xlabel('Threshold')
    plt.ylabel('Precision/Recall')
    plt.legend(loc='best')
    plt.title('Precision-Recall Curve')
    plt.show()

    return best_threshold, best_fbeta


def plot_data_over_threshold(anomaly_scores, true_labels, threshold, file_name, time_steps):
    """
    Plots the data points under the threshold as blue dots, the ones over the threshold as red crosses,
    and the threshold itself as a red dotted line.

    Parameters:
    - anomaly_scores: 2D numpy array (sequences x time steps) of anomaly scores
    - true_labels: 3D numpy array (sequences x time steps x 1) of true labels (1 for anomaly, 0 for normal)
    - threshold: A scalar value representing the threshold
    - file_name: Name of the file to save the plot
    """

    true_labels = np.squeeze(true_labels).flatten()
    anomaly_scores = anomaly_scores.flatten()

    print('true labels in plot_data_over_threshold: \n', true_labels)

    if anomaly_scores.shape != true_labels.shape:
        print("Anomaly scores shape is not equal to true labels shape")
        return

    # window_anomaly_scores = []
    # window_labels = []
    #
    # for i in range(0, len(anomaly_scores), time_steps):
    #     window_scores = anomaly_scores[i:i + time_steps]
    #     window_labels_slice = true_labels[i:i + time_steps]
    #
    #     if np.any(window_labels_slice == 1):
    #         # window_anomaly_scores.append(np.max(window_scores))
    #         # if np.mean(window_scores > threshold) > 0.4:
    #         #    window_anomaly_scores.append(np.max(window_scores))
    #         # else:
    #         #    window_anomaly_scores.append(np.mean(window_scores))
    #         window_anomaly_scores.append(np.max(window_scores))
    #         window_labels.append(1)
    #     else:
    #         # Check if more than 30% of the anomaly scores in window are higher than threshold
    #         # if np.mean(window_scores > threshold) > 0.4:
    #         #     window_anomaly_scores.append(np.max(window_scores))
    #         # else:
    #         #     window_anomaly_scores.append(np.mean(window_scores))
    #         if np.any(window_scores > threshold):
    #             window_anomaly_scores.append(np.max(window_scores))
    #         else:
    #             window_anomaly_scores.append(np.mean(window_scores))
    #         window_labels.append(0)

    for k in range(2):
        plt.figure(figsize=(12, 6))

        # for idx, (score, label) in enumerate(zip(window_anomaly_scores, window_labels)):
        #     if label == 1:  # True anomaly (window)
        #         if score >= threshold:
        #             color = 'red'  # Anomalous window above threshold
        #         else:
        #             color = 'yellow'  # Anomalous window below threshold
        #         marker = 'x'
        #     else:  # Normal window
        #         if score >= threshold:
        #             color = 'purple'  # Normal window above threshold
        #         else:
        #             color = 'blue'  # Normal window below threshold
        #         marker = 'o'
        #
        #     plt.scatter(idx, score, color=color, marker=marker, s=20, label='Anomaly' if label == 1 else 'Normal')

        anomalous_above = {'x': [], 'y': []}
        anomalous_below = {'x': [], 'y': []}
        normal_above = {'x': [], 'y': []}
        normal_below = {'x': [], 'y': []}
        for idx, (score, label) in enumerate(zip(anomaly_scores, true_labels)):
            if label == 1:  # True anomaly
                if score >= threshold:
                    anomalous_above['x'].append(idx)
                    anomalous_above['y'].append(score)
                else:
                    anomalous_below['x'].append(idx)
                    anomalous_below['y'].append(score)
            else:  # Normal
                if score >= threshold:
                    normal_above['x'].append(idx)
                    normal_above['y'].append(score)
                else:
                    normal_below['x'].append(idx)
                    normal_below['y'].append(score)

        plt.scatter(anomalous_above['x'], anomalous_above['y'], color='red', marker='x', s=20, label='Anomaly')
        plt.scatter(anomalous_below['x'], anomalous_below['y'], color='yellow', marker='x', s=20)
        plt.scatter(normal_above['x'], normal_above['y'], color='purple', marker='o', s=20, label='Normal')
        plt.scatter(normal_below['x'], normal_below['y'], color='blue', marker='o', s=20)

        if k == 0:
            plt.ylim(0, threshold * 1.1)
        else:
            plt.ylim(0, max(anomaly_scores) * 1.1)

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Normal', markerfacecolor='blue', markersize=10),
            Line2D([0], [0], marker='x', color='w', label='Anomaly', markerfacecolor='red', markersize=10)
        ]

        plt.axhline(y=threshold, color='red', linestyle='--', label='Anomaly Threshold', linewidth=2)

        plt.text(-0.05 * len(anomaly_scores), threshold, f'{threshold}', va='center', ha='right', color='red',
                 fontsize=12, fontweight='bold')

        plt.legend(handles=legend_elements, loc='upper left')
        plt.title('Anomaly Scores with True Labels of ' + file_name)
        plt.xlabel('Window Index')
        plt.ylabel('Anomaly Score')

        plt.savefig(file_name + str(k) + '.png')
        plt.show()

    print("Number of windows above threshold: " + str(np.sum(np.array(anomaly_scores) > threshold)))
    print("Number of windows below threshold: " + str(np.sum(np.array(anomaly_scores) <= threshold)))
