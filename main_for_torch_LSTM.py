import time

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer, MaxAbsScaler
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from data_processing import clean_csv, convert_timestamp_to_relative_time_diff, reverse_normalization
from torch_LSTM_autoenc import LSTMAutoEncoder
from torch_utils import EarlyStopping, \
    get_data_as_list_of_single_batches_of_subseqs, batched_tensor_to_numpy_and_invert_scaling, plot_time_series, \
    get_data_as_shifted_batches_seqs, calculate_mle_mu_sigma, compute_anomaly_score, find_optimal_threshold, \
    plot_anomaly_scores_over_threshold, plot_loss_over_epochs, plot_detection_results

# directories=["./aufnahmen/csv/autocross_valid_16_05_23", "./aufnahmen/csv/autocross_valid_run", "./aufnahmen/csv/anomalous data",
# "./aufnahmen/csv/test data/ebs_test_steering_motor_encoder_damage"],
# single_sensor_name="can_interface-wheelspeed.csv")

#todo:
# steering angle: step_window = 0;size_window = 300;batch_size = 16;hidden_size=100;scaler = StandardScaler();dropout = 0.4 (auch wenn das nichts macht glaube ich)

device = torch.device('cpu')

#scaler = MinMaxScaler(feature_range=(0, 1))
scaler = MinMaxScaler()  # Scales the data to a fixed range, typically [0, 1].
#scaler = StandardScaler()                      #Scales the data to have a mean of 0 and a standard deviation of 1.
#scaler = MaxAbsScaler()                         #Scales each feature by its maximum absolute value, so that each feature is in the range [0, 1] or [-1, 0] or [-1, 1]
#scaler = QuantileTransformer(output_distribution='normal')
#scaler = None


#  previous 50:3;
# sweet spot ist definitiv 50:3
step_window = 3
size_window = 50
#single_sensor_name = "can_interface-wheelspeed.csv"
single_sensor_name = "can_interface-current_steering_angle.csv"
#single_sensor_name = "control-acceleration.csv"

#not_shifted_data_winds, shifted_data_winds, not_shifted_true_winds, shifted_true_winds = get_data_as_shifted_batches_seqs(size_window, True, window_step=step_window, scaler=scaler, directories=["./aufnahmen/csv/autocross_valid2_17_23_44", "./aufnahmen/csv/autocross_cone_mitgenommen", "./aufnahmen/csv/anomalous data/anom_data_ac", "./injectedAnomalyData"], single_sensor_name=single_sensor_name)

not_shifted_data_winds, shifted_data_winds, not_shifted_true_winds, shifted_true_winds = get_data_as_shifted_batches_seqs(size_window, True, window_step=step_window, scaler=scaler, directories=["./aufnahmen/csv/skidpad_valid_fast2_17_47_28", "./aufnahmen/csv/skidpad_valid_fast3_17_58_41", "./aufnahmen/csv/anomalous data/anom_data_skpd", "./injectedAnomalyData"], single_sensor_name=single_sensor_name)
# not_shifted_data_winds, shifted_data_winds, not_shifted_true_winds, shifted_true_winds = get_data_as_shifted_batches_seqs(
#     size_window, True, window_step=step_window, scaler=scaler,
#     directories=["./aufnahmen/csv/autocross_valid2_17_23_44", "./aufnahmen/csv/autocross_valid_run",
#                  "./aufnahmen/csv/anomalous data", "./injectedAnomalyData"], single_sensor_name=single_sensor_name)

# not_shifted_data_winds, shifted_data_winds, not_shifted_true_winds, shifted_true_winds = get_data_as_shifted_batches_seqs(
#     size_window, True, window_step=step_window, scaler=scaler,
#     directories=["./aufnahmen/csv/encoder damage detection combined csvs/train", "./aufnahmen/csv/encoder damage detection combined csvs/valid",
#                  "./aufnahmen/csv/encoder damage detection combined csvs/valid anom", "./injectedAnomalyData"], single_sensor_name=single_sensor_name)
#

#not_shifted_data_winds, shifted_data_winds, not_shifted_true_winds, shifted_true_winds = get_data_as_shifted_batches_seqs(size_window, True, window_step=step_window, scaler=scaler, directories=["./aufnahmen/csv/skidpad_valid_fast2_17_47_28", "./aufnahmen/csv/skidpad_valid_fast3_17_58_41", "./aufnahmen/csv/anomalous data", "./aufnahmen/csv/test data/skidpad_falscher_lenkungsoffset"], single_sensor_name=single_sensor_name)


test = not_shifted_data_winds[0][0]
test_shifted = shifted_data_winds[0][0]
test2 = not_shifted_data_winds[0][1]
test_shifted2 = shifted_data_winds[0][1]

test_shifted_true_labels = shifted_true_winds[0][0]
test_not_shifted_true_labels = not_shifted_true_winds[0][0]

train_seq = not_shifted_data_winds[0]  #[0][0]      # because get_data_as_single_batches_of_subseqs return [data_with_time_diffs, true_label_list]
train_seq_true_labels = not_shifted_true_winds[0]
train_true_seq = shifted_data_winds[0]
train_true_seq_true_labels = shifted_true_winds[0]

valid_seq = not_shifted_data_winds[1]  #[0][1]
valid_seq_true_labels = not_shifted_true_winds[1]
valid_true_seq = shifted_data_winds[1]
valid_true_seq_true_labels = shifted_true_winds[1]

anomaly_seq = not_shifted_data_winds[2]  #[0][2]
anomaly_seq_true_labels = not_shifted_true_winds[2]
anomaly_true_seq = shifted_data_winds[2]
anomaly_true_seq_true_labels = shifted_true_winds[2]

x_T_seq = not_shifted_data_winds[3]  #[0][3]
x_T_seq_true_labels = not_shifted_true_winds[3]
x_T_true_seq = shifted_data_winds[3]
x_T_true_seq_true_labels = shifted_true_winds[3]

print("test _train_seq: ", train_seq[0])
print('train_seq shape:', train_seq.shape)

#seq_length, size_data, nb_feature = train_seq.data[0].shape
size_data, nb_feature = not_shifted_data_winds[0][0].shape

# all_data = get_data_as_list_of_single_batches_of_subseqs(size_window, step_window, True, scaler=scaler, directories=["./aufnahmen/csv/skidpad_valid_fast2_17_47_28/short_ts_for_lstm", "./aufnahmen/csv/skidpad_valid_fast3_17_58_41/short_ts_for_lstm", "./aufnahmen/csv/anomalous data/short_ts_for_lstm", "./aufnahmen/csv/test data/skidpad_falscher_lenkungsoffset"], single_sensor_name="can_interface-current_steering_angle.csv")
# #["./aufnahmen/csv/skidpad_valid_fast2_17_47_28", "./aufnahmen/csv/skidpad_valid_fast3_17_58_41", "./aufnahmen/csv/anomalous data", "./aufnahmen/csv/test data/skidpad_falscher_lenkungsoffset"], "can_interface-current_steering_angle.csv")
# batch_size = 16
# train_seq = all_data[0][0]      # because get_data_as_single_batches_of_subseqs return [data_with_time_diffs, true_label_list]
# valid_seq = all_data[0][1]
# anomaly_seq = all_data[0][2]
# x_T_seq = all_data[0][3]
# seq_length, size_data, nb_feature = train_seq.data.shape
# print('train_seq shape:', train_seq.shape)

batch_size = 2

train_loader = DataLoader(train_seq, batch_size=batch_size, shuffle=False)  # shuffle needs to be off;
true_train_loader = DataLoader(train_true_seq, batch_size=batch_size, shuffle=False)
valid_loader = DataLoader(valid_seq, batch_size=batch_size, shuffle=False)  # shuffle needs to be off;
true_valid_loader = DataLoader(valid_true_seq, batch_size=batch_size, shuffle=False)
anomaly_loader = DataLoader(anomaly_seq, batch_size=batch_size, shuffle=False)  # shuffle needs to be off;
true_anomaly_loader = DataLoader(anomaly_true_seq, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(x_T_seq, batch_size=batch_size, shuffle=False)  # shuffle needs to be off;
true_test_loader = DataLoader(x_T_true_seq, batch_size=batch_size, shuffle=False)

num_layers = 1
hidden_size = 128
nb_feature = nb_feature
batch_size = batch_size
dropout = 0.2
name_model = 'lstm_model'
path_model = './models/'
model = LSTMAutoEncoder(num_layers=num_layers, hidden_size=hidden_size, nb_feature=nb_feature, batch_size=batch_size,
                        dropout=dropout, device=device)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
earlyStopping = EarlyStopping(patience=10)

train_losses, valid_losses = [], []
times = []


pdist = torch.nn.PairwiseDistance(p=2)
#def squared_euclidean_loss(x, x_prime):
def criterion(x, y):
    #return torch.sum((y - x) ** 2)
    result = pdist(y, x)
    return torch.sum(result ** 2)

def train(epoch):
    model.train()
    train_loss = 0
    model.is_training = True
    start_time = time.time()

    #for id_batch, data in enumerate(train_loader):
    for (id_batch, train_data), (_, y_true_data) in zip(enumerate(train_loader), enumerate(true_train_loader)):
        # print("data: " + str(data))
        # print("data_shape: " + str(data.shape))
        optimizer.zero_grad()
        # forward
        train_data = train_data.to(device).float()

        output = model.forward(train_data,
                               step_window)  #todo: could make windows of unequal sizes for input and pred here

        #print("y_true__shape: " + str(y_true_data.shape))
        #print("output_shape: " + str(output.shape))

        #loss = criterion(data, output.to(device))
        loss = criterion(output.to(device), y_true_data.to(device).float())

        # backward
        loss.backward()
        train_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
        optimizer.step()

        print('\r', 'Training [{}/{} ({:.0f}%)] \tLoss: {:.6f})]'.format(
            id_batch + 1, len(train_loader),
            (id_batch + 1) * 100 / len(train_loader),
            loss.item()), sep='', end='', flush=True)

    end_time = time.time()
    epoch_time = end_time - start_time
    times.append(epoch_time)

    avg_loss = train_loss / len(train_loader)
    print('====> Epoch: {} Average loss: {:.6f}'.format(epoch, avg_loss))
    train_losses.append(avg_loss)
    model.is_training = False


def evaluate(loader, true_loader, validation=False, epoch=0):
    model.eval()
    eval_loss = 0
    with torch.no_grad():

        #for id_batch, data in enumerate(loader):
        for (id_batch, input_data), (_, true_data) in zip(enumerate(loader), enumerate(true_loader)):
            input_data = input_data.to(device).float()
            output = model.forward(input_data, step_window)

            loss = criterion(output.to(device), true_data.to(device).float())
            eval_loss += loss.item()

        print('\r', 'Eval [{}/{} ({:.0f}%)] \tLoss: {:.6f})]'.format(
            id_batch + 1, len(loader),
            (id_batch + 1) * 100 / len(loader),
            loss.item()), sep='', end='', flush=True)

    avg_loss = eval_loss / len(loader)
    print('====> Validation Average loss: {:.6f}'.format(avg_loss))
    if validation:
        valid_losses.append(avg_loss)
        return earlyStopping.check_training(avg_loss)


def predict(loader, true_loader, input_data):  #, model)
    eval_loss = 0
    model.eval()
    predictions = torch.zeros(size=input_data.shape, dtype=torch.float)
    with torch.no_grad():
        #for id_batch, data in enumerate(loader):
        for (id_batch, data), (_, true_data) in zip(enumerate(loader), enumerate(true_loader)):
            data = data.to(device).float()
            output = model.forward(data, step_window)

            # print('input_data shape: ', input_data.shape)
            # print('predict shape: ', predict.shape)
            #
            # print('output shape: ', output.shape)
            # print('id_batch: ', id_batch)
            # print('id_batch*data.shape[0]: ', id_batch * data.shape[0])
            # print('data shape: ', data.shape)

            #predictions[id_batch * data.shape[0]:(id_batch + 1) * data.shape[0], :, :] = output.reshape(data.shape[0], seq_length, -1)
            predictions[id_batch * data.shape[0]:(id_batch + 1) * data.shape[0], :, :] = output
            loss = criterion(output.to(device), true_data.to(device).float())
            eval_loss += loss.item()

    avg_loss = eval_loss / len(loader)
    print('====> Prediction Average loss: {:.6f}'.format(avg_loss))
    return predictions


def plot_train_time(times):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, times, label='Training Time per Epoch', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.title('Training Time per Epoch')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # for epoch in range(1, 70):
    #     train(epoch)
    #     if evaluate(valid_loader, true_valid_loader, validation=True, epoch=epoch):
    #         break
    #
    #     if earlyStopping.patience_count == 7:
    #         print('lr on plateau ', optimizer.param_groups[0]['lr'], ' -> ', optimizer.param_groups[0]['lr'] / 10)
    #         optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 10
    # torch.save(model.state_dict(), "./models/torch_LSTM_" + "windSz" + str(size_window) + "_windStp" + str(step_window) + "_numlay" + str(
    #                num_layers) + "_hidSize" + str(hidden_size) + "_nbFeat" + str(nb_feature) + "_batchSz" + str(
    #                batch_size) + "_" + single_sensor_name + ".pth")
    # plot_loss_over_epochs(train_losses, valid_losses)
    # plot_train_time(times)


    #Load
    model = LSTMAutoEncoder(num_layers=num_layers, hidden_size=hidden_size, nb_feature=nb_feature, batch_size=batch_size, device=device)
    #model.load_state_dict(torch.load('./models/AC torch_LSTM_windSz50_windStp3_numlay1_hidSize128_nbFeat1_batchSz2_can_interface-current_steering_angle.csv.pth'))
    model.load_state_dict(torch.load('./models/SK torch_LSTM_windSz50_windStp3_numlay1_hidSize128_nbFeat1_batchSz2_can_interface-current_steering_angle.csv.pth'))
    model = model.to(device)




    model.eval()
    predictions_train = predict(train_loader, true_train_loader, train_true_seq)  #, model)
    print('shape of predictions: ', predictions_train.shape)
    #print('predictions: \n' + str(predictions_train))

    train_seq_numpy = batched_tensor_to_numpy_and_invert_scaling(train_true_seq, scaler)
    print('numpy shape of input: ', train_seq_numpy.shape)
    #print('Original scaled tensor input: \n', train_seq)
    print('Reverse scaled numpy input: \n' + str(train_seq_numpy))

    predictions_train_numpy = batched_tensor_to_numpy_and_invert_scaling(predictions_train, scaler)
    print('numpy shape of predictions: ', predictions_train_numpy.shape)
    print('Reverse scaled numpy predictions: \n' + str(predictions_train_numpy))

    predictions_anom = predict(anomaly_loader, true_anomaly_loader, anomaly_true_seq)  #, model)
    predictions_anom_numpy = batched_tensor_to_numpy_and_invert_scaling(predictions_anom, scaler)

    predictions_val = predict(valid_loader, true_valid_loader, valid_true_seq)  # , model)
    predictions_val_numpy = batched_tensor_to_numpy_and_invert_scaling(predictions_val, scaler)

    print('Original shape of predictions before reshape: ', predictions_anom.shape)
    print('numpy shape of predictions after reshape: ', predictions_anom_numpy.shape)
    print('Original scaled tensor input: \n', anomaly_seq.shape)

    #TODO: Not sure if to use Shifted or unshifted Seq for comparison
    # logically it should be the unshifted one

    anomaly_true_seq_numpy = batched_tensor_to_numpy_and_invert_scaling(anomaly_true_seq, scaler)  #anomaly_true_seq
    error_vecs_anom = np.absolute(np.subtract(predictions_anom_numpy, anomaly_true_seq_numpy))
    # anomaly_seq_numpy = batched_tensor_to_numpy_and_invert_scaling(anomaly_seq, scaler)                     # anomaly_seq
    # error_vecs_anom = np.absolute(np.subtract(predictions_anom_numpy, anomaly_seq_numpy))

    plot_time_series(error_vecs_anom, "Prediction error for X_vA")

    valid_true_seq_numpy = batched_tensor_to_numpy_and_invert_scaling(valid_true_seq, scaler)  #valid_true_seq
    error_vecs_val = np.absolute(np.subtract(predictions_val_numpy, valid_true_seq_numpy))
    # valid_seq_numpy = batched_tensor_to_numpy_and_invert_scaling(valid_seq, scaler)                         #valid_seq
    # error_vecs_val = np.absolute(np.subtract(predictions_val_numpy, valid_seq_numpy))

    #plot_time_series(error_vecs_val, "Pred Error X_vN")

    train_true_seq_numpy = batched_tensor_to_numpy_and_invert_scaling(train_true_seq, scaler)  #train_true_seq
    error_vecs_train = np.absolute(np.subtract(predictions_train_numpy, train_true_seq_numpy))
    # train_seq_numpy = batched_tensor_to_numpy_and_invert_scaling(train_seq, scaler)                         #train_seq
    # error_vecs_train = np.absolute(np.subtract(predictions_train_numpy, train_seq_numpy))

    #plot_time_series(error_vecs_train, "Pred Error X_sN")

    mu, sigma = calculate_mle_mu_sigma(error_vecs_val)
    anom_anomaly_scores = compute_anomaly_score(error_vecs_anom, mu, sigma)

    anomaly_true_seq_true_labels_numpy = batched_tensor_to_numpy_and_invert_scaling(anomaly_true_seq_true_labels, None)
    #print('true labels numpy: ', anomaly_true_seq_true_labels_numpy)
    #print('anom_anomaly_scores: ', anom_anomaly_scores)

    best_anomaly_threshold, best_fbeta = find_optimal_threshold(anom_anomaly_scores, anomaly_true_seq_true_labels_numpy,
                                                                1.0)
    print("Best anomaly threshold: " + str(best_anomaly_threshold))

    #X_tN_anomaly_scores = compute_anomaly_score(X_tN_error_vecs, mu, sigma)
    valid_true_seq_true_labels_numpy = batched_tensor_to_numpy_and_invert_scaling(valid_true_seq_true_labels, None)
    val_anomaly_scores = compute_anomaly_score(error_vecs_val, mu, sigma)

    predictions_test = predict(test_loader, true_test_loader, x_T_true_seq)
    predictions_test_numpy = batched_tensor_to_numpy_and_invert_scaling(predictions_test, scaler)
    plot_time_series(predictions_test_numpy, "Predicted time Series for: X_tN")

    test_true_seq_numpy = batched_tensor_to_numpy_and_invert_scaling(x_T_true_seq, scaler)  # test_true_seq
    test_error_vecs_anom = np.absolute(np.subtract(predictions_test_numpy, test_true_seq_numpy))
    test_anomaly_scores = compute_anomaly_score(test_error_vecs_anom, mu, sigma)
    test_true_seq_true_labels_numpy = batched_tensor_to_numpy_and_invert_scaling(x_T_true_seq_true_labels, None)

    #plot_time_series(predictions_train_numpy, "Predicted time series for training data")
    #plot_time_series(predictions_anom_numpy, "Predicted time series for X_VA")
    #plot_time_series(predictions_val_numpy, "Predicted time series for validation data")


    #plot_anomaly_scores_over_threshold(val_anomaly_scores, valid_true_seq_true_labels_numpy, best_anomaly_threshold,"Anom scores X_VN")

    #plot_anomaly_scores_over_threshold(anom_anomaly_scores, anomaly_true_seq_true_labels_numpy, best_anomaly_threshold,"Anom scores X_VA")

    #plot_detection_results(anomaly_true_seq_numpy, anom_anomaly_scores, anomaly_true_seq_true_labels_numpy, step_window, best_anomaly_threshold, "Steering angle")

    #plot_anomaly_scores_over_threshold(test_anomaly_scores, test_true_seq_true_labels_numpy, best_anomaly_threshold,"Anomaly scores X_tN")


    #plot_time_series(error_vecs_train, "Prediction error for Training data")

    #plot_time_series(error_vecs_anom, "Prediction error for Normal data")

    #plot_time_series(error_vecs_val, "Prediction error for Normal data")

    #plot_time_series(test_error_vecs_anom, "Prediction error for Encoder damage")

    plot_detection_results(valid_true_seq_numpy, val_anomaly_scores, valid_true_seq_true_labels_numpy, step_window,
                           best_anomaly_threshold, "")

    plot_detection_results(test_true_seq_numpy, test_anomaly_scores, test_true_seq_true_labels_numpy, step_window,
                           best_anomaly_threshold, "")




    #todo: For enocder damage:
    # mean_error_steer = np.mean(error_vecs_val[:, 0])
    # std_dev = np.std(error_vecs_val[:, 0])
    # lower_threshhold = mean_error_steer - 3*std_dev
    # upper_threshhold = mean_error_steer + 3*std_dev
    #
    # #print("avg rec error shape: ", avg_rec_error_valid)
    #
    # plot_time_series(error_vecs_train, "Prediction error for Training data")
    #
    # plot_time_series(error_vecs_anom, "Prediction error for Normal data", lower_threshhold, upper_threshhold)
    #
    # plot_time_series(error_vecs_val, "Prediction error for Normal data", lower_threshhold, upper_threshhold)
    #
    # plot_time_series(test_error_vecs_anom, "Prediction error for Encoder damage", lower_threshhold, upper_threshhold)
    #
    # plot_detection_results(test_true_seq_numpy, test_anomaly_scores, test_true_seq_true_labels_numpy, step_window,
    #                        best_anomaly_threshold, "")
    #


