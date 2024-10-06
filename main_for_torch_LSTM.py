import torch
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer, MaxAbsScaler
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from data_processing import clean_csv, convert_timestamp_to_relative_time_diff, reverse_normalization
from torch_LSTM_autoenc import LSTMAutoEncoder
from torch_preprocessing import preprocessing
from torch_utils import EarlyStopping, ModelManagement, LossCheckpoint, FormulaStudentDataset, \
    get_data_as_list_of_single_batches_of_subseqs, batched_tensor_to_numpy_and_invert_scaling, plot_time_series, \
    get_data_as_shifted_batches_seqs, calculate_mle_mu_sigma, compute_anomaly_score, find_optimal_threshold, \
    plot_data_over_threshold

name_model = 'lstm_model'
path_model = './models/'
device = torch.device('cpu')

# ----------------------------------#
#         load dataset              #
# ----------------------------------#

# directories=["./aufnahmen/csv/autocross_valid_16_05_23", "./aufnahmen/csv/autocross_valid_run", "./aufnahmen/csv/anomalous data",
# "./aufnahmen/csv/test data/ebs_test_steering_motor_encoder_damage"],
# single_sensor_name="can_interface-wheelspeed.csv")

#todo:
# steering angle: step_window = 0;size_window = 300;batch_size = 16;hidden_size=100;scaler = StandardScaler();dropout = 0.4 (auch wenn das nichts macht glaube ich)


step_window = 29
size_window = 30

#scaler = MinMaxScaler(feature_range=(0, 1))
#scaler = MinMaxScaler(feature_range=(0, 1))  # Scales the data to a fixed range, typically [0, 1].
#scaler = StandardScaler()                      #Scales the data to have a mean of 0 and a standard deviation of 1.
scaler = MaxAbsScaler()                         #Scales each feature by its maximum absolute value, so that each feature is in the range [0, 1] or [-1, 0] or [-1, 1]
#scaler = QuantileTransformer(output_distribution='normal')
#scaler = None

#todo:
# try increase batch_size; increase window step; adding timestamp; try shuffling ;scaling; try changing hidden size;
# maybe only one per batch but i dont think tis fixes the issue

# all_data = get_data_as_single_batches_of_subseqs(size_window, step_window, True, scaler=scaler,
#                                                  directories=["./aufnahmen/csv/skidpad_valid_fast2_17_47_28", "./aufnahmen/csv/skidpad_valid_fast3_17_58_41", "./aufnahmen/csv/anomalous data", "./aufnahmen/csv/test data/skidpad_falscher_lenkungsoffset"],
#                                                  single_sensor_name="can_interface-current_steering_angle.csv")
# all_data = get_data_as_single_batches_of_subseqs(size_window, step_window, True, scaler=scaler,
#                                                  directories=["./aufnahmen/csv/autocross_valid_16_05_23",
#                                                               "./aufnahmen/csv/autocross_valid_run",
#                                                               "./aufnahmen/csv/anomalous data",
#                                                               "./aufnahmen/csv/test data/ebs_test_steering_motor_encoder_damage"], single_sensor_name="can_interface-wheelspeed.csv")

not_shifted_data_winds, shifted_data_winds, not_shifted_true_winds, shifted_true_winds = get_data_as_shifted_batches_seqs(size_window, True, window_step=step_window, scaler=scaler, directories=["./aufnahmen/csv/skidpad_valid_fast2_17_47_28", "./aufnahmen/csv/skidpad_valid_fast3_17_58_41", "./aufnahmen/csv/anomalous data", "./aufnahmen/csv/test data/skidpad_falscher_lenkungsoffset"], single_sensor_name="can_interface-current_steering_angle.csv")

#not_shifted_data_winds, shifted_data_winds, not_shifted_true_winds, shifted_true_winds = get_data_as_shifted_batches_seqs(size_window, True, window_step=step_window, scaler=scaler, directories=["./aufnahmen/csv/autocross_valid_16_05_23", "./aufnahmen/csv/autocross_valid_run", "./aufnahmen/csv/anomalous data", "./aufnahmen/csv/test data/ebs_test_steering_motor_encoder_damage"], single_sensor_name="can_interface-wheelspeed.csv")


howMuchIsThis = not_shifted_data_winds[0]
test = not_shifted_data_winds[0][0]
test_shifted = shifted_data_winds[0][0]
test2 = not_shifted_data_winds[0][1]
test_shifted2 = shifted_data_winds[0][1]

test_shifted_true_labels = shifted_true_winds[0][0]
test_not_shifted_true_labels = not_shifted_true_winds[0][0]


batch_size = 4
train_seq = not_shifted_data_winds[0]#[0][0]      # because get_data_as_single_batches_of_subseqs return [data_with_time_diffs, true_label_list]
train_seq_true_labels = not_shifted_true_winds[0]
train_true_seq = shifted_data_winds[0]
train_true_seq_true_labels = shifted_true_winds[0]

valid_seq = not_shifted_data_winds[1]#[0][1]
valid_seq_true_labels = not_shifted_true_winds[1]
valid_true_seq = shifted_data_winds[1]
valid_true_seq_true_labels = shifted_true_winds[1]

anomaly_seq = not_shifted_data_winds[2]#[0][2]
anomaly_seq_true_labels = not_shifted_true_winds[2]
anomaly_true_seq = shifted_data_winds[2]
anomaly_true_seq_true_labels = shifted_true_winds[2]

x_T_seq = not_shifted_data_winds[3]#[0][3]
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

train_loader = DataLoader(train_seq, batch_size=batch_size, shuffle=False)      # shuffle needs to be off;
true_train_loader = DataLoader(train_true_seq, batch_size=batch_size, shuffle=False)
valid_loader = DataLoader(valid_seq, batch_size=batch_size, shuffle=False)       # shuffle needs to be off;
true_valid_loader = DataLoader(valid_true_seq, batch_size=batch_size, shuffle=False)
anomaly_loader = DataLoader(anomaly_seq, batch_size=batch_size, shuffle=False)  # shuffle needs to be off;
true_anomaly_loader = DataLoader(anomaly_true_seq, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(x_T_seq, batch_size=batch_size, shuffle=False)         # shuffle needs to be off;
true_test_loader = DataLoader(x_T_true_seq, batch_size=batch_size, shuffle=False)

# ----------------------------------#
#         build model              #
# ----------------------------------#

model = LSTMAutoEncoder(num_layers=1, hidden_size=500, nb_feature=nb_feature, batch_size=batch_size, dropout=0.2, device=device)
model = model.to(device)
# optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
# loss
criterion = torch.nn.MSELoss()
# Callbacks
earlyStopping = EarlyStopping(patience=20)
model_management = ModelManagement(path_model, name_model)
# Plot
loss_checkpoint_train = LossCheckpoint()
loss_checkpoint_valid = LossCheckpoint()


def train(epoch):
    model.train()
    train_loss = 0
    model.is_training = True

    #for id_batch, data in enumerate(train_loader):
    for (id_batch, train_data), (_, y_true_data) in zip(enumerate(train_loader), enumerate(true_train_loader)):
        # print("data: " + str(data))
        # print("data_shape: " + str(data.shape))

        optimizer.zero_grad()
        # forward
        train_data = train_data.to(device).float()
        output = model.forward(train_data)
        output = torch.flip(output, dims=[1])   #todo: ?

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

    avg_loss = train_loss / len(train_loader)
    print('====> Epoch: {} Average loss: {:.6f}'.format(epoch, avg_loss))
    loss_checkpoint_train.losses.append(avg_loss)
    model.is_training = False


def evaluate(loader, true_loader, validation=False, epoch=0):
    model.eval()
    eval_loss = 0
    with torch.no_grad():

        #for id_batch, data in enumerate(loader):
        for (id_batch, input_data), (_, true_data) in zip(enumerate(loader), enumerate(true_loader)):
            input_data = input_data.to(device).float()
            output = model.forward(input_data)

            output = torch.flip(output, dims=[1])   #todo: ?

            loss = criterion(output.to(device), true_data.to(device).float())
            eval_loss += loss.item()
        print('\r', 'Eval [{}/{} ({:.0f}%)] \tLoss: {:.6f})]'.format(
            id_batch + 1, len(loader),
            (id_batch + 1) * 100 / len(loader),
            loss.item()), sep='', end='', flush=True)
    avg_loss = eval_loss / len(loader)
    print('====> Validation Average loss: {:.6f}'.format(avg_loss))
    # Checkpoint
    if validation:
        loss_checkpoint_valid.losses.append(avg_loss)
        model_management.checkpoint(epoch, model, optimizer, avg_loss)
        return earlyStopping.check_training(avg_loss)


def predict(loader, true_loader, input_data):    #, model)
    eval_loss = 0
    model.eval()
    predictions = torch.zeros(size=input_data.shape, dtype=torch.float)
    with torch.no_grad():
        #for id_batch, data in enumerate(loader):
        for (id_batch, data), (_, true_data) in zip(enumerate(loader), enumerate(true_loader)):

            data = data.to(device).float()
            output = model.forward(data)

            # print('input_data shape: ', input_data.shape)
            # print('predict shape: ', predict.shape)
            #
            # print('output shape: ', output.shape)
            # print('id_batch: ', id_batch)
            # print('id_batch*data.shape[0]: ', id_batch * data.shape[0])
            # print('data shape: ', data.shape)

            #todo: I THINK(!) i need to reverse the sequence here. This NEEDS to be tested!!
            output = torch.flip(output, dims=[1])       #todo: ?

            #predictions[id_batch * data.shape[0]:(id_batch + 1) * data.shape[0], :, :] = output.reshape(data.shape[0], seq_length, -1)
            predictions[id_batch * data.shape[0]:(id_batch + 1) * data.shape[0], :, :] = output
            loss = criterion(output.to(device), true_data.to(device).float())
            eval_loss += loss.item()

    avg_loss = eval_loss / len(loader)
    print('====> Prediction Average loss: {:.6f}'.format(avg_loss))
    return predictions


if __name__ == '__main__':
    print("I hate LSTMs")
    # transformed_ts = preprocessing(train_seq, "mean-max-min-trend-kurtosis-max_diff_var-var_var-level_shift", 50, size_window, step_window)
    # print(transformed_ts.shape)
    # print(str(transformed_ts[0]))

    for epoch in range(1, 10):
        train(epoch)
        if evaluate(valid_loader, true_valid_loader, validation=True, epoch=epoch):
            break
        # Lr on plateau
        if earlyStopping.patience_count == 20:
            print('lr on plateau ', optimizer.param_groups[0]['lr'], ' -> ', optimizer.param_groups[0]['lr'] / 10)
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 10
    model_management.save_best_model()


    # optimizer = torch.optim.Adam(model.parameters())
    # checkpoint = torch.load('./models/lstm_model_epoch_5')
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    # loss = checkpoint['loss']
    # model = model.to(device)


    predictions_train = predict(train_loader, true_train_loader, train_seq) #, model)
    print('shape of predictions: ', predictions_train.shape)
    #print('predictions: \n' + str(predictions_train))

    train_seq_numpy = batched_tensor_to_numpy_and_invert_scaling(train_seq, scaler)
    print('numpy shape of input: ', train_seq_numpy.shape)
    #print('Original scaled tensor input: \n', train_seq)
    print('Reverse scaled numpy input: \n' + str(train_seq_numpy))

    predictions_train_numpy = batched_tensor_to_numpy_and_invert_scaling(predictions_train, scaler)
    print('numpy shape of predictions: ', predictions_train_numpy.shape)
    print('Reverse scaled numpy predictions: \n' + str(predictions_train_numpy))

    plot_time_series(predictions_train_numpy, "Predicted time Series for: X_sN")

    predictions_anom = predict(anomaly_loader, true_anomaly_loader, anomaly_seq) #, model)
    predictions_anom_numpy = batched_tensor_to_numpy_and_invert_scaling(predictions_anom, scaler)
    plot_time_series(predictions_anom_numpy, "Predicted time Series for: X_vNA")

    predictions_val = predict(valid_loader, true_valid_loader, valid_seq)  # , model)
    predictions_val_numpy = batched_tensor_to_numpy_and_invert_scaling(predictions_val, scaler)
    plot_time_series(predictions_val_numpy, "Predicted time Series for: X_vN")

    print('Original shape of predictions before reshape: ', predictions_anom.shape)
    print('numpy shape of predictions after reshape: ', predictions_anom_numpy.shape)
    print('Original scaled tensor input: \n', anomaly_seq.shape)


    #TODO: Not sure if to use Shifted or unshifted Seq for comparison
    # logically it should be the unshifted one

    anomaly_true_seq_numpy = batched_tensor_to_numpy_and_invert_scaling(anomaly_true_seq, scaler)         #anomaly_true_seq
    error_vecs_anom = np.absolute(np.subtract(predictions_anom_numpy, anomaly_true_seq_numpy))
    # anomaly_seq_numpy = batched_tensor_to_numpy_and_invert_scaling(anomaly_seq, scaler)                     # anomaly_seq
    # error_vecs_anom = np.absolute(np.subtract(predictions_anom_numpy, anomaly_seq_numpy))

    plot_time_series(error_vecs_anom, "Pred Error X_vNA")

    valid_true_seq_numpy = batched_tensor_to_numpy_and_invert_scaling(valid_true_seq, scaler)              #valid_true_seq
    error_vecs_val = np.absolute(np.subtract(predictions_val_numpy, valid_true_seq_numpy))
    # valid_seq_numpy = batched_tensor_to_numpy_and_invert_scaling(valid_seq, scaler)                         #valid_seq
    # error_vecs_val = np.absolute(np.subtract(predictions_val_numpy, valid_seq_numpy))

    plot_time_series(error_vecs_val, "Pred Error X_vN")

    train_true_seq_numpy = batched_tensor_to_numpy_and_invert_scaling(train_true_seq, scaler)             #train_true_seq
    error_vecs_train = np.absolute(np.subtract(predictions_train_numpy, train_true_seq_numpy))
    # train_seq_numpy = batched_tensor_to_numpy_and_invert_scaling(train_seq, scaler)                         #train_seq
    # error_vecs_train = np.absolute(np.subtract(predictions_train_numpy, train_seq_numpy))

    plot_time_series(error_vecs_train, "Pred Error X_sN")

    mu, sigma = calculate_mle_mu_sigma(error_vecs_val)
    anom_anomaly_scores = compute_anomaly_score(error_vecs_anom, mu, sigma)


    #TODO: I'm not completely sure if should take the shifted or the unshifted true values?
    # bei wheelspeed könnte die künstliche Überkänge von X_sN Probleme machen; Denke Datasets sollten alle ähnliche Länge haben
    # Kann theoretisch auch mal shufflen ausprobieren
    # Evtl. für jede Batch aus Windows eigene mu, sigma und Anom Score Threshold berechnen
    # Jedes Window + threshold dann zusammenfügen oder einzeln plotten
    # ODER: Time Series in Hälfte (oder anderweitig in den "konstanten" und "hoch variablen" Teil teilen (funktioniert bei Steering Angele gut)
    # Dann Threshold Berechnung für jeden Teil
    # Gucken ob ich doch die unshifted Sequences zur error_vec Berechnung nehmen sollte
    # Direkt vor und nach Anomalie Sequenzen in csv X_vNA Punkte auch als Anomalien markieren?

    anomaly_true_seq_true_labels_numpy = batched_tensor_to_numpy_and_invert_scaling(anomaly_true_seq_true_labels, None)
    print('true labels numpy: ', anomaly_true_seq_true_labels_numpy)
    print('anom_anomaly_scores: ', anom_anomaly_scores)


    best_anomaly_threshold, best_fbeta = find_optimal_threshold(anom_anomaly_scores, anomaly_true_seq_true_labels_numpy, 0.3)
    print("Best anomaly threshold: " + str(best_anomaly_threshold))

    #X_tN_anomaly_scores = compute_anomaly_score(X_tN_error_vecs, mu, sigma)
    valid_true_seq_true_labels_numpy = batched_tensor_to_numpy_and_invert_scaling(valid_true_seq_true_labels, None)
    val_anomaly_scores = compute_anomaly_score(error_vecs_val, mu, sigma)


    #todo: may have to adjust these functions if using window_step
    plot_data_over_threshold(anom_anomaly_scores, anomaly_true_seq_true_labels_numpy, best_anomaly_threshold, "Anom scores X_vNA", size_window)
    plot_data_over_threshold(val_anomaly_scores, valid_true_seq_true_labels_numpy, best_anomaly_threshold, "Anom scores X_vN", size_window)