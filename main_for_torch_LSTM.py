import torch
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from data_processing import clean_csv, convert_timestamp_to_relative_time_diff, reverse_normalization
from torch_LSTM_autoenc import LSTMAutoEncoder
from torch_preprocessing import preprocessing
from torch_utils import EarlyStopping, ModelManagement, LossCheckpoint, FormulaStudentDataset, \
    get_data_as_single_batches_of_subseqs, batched_tensor_to_numpy_and_invert_scaling, plot_time_series


name_model = 'lstm_model'
path_model = './models/'
device = torch.device('cpu')

# ----------------------------------#
#         load dataset              #
# ----------------------------------#

# train_seq = FormulaStudentDataset('./aufnahmen/csv/autocross_valid_16_05_23/can_interface-wheelspeed.csv', type='csv')
#
#
# transformed_ts = preprocessing(train_seq.data, "mean-max-min", "2", size_window, step_window) #"mean-max-min-trend-kurtosis-max_diff_var-var_var-level_shift"
# print(transformed_ts.shape)
# print(str(transformed_ts))
#
# train_loader = DataLoader(train_seq, batch_size, shuffle=False)  #todo: turned shuffle off for now; it was true before
#
#
# seq_length, size_data, nb_feature = train_seq.data.shape
#
#
#
# valid_seq = FormulaStudentDataset('./aufnahmen/csv/autocross_valid_run/can_interface-wheelspeed.csv', type='csv')
# valid_loader = DataLoader(valid_seq, batch_size, shuffle=False)
#
# test_seq = FormulaStudentDataset('./aufnahmen/csv/anomalous data/can_interface-wheelspeed.csv', type='csv')
# test_loader = DataLoader(test_seq, batch_size, shuffle=False)

step_window = 5
size_window = 20

scaler = MinMaxScaler(feature_range=(0, 1))
#scaler = None
#todo: try increase batch_size; increase window step; adding timestamp; try shuffling ;scaling; try changing hidden size;
all_data = get_data_as_single_batches_of_subseqs(size_window, step_window, True, scaler=scaler,
                                                 directories=["./aufnahmen/csv/autocross_valid_16_05_23",
                                                              "./aufnahmen/csv/autocross_valid_run",
                                                              "./aufnahmen/csv/anomalous data",
                                                              "./aufnahmen/csv/test data/ebs_test_steering_motor_encoder_damage"],
                                                 single_sensor_name="can_interface-wheelspeed.csv")
batch_size = 32
train_seq = all_data[0][0]
valid_seq = all_data[0][1]
anomaly_seq = all_data[0][2]
x_T_seq = all_data[0][3]

seq_length, size_data, nb_feature = train_seq.data.shape
print('train_seq shape:', train_seq.shape)

train_loader = DataLoader(train_seq, batch_size=batch_size, shuffle=False)      # shuffle needs to be off;
valid_loader = DataLoader(valid_seq, batch_size=batch_size, shuffle=False)       # shuffle needs to be off;
anomaly_loader = DataLoader(anomaly_seq, batch_size=batch_size, shuffle=False)  # shuffle needs to be off;
test_loader = DataLoader(x_T_seq, batch_size=batch_size, shuffle=False)         # shuffle needs to be off;

# ----------------------------------#
#         build model              #
# ----------------------------------#

model = LSTMAutoEncoder(num_layers=1, hidden_size=100, nb_feature=nb_feature, device=device)
model = model.to(device)
# optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
# loss
criterion = torch.nn.MSELoss()
# Callbacks
earlyStopping = EarlyStopping(patience=5)
model_management = ModelManagement(path_model, name_model)
# Plot
loss_checkpoint_train = LossCheckpoint()
loss_checkpoint_valid = LossCheckpoint()


#todo: load data; seperate it into batches

def train(epoch):
    model.train()
    train_loss = 0
    for id_batch, data in enumerate(train_loader):  #for i, sample in enumerate(X_sN):
        optimizer.zero_grad()
        # forward
        data = data.to(device).float()
        output = model.forward(data)
        loss = criterion(data, output.to(device))
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


def evaluate(loader, validation=False, epoch=0):
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for id_batch, data in enumerate(loader):
            data = data.to(device).float()
            output = model.forward(data)
            loss = criterion(data, output.to(device))
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


def predict(loader, input_data, modell):
    eval_loss = 0
    modell.eval()
    predict = torch.zeros(size=input_data.shape, dtype=torch.float)
    with torch.no_grad():
        for id_batch, data in enumerate(loader):
            data = data.to(device).float()  #todo: hier muss safe .float() hin
            output = modell.forward(data)

            # print('input_data shape: ', input_data.shape)
            # print('predict shape: ', predict.shape)
            #
            # print('output shape: ', output.shape)
            # print('id_batch: ', id_batch)
            # print('id_batch*data.shape[0]: ', id_batch * data.shape[0])
            # print('data shape: ', data.shape)

            #todo: I THINK(!) i need to reverse the sequence here. This NEEDS to be tested!!
            output = torch.flip(output, dims=[1])

            #predict[id_batch * data.shape[0]:(id_batch + 1) * data.shape[0], :, :] = output.reshape(data.shape[0], seq_length, -1)  #todo: this is probably supposed to reverse the sequence but i think the syntax is wrong
            predict[id_batch * data.shape[0]:(id_batch + 1) * data.shape[0], :, :] = output
            loss = criterion(data, output.to(device))
            eval_loss += loss.item()

    avg_loss = eval_loss / len(loader)
    print('====> Prediction Average loss: {:.6f}'.format(avg_loss))
    return predict


if __name__ == '__main__':
    print("I hate LSTMs")

    # transformed_ts = preprocessing(train_seq, "mean-max-min-trend-kurtosis-max_diff_var-var_var-level_shift", 50, size_window, step_window)
    # print(transformed_ts.shape)
    # print(str(transformed_ts[0]))

    for epoch in range(1, 20):
        train(epoch)
        if evaluate(valid_loader, validation=True, epoch=epoch):
            break
        # Lr on plateau
        if earlyStopping.patience_count == 5:
            print('lr on plateau ', optimizer.param_groups[0]['lr'], ' -> ', optimizer.param_groups[0]['lr'] / 10)
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 10
    predictions = predict(train_loader, train_seq, model)
    print('shape of predictions: ', predictions.shape)
    print('predictions: \n' + str(predictions))

    #predictions_numpy = predictions.reshape(-1, predictions.shape[-1]).cpu().numpy()
    #predictions_original_scale = scaler.inverse_transform(predictions_numpy)
    input_data_numpy = train_seq
    #input_data_numpy = reverse_normalization(input_data_numpy, scaler)
    input_data_numpy = batched_tensor_to_numpy_and_invert_scaling(input_data_numpy, scaler)
    print('numpy shape of input: ', input_data_numpy.shape)
    print('Original scaled tensor input: \n', train_seq)
    print('Reverse scaled numpy input: \n' + str(input_data_numpy))

    predictions_numpy = batched_tensor_to_numpy_and_invert_scaling(predictions, scaler)
    print('numpy shape of predictions: ', predictions_numpy.shape)
    print('Reverse scaled numpy predictions: \n' + str(predictions_numpy))

    plot_time_series(predictions_numpy, "X_sN")

    predictions_anom = predict(anomaly_loader, anomaly_seq, model)
    predictions_anom_numpy = batched_tensor_to_numpy_and_invert_scaling(predictions_anom, scaler)
    plot_time_series(predictions_anom_numpy, "X_vNA")

