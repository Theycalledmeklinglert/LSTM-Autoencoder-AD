import numpy as np
from matplotlib import pyplot as plt
from pmdarima.preprocessing import LogEndogTransformer, BoxCoxEndogTransformer
from scipy.stats import normaltest
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, PowerTransformer

from data_processing import get_normalized_data_and_labels, get_flattened_single_column_from_nd_nparray, csv_files_to_df
from utils import get_matching_file_pairs_from_directories
from pmdarima.utils import tsdisplay
import pmdarima as pm
from sklearn.metrics import mean_squared_error as mse


def run_auto_arima(directories, specific_sensor):
    scaler = None

    all_file_pairs = get_matching_file_pairs_from_directories(directories, specific_sensor)
    print("all_file_pairs: " + str(all_file_pairs))

    dfs = []
    anomaly_labels = []
    for file_pair in all_file_pairs:
        for single_file in file_pair:
            df = csv_files_to_df(single_file, remove_timestamps=False)
            anomaly_df = df[['Anomaly']]
            df = df.drop(columns=['Anomaly'])

            df.set_index('Time', inplace=True)


            print("df: \n" + str(df))
            print("anomaly_df: \n" + str(anomaly_df))

            print("df head: \n" + str(df.head()))
            dfs.append(df)
            anomaly_labels.append(anomaly_df)

    test_df_0 = dfs[0]
    test_df_1 = dfs[1]

    #tsdisplay(test_df_0.iloc[:, 0], lag_max=100)
    #scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = StandardScaler()
    #scaler = MaxAbsScaler()
    print(str(test_df_1.iloc[:, 0]))
    print("reshaped: \n" + str(test_df_1.iloc[:, 0].to_numpy().reshape(-1, 1)))
    scaled = scaler.fit_transform(test_df_1.iloc[:, 0].to_numpy().reshape(-1, 1)).flatten()
    print("scaled: \n" + str(scaled))

    #diff = np.diff(scaled)
    data_for_display = test_df_1.iloc[:, 0].to_numpy().reshape(-1, 1)
    diffed = np.diff(data_for_display.flatten()).reshape(-1, 1)
    data_for_fit = data_for_display.flatten()
    tsdisplay(data_for_display, lag_max=100)
    tsdisplay(diffed, lag_max=100)

    # y_train_log, _ = LogEndogTransformer(lmbda=1e-6).fit_transform(data_for_display)
    # tsdisplay(y_train_log, lag_max=100)
    #
    # y_train_bc, _ = BoxCoxEndogTransformer(lmbda2=1e-6).fit_transform(data_for_display)
    # tsdisplay(y_train_bc, lag_max=100)

    scaler = PowerTransformer(method='yeo-johnson')
    data_yeojohnson = scaler.fit_transform(data_for_display.reshape(-1, 1))
    tsdisplay(data_yeojohnson, lag_max=100)

    #todo: takes forever; let it run to see
    fit1 = pm.auto_arima(data_for_fit, m=50, trace=True, suppress_warnings=True)
    print(fit1.summary())
    forecasts, conf_int = fit1.predict(n_periods=data_for_fit.size, return_conf_int=True)

    plot_forecasts(data_for_fit, forecasts, "Test for wheelspeed")

    #for col in test_df_0.columns:


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

def plot_forecasts(y_train, forecasts, title, figsize=(8, 12)):
    y_test = y_train    #todo: i bullshitted this


    x = np.arange(y_train.shape[0] + forecasts.shape[0])

    fig, axes = plt.subplots(2, 1, sharex=False, figsize=figsize)

    # Plot the forecasts
    axes[0].plot(x[:y_train.shape[0]], y_train, c='b')
    axes[0].plot(x[y_train.shape[0]:], forecasts, c='g')
    axes[0].set_xlabel(f'Sunspots (RMSE={np.sqrt(mse(y_test, forecasts)):.3f})')
    axes[0].set_title(title)

    # Plot the residuals
    resid = y_test - forecasts
    _, p = normaltest(resid)
    axes[1].hist(resid, bins=15)
    axes[1].axvline(0, linestyle='--', c='r')
    axes[1].set_title(f'Residuals (p={p:.3f})')

    plt.tight_layout()
    plt.show()