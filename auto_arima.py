import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pmdarima import StepwiseContext
from pmdarima.arima import ADFTest
from pmdarima.preprocessing import LogEndogTransformer, BoxCoxEndogTransformer
from scipy.stats import normaltest
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, PowerTransformer
from statsmodels.tsa.seasonal import seasonal_decompose

from data_processing import get_normalized_data_and_labels, get_flattened_single_column_from_nd_nparray, \
    csv_files_to_df, get_matching_file_pairs_from_directories
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

            #print("df: \n" + str(df))
            #print("anomaly_df: \n" + str(anomaly_df))

            print("df head: " + str(df.head()))
            dfs.append(df)
            anomaly_labels.append(anomaly_df)

    total_length = 0
    for df in dfs:
        total_length += len(df)
    print('Length: ', total_length / len(dfs))


    test_df_0 = dfs[0]
    test_df_1 = dfs[1]

    #tsdisplay(test_df_0, title= "autocross 16_05", lag_max=1000)

    #scaler = MinMaxScaler(feature_range=(0, 1))
    #scaler = StandardScaler()
    #scaler = MaxAbsScaler()
    print(str(test_df_1.iloc[:, 0]))
    print("reshaped: \n" + str(test_df_1.iloc[:, 0].to_numpy().reshape(-1, 1)))
    #scaled = scaler.fit_transform(test_df_1.iloc[:, 0].to_numpy().reshape(-1, 1)).flatten()
    #print("scaled: \n" + str(scaled))

    df0_orig_data = test_df_1.iloc[:, 0].to_numpy()                                                  #.reshape(-1, 1)
    first_order_diffed = np.diff(df0_orig_data)                                                      #.flatten())    .reshape(-1, 1)
    # data_for_fit = df0_orig_data.flatten()
    # tsdisplay(df0_orig_data, title= "autocross_valid_run", lag_max=200)
    # tsdisplay(first_order_diffed, title="first_order_diffed autocross_valid_run", lag_max=10)
    # second_order_diffed = np.diff(first_order_diffed)                                                   #.flatten()).reshape(-1, 1)
    # tsdisplay(second_order_diffed, title="2nd order first_order_diffed autocross_valid_run", lag_max=10)

    adf_test = ADFTest(alpha=0.05)

    p_val, should_diff = adf_test.should_diff(dfs[0].iloc[:, 0].to_numpy().flatten())
    print("For autocross_valid_16_05_23 ---> p_val: " + str(p_val) + " should_diff: " + str(should_diff))
    p_val, should_diff = adf_test.should_diff(dfs[1].iloc[:, 0].to_numpy().flatten())
    print("For autocross_valid_run ---> p_val: " + str(p_val) + " should_diff: " + str(should_diff))
    p_val, should_diff = adf_test.should_diff(dfs[2].iloc[:, 0].to_numpy().flatten())
    print("For autocross_valid2_17_23_44 ---> p_val: " + str(p_val) + " should_diff: " + str(should_diff))
    p_val, should_diff = adf_test.should_diff(dfs[3].iloc[:, 0].to_numpy().flatten())
    print("For skidpad_valid_fast2_17_47_28 ---> p_val: " + str(p_val) + " should_diff: " + str(should_diff))
    p_val, should_diff = adf_test.should_diff(dfs[4].iloc[:, 0].to_numpy().flatten())
    print("For skidpad_valid_fast3_17_58_41 ---> p_val: " + str(p_val) + " should_diff: " + str(should_diff))
    p_val, should_diff = adf_test.should_diff(dfs[5].iloc[:, 0].to_numpy().flatten())
    print("For skidpad_valid_run ---> p_val: " + str(p_val) + " should_diff: " + str(should_diff))

    # p_val, should_diff = adf_test.should_diff(first_order_diffed.flatten())
    # print("For 1st order diff ---> p_val: " + str(p_val) + " should_diff: " + str(should_diff))
    #
    # p_val, should_diff = adf_test.should_diff(second_order_diffed.flatten())
    # print("For 2nt order diff ---> p_val: " + str(p_val) + " should_diff: " + str(should_diff))
    # y_train_log, _ = LogEndogTransformer(lmbda=1e-6).fit_transform(df0_orig_data)
    # tsdisplay(y_train_log, lag_max=100)
    #
    # y_train_bc, _ = BoxCoxEndogTransformer(lmbda2=1e-6).fit_transform(df0_orig_data)
    # tsdisplay(y_train_bc, lag_max=100)

    tsdisplay(df0_orig_data, title= "unscaled data ", lag_max=100)

    scaler = MaxAbsScaler()
    max_abs_df0_orig_data = scaler.fit_transform(df0_orig_data.reshape(-1, 1))
    tsdisplay(max_abs_df0_orig_data, title= "MaxAbsScaler ", lag_max=100)

    scaler = StandardScaler()
    standard_scaler_df0_orig_data = scaler.fit_transform(df0_orig_data.reshape(-1, 1))
    tsdisplay(standard_scaler_df0_orig_data, title="StandardScaler ", lag_max=100)

    #print("no rescale: ", df0_orig_data)
    #print("rescale: ", df0_orig_data.reshape(-1, 1))

    #todo: period=2000

    #todo: so like around every 8000-9000 periods; somehow need to convert this to frequency / m param
    #todo: 9000 is too long/doesnt work
    #decomposition = seasonal_decompose(df0_orig_data.flatten(), model='additive', period=9000)
    decomposition = seasonal_decompose(df0_orig_data.flatten(), model='additive', period=8500)
    decomposition.plot()
    plt.show()

    decomposition = seasonal_decompose(df0_orig_data.flatten(), model='additive', period=8000)
    decomposition.plot()
    plt.show()

    decomposition = seasonal_decompose(df0_orig_data.flatten(), model='additive', period=4000)
    decomposition.plot()
    plt.show()

    decomposition = seasonal_decompose(df0_orig_data.flatten(), model='additive', period=3000)
    decomposition.plot()
    plt.show()

    decomposition = seasonal_decompose(df0_orig_data.flatten(), model='additive', period=1000)
    decomposition.plot()
    plt.show()

    decomposition = seasonal_decompose(df0_orig_data.flatten(), model='additive', period=500)
    decomposition.plot()
    plt.show()

    decomposition = seasonal_decompose(df0_orig_data.flatten(), model='additive', period=2)
    decomposition.plot()
    plt.show()

    decomposition = seasonal_decompose(df0_orig_data.flatten(), model='additive', period=1)
    decomposition.plot()
    plt.show()
    # decomposition = seasonal_decompose(df0_orig_data.flatten(), model='multiplicative', period=3000)
    # decomposition.plot()
    # plt.show()

    # Create a date range at 10ms intervals starting from an arbitrary time (e.g., '2022-01-01 00:00:00')
    #test_df_1.drop(columns=['Time'], inplace=True)
    #timestamps = pd.date_range(start='2022-01-01', periods=num_samples, freq=sampling_interval)


    #tsdisplay(data_yeojohnson, title="undiffed autocross_valid_run yeo-johnson transform ", lag_max=100)

    scaler = PowerTransformer(method='yeo-johnson')
    data_yeojohnson = scaler.fit_transform(df0_orig_data.reshape(-1, 1))
    tsdisplay(data_yeojohnson, title= "undiffed autocross_valid_run yeo-johnson transform ", lag_max=100)

    # can try maxiter=10 or 20 to increase speed but lose robustness
    # try d=None; data has no autocorrelation at d=1 but autoArima uses other methods than just ADF and ACF and may have a better result
    #TODO: Best model config so far: ARIMA(5,1,2)(0,0,0)[0]     / after longer fit: ARIMA(5,1,1)(0,0,0)[0]

    with StepwiseContext(max_dur=600):

        fit1 = pm.auto_arima(df0_orig_data, m=1, trace=True, suppress_warnings=True, seasonal=True, stepwise=True)
    print(fit1.summary())

    with open('arima.pkl', 'wb') as pkl:
        pickle.dump(fit1, pkl)

    #manual_order = (5, 1, 2)  # Example: p=2, d=1, q=1

    #seasonal_order = (1, 1, 1, 12)
    #model = pm.ARIMA(order=manual_order)                #, seasonal_order=seasonal_order)
    #model.fit(df0_orig_data)
    #forecasts, conf_int = model.predict(n_periods=df0_orig_data.size, return_conf_int=True)
    forecasts, conf_int = fit1.predict(n_periods=df0_orig_data.size, return_conf_int=True)   #Todo: Try plotting confidence intervals over forecast and against true data for AD

    plot_forecasts(df0_orig_data, forecasts, "steer angle orig data + forecast")

    plot_forecasts(first_order_diffed, forecasts, "steer angle diff data + forecast")

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
    #axes[0].set_xlabel(f'y_train (RMSE={np.sqrt(mse(y_test, forecasts)):.3f})')
    axes[0].set_title(title)

    # Plot the residuals
    resid = y_test - forecasts
    _, p = normaltest(resid)
    axes[1].hist(resid, bins=15)
    axes[1].axvline(0, linestyle='--', c='r')
    axes[1].set_title(f'Residuals (p={p:.3f})')

    plt.tight_layout()
    plt.show()