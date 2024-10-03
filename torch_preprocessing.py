import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import kurtosis

from data_processing import clean_csv, convert_timestamp_to_relative_time_diff


#############################################################################################
#   File containing the method use for resampling the time series before training the model #
#############################################################################################

def preprocessing(input, list_transfo, resample_freq, size_window, step_window):
    """
    Preprocessing take a pandas DataFrame as input with time_stamp as cols.
    :param input: DataFrame containing the data
    :param list_transfo: list of string containing the transformation to apply
    :param resample_freq: Resample frequency in order to apply the transformation
    :param size_window: size of sequence output
    :param step_window: step of the moving window creating the output sequence
    :return:  a 3d numpy array with newly created sequence of the data on each feature in list_transfo.
    """
    #input = input.T
    data = input.copy()
    #data = data.T
    print("original: \n" + str(input))
    print("original shape: " + str(input.shape))
    #print("here2: " + str(input[0]))
    #print("here2: " + str(input[1]))
    #print("here2: " + str(input[2]))
    print("transposed: \n" + str(data))
    print("transposed shape: \n" + str(data.shape))

    #todo: his whole dataset might have been column by column but mine is row by row
    data.index = pd.date_range(start='1/1/2018', periods=input.shape[0], freq='L')      #todo: changed  periods=input.shape[1] to periods=input.shape[0] as my time points are in rows, not cols
    #data = data.T
    print("here4: \n" + str(data))

    list_var = []
    if ('mean' in list_transfo):
        list_var.append(create_seq(data.resample(resample_freq + 'L').mean(), step_window, size_window))
    if ('max' in list_transfo):
        list_var.append(create_seq(data.resample(resample_freq + 'L').max(), step_window, size_window))
    if ('min' in list_transfo):
        list_var.append(create_seq(data.resample(resample_freq + 'L').min(), step_window, size_window))
    if ('trend' in list_transfo):
        trend = input.apply(lambda x: seasonal_decompose(x, model='additive', period=1024).trend[::int(resample_freq)],
                            axis=0).fillna(method='ffill', axis=1).fillna(method='bfill', axis=0).T
        trend.index = pd.date_range(start='1/1/2018', periods=trend.shape[0], freq='L')
        list_var.append(create_seq(trend, step_window, size_window))
    if ('kurtosis' in list_transfo):
        list_var.append(
            create_seq(data.resample(resample_freq + 'L').apply(lambda x: kurtosis(x)), step_window, size_window))
    if ('max_diff_var' in list_transfo):
        data_resample_var = data.resample('30L').var()
        diff_data_resample_var = (data_resample_var - data_resample_var.shift(-1)).interpolate()
        list_var.append(
            create_seq(diff_data_resample_var.resample(resample_freq + 'L').max(), step_window, size_window))
    if ('var_var' in list_transfo):
        list_var.append(
            create_seq(data.resample('30L').var().resample(resample_freq + 'L').var(), step_window, size_window))
    if ('level_shift' in list_transfo):
        data_resample_mean = data.resample('30L').mean()
        level_shift = (data_resample_mean - data_resample_mean.shift(-1)).interpolate()
        list_var.append(create_seq(level_shift.resample(resample_freq + 'L').max(), step_window, size_window))
    return regroup_multivariate_ts(list_var)


def create_seq(data, step_window, size_window):
    """
    Take a pandas DataFrame with time index as input and return a (size_window * )
    :return: numpy array of ((size_ts-size_window) / step_window, size_window)
    """
    time_lenght, nb_time_series = data.shape                                    #The create_seq function is used to generate a sliding window of sequences from the resampled data.
    additional_ts = len(range(0, time_lenght - size_window + 1, step_window))   #number of new windows
    output = np.zeros((nb_time_series, additional_ts, size_window))
    seq = pd.DataFrame(data.iloc[:size_window, 0])
    for j, shift_value in enumerate(range(0, time_lenght - size_window + 1, step_window)):
        data_shift = data.shift(-shift_value)
        for i in range(0, nb_time_series):
            output[i, j, :] = data_shift.iloc[:size_window, i]
    return output.reshape(-1, size_window)


def regroup_multivariate_ts(list_ts):
    """
    Create a 3 dimensional array from a list of time serie of same size
    :param list_ts: list of list of 2 dimensional time series sequence to concatenate in a third dimension
    """
    print("a: " + str(len(list_ts)))
    print("aa: " + str(list_ts[0].shape))
    print("aaa: " + str(list_ts[1].shape))

    for i, ts in enumerate(list_ts):
        if ts.shape[0] == 0:
            print(f"Warning: Skipping empty array at index {i}")
            continue
    for i, ts in enumerate(list_ts):
        print(str(i) + ": " + str(ts.shape))

    global_array = np.empty(shape=(list_ts[0].shape[0], list_ts[0].shape[1], len([ts for ts in list_ts if ts.shape[0] > 0])))
    print("aaaa: " + str(global_array.shape))

    valid_index = 0
    for i, ts in enumerate(list_ts):
        if ts.shape[0] == 0:
            continue  # Skip the empty arrays
        global_array[:, :, valid_index] = ts
        valid_index += 1
    return global_array
