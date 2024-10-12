import os
import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MaxAbsScaler

from data_processing import filter_df_by_start_and_end_time_of_activity_phase, clean_csv


def add_anomalies_and_drift(num_anomalies, csv_file, output_file):
    # Step 1: Load the data
    df = pd.read_csv(csv_file)
    columns_to_remove = ['header.stamp.secs', 'header.stamp.nsecs', 'header.frame_id', 'child_frame_id',
                         'twist.covariance',
                         'header.seq']
    for col in columns_to_remove:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    # Step 2: Introduce anomalies
    def introduce_anomalies(df, num_anomalies):
        for _ in range(num_anomalies):
            anomaly_factor = 0
            while anomaly_factor == 0:
                anomaly_factor = random.randint(-20, 20)

            row_idx = random.randint(0, len(df) - 1)
            col_idx = random.randint(1, df.shape[1] - 1 - 1) #for time and anomaly column
            original_value = df.iloc[row_idx, col_idx]
            if bool(random.getrandbits(1)):
                anomaly = original_value * anomaly_factor  # or use a random value
            else:
                anomaly = original_value / anomaly_factor  # or use a random value

            df.iloc[row_idx, col_idx] = anomaly
            df.iloc[row_idx, df.shape[1] - 1] = 1
        return df

    # Step 3: Introduce moderate concept drift
    def introduce_moderate_concept_drift(df, start_row, end_row, columns, drift_factor=1.05):
        for col in columns:
            if col != 'Anomaly' and col != 'Time':
                for row_idx in range(start_row, end_row):
                    df.at[row_idx, col] *= drift_factor  # Slightly increase values
        return df

    # Step 4: Introduce strong concept drift
    def introduce_strong_concept_drift(df, start_row, end_row, columns, drift_factor=1.5):
        for col in columns:
            if col != 'Anomaly' and col != 'Time':
                for row_idx in range(start_row, end_row):
                    df.at[row_idx, col] *= drift_factor  # Significantly increase values
        return df

    # Apply these transformations
    df = introduce_anomalies(df, num_anomalies)

    # Moderate concept drift in the middle of the data

    middle_start = random.randint(0, len(df) // 3)  #len(df) // 3
    middle_end = 2 * middle_start
    df = introduce_moderate_concept_drift(df, middle_start, middle_end, df.columns)

    # Strong concept drift towards the end of the data
    strong_drift_start = random.randint(middle_end, random.randint(middle_end, len(df)))
    df = introduce_strong_concept_drift(df, strong_drift_start, len(df), df.columns)

    # Step 5: Save the modified DataFrame to a new CSV file
    df.to_csv(output_file, index=False)

    print(f"Anomalies and concept drift added to {output_file}")


# def add_coll_anomalies(dir_path, sensor_name, col_name, collective_len, output_dir="./injectedAnomalyData/"):
#
#     _, df = filter_df_by_start_and_end_time_of_activity_phase(dir_path, False, control_acc_filename="control-acceleration.csv", target_df_filename=sensor_name)
#
#     start_idx = random.randint(0, len(df) - collective_len - 1)
#     end_idx = start_idx + collective_len
#
#     # Add collective anomalies by replacing a series of points with unusual values
#     df.loc[start_idx:start_idx + collective_len, col_name] = df[col_name].mean() + 5 * df[col_name].std()
#
#     df.loc[start_idx:end_idx, 'Anomaly'] = 1
#
#
#     output_file = os.path.join(output_dir, sensor_name)
#     df.to_csv(output_file, index=False)
#     print(f"File saved with injected anomalies to {output_file}")
#

def add_contextual_anomalies(dir_path, sensor_name, col_name, contextual_len, output_dir="./injectedAnomalyData/"):

    full_df = clean_csv(dir_path + sensor_name, False)

    _, cut_df = filter_df_by_start_and_end_time_of_activity_phase(dir_path, False, control_acc_filename="control-acceleration.csv", target_df_filename=sensor_name)

    min_index = cut_df.index.min()
    max_index = cut_df.index.max()
    if max_index - min_index < contextual_len:
        raise ValueError("The range between min and max index is smaller than the contextual anomaly length.")

    src_start_idx = random.randint(min_index, max_index - contextual_len - 1)
    src_end_idx = src_start_idx + contextual_len

    anomaly_start_idx = random.randint(min_index, max_index - contextual_len - 1)
    anomaly_end_idx = anomaly_start_idx + contextual_len

    mirrored_subseq = full_df[col_name].iloc[src_start_idx:src_end_idx].copy()

    #noise = np.random.normal(loc=0.0, scale=0.1, size=contextual_len)
    #modified_subseq = mirrored_subseq + noise

    #modified_subseq = np.flip(mirrored_subseq) * 1.5 + np.random.normal(loc=0.0, scale=0.5, size=contextual_len)
    modified_subseq = np.flip(mirrored_subseq) * 1.3 + np.random.normal(loc=0.0, scale=0.5, size=contextual_len)

    full_df.loc[anomaly_start_idx: (anomaly_start_idx + len(modified_subseq) - 1), col_name] = modified_subseq.values
    full_df.loc[anomaly_start_idx: (anomaly_start_idx + len(modified_subseq) - 1), 'Anomaly'] = 1

    output_file = os.path.join(output_dir, sensor_name)
    full_df.to_csv(output_file, index=False)
    print(f"File saved with injected anomalies to {output_file}")


def plot_normal_vs_injected_anomalies(normalSource_path, injectAnom_path, sensor_name, columns_to_plot=None):
    _, normal_df = filter_df_by_start_and_end_time_of_activity_phase(normalSource_path, True,
                                                                     control_acc_filename="control-acceleration.csv",
                                                                     target_df_filename=sensor_name)

    _, anomaly_df = filter_df_by_start_and_end_time_of_activity_phase(injectAnom_path, True,
                                                                      control_acc_filename="control-acceleration.csv",
                                                                      target_df_filename=sensor_name)

    if columns_to_plot is None:
        columns_to_plot = [col for col in normal_df.columns if col != 'Anomaly']

    colors = plt.cm.jet(np.linspace(0, 1, len(columns_to_plot)))
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

    ax1 = axes[0]
    for i, col in enumerate(columns_to_plot):
        ax1.plot(normal_df.index, normal_df[col], color=colors[i])
    ax1.set_title('Normal Data')
    ax1.set_xlabel('Consecutive Measurements')
    ax1.set_ylabel('Values')
    ax1.grid(True)
    ax2 = axes[1]

    for i, col in enumerate(columns_to_plot):
        ax2.plot(anomaly_df.index, anomaly_df[col], color=colors[i])

    if 'Anomaly' in anomaly_df.columns:
        anomalies = anomaly_df[anomaly_df['Anomaly'] == 1]
        for col in columns_to_plot:
            anomaly_regions = np.split(anomalies.index, np.where(np.diff(anomalies.index) != 1)[0] + 1)
            for region in anomaly_regions:
                if len(region) > 0:
                    ax2.axvspan(region[0], region[-1], color='red', alpha=0.3)

    ax2.set_title('Injected Anomalies Data')
    ax2.set_xlabel('Consecutive Measurements')
    ax2.set_ylabel('Values')
    ax2.grid(True)

    fig.suptitle(f'Comparison of Normal and Injected Anomalies for {sensor_name}')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("./injectedAnomalyData/Normal_vs_anom_plot" + '.png')
    plt.show()


def add_flip_anomalies(dir_path, sensor_name, col_name, num_anomalies, output_dir="./injectedAnomalyData/"):
    full_df = clean_csv(dir_path + sensor_name, False)

    _, cut_df = filter_df_by_start_and_end_time_of_activity_phase(dir_path, False,
                                                                  control_acc_filename="control-acceleration.csv",
                                                                  target_df_filename=sensor_name)

    valid_indices = cut_df[(cut_df[col_name] > 1.5) | (cut_df[col_name] < -1.5)]
    valid_indices = valid_indices[(valid_indices[col_name] >= -5) & (valid_indices[col_name] <= 5)].index.tolist()

    if len(valid_indices) < num_anomalies:
        raise ValueError(f"Not enough valid points to insert {num_anomalies} anomalies in the cut dataframe range.")
    selected_indices = random.sample(valid_indices, num_anomalies)
    for idx in selected_indices:
        full_df.at[idx, col_name] = -full_df.at[idx, col_name]

    full_df.loc[selected_indices, 'Anomaly'] = 1

    output_file = os.path.join(output_dir, sensor_name)
    full_df.to_csv(output_file, index=False)
    print(f"File saved with injected anomalies to {output_file}")



if __name__ == '__main__':
    src_dir = './aufnahmen/csv/anomalous data/'
    sensor_name = 'can_interface-current_steering_angle.csv'
    col_name = 'data'
    injected_anoms_dir = './injectedAnomalyData/'

    contextual_len = 300

    #add_contextual_anomalies(src_dir, sensor_name, col_name, contextual_len, output_dir="./injectedAnomalyData/")
    add_flip_anomalies(src_dir, sensor_name, col_name, 50, output_dir="./injectedAnomalyData/")

    plot_normal_vs_injected_anomalies(src_dir, injected_anoms_dir, sensor_name)

        #todo: Kanns doch über Torch Main machen; Index wird nicht abeschnitten; muss nur original index in Anom Injec nehmen