import random

import pandas as pd
from sklearn.preprocessing import MaxAbsScaler


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