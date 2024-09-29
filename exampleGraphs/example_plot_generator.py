import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_contextual_anomaly():
    # Generate time series data for 7 days (168 hours)
    np.random.seed(42)  # For reproducibility
    hours = np.arange(0, 144)

    # Simulate electricity usage with seasonality (peaks in the morning and evening)
    daily_pattern = np.array([5] * 6 + [20] * 3 + [10] * 6 + [30] * 3 + [5] * 6)
    weekly_pattern = np.tile(daily_pattern, 6)  # Repeat the daily pattern for 7 days

    # Add some random noise to simulate variability
    noise = np.random.normal(0, 2, weekly_pattern.shape)
    data = weekly_pattern + noise

    # Introduce the contextual anomaly: higher values during one specific night
    data[24 * 3 + 0: 24 * 3 + 4] = data[24 * 3 + 17: 24 * 3 + 21]  # Day 4 from 00:00 to 04:00

    # Create a DataFrame to manage timestamps
    timestamps = pd.date_range(start='2023-01-01', periods=144, freq='H')
    df = pd.DataFrame({'Time': timestamps, 'Electricity Usage': data})

    # Plotting the time series data
    plt.figure(figsize=(14, 7))
    plt.plot(df['Time'], df['Electricity Usage'], color='blue', label='Normal Usage')

    # Highlight the anomalous sequence in red
    plt.plot(df['Time'][24 * 3 + 0 - 2: 24 * 3 + 4], df['Electricity Usage'][24 * 3 + 0 - 2: 24 * 3 + 4],
             color='red', linewidth=3, label='Anomalous Usage (22:00-04:00)')

    plt.axvspan(df['Time'][24 * 3 + 0 - 2], df['Time'][24 * 3 + 4 - 1], color='red', alpha=0.3, label='Anomalous Period')

    # Customize the plot
    plt.title('Contextual Anomalies in Time Series Data: Electricity Usage')
    plt.xlabel('Time')
    plt.ylabel('Average Electricity Usage')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("./exampleGraphs/contextual_anomaly.png", format='png', dpi=300)  # You can adjust the format and dpi as needed

    # Show the plot
    plt.show()


def plot_collective_anomaly_similar():
    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate a time series with a periodic pattern (sinusoidal waves)
    time = np.linspace(0, 3000, 3000)
    #normal_pattern = np.sin(time / 4) * 2
    normal_pattern = np.random.normal(4, 0.8, 3000)

    # Add more pronounced noise
    noise_amplitude = 1.5  # Larger noise amplitude for more pronounced effect
    noise = np.random.normal(0, 0.5, 3000)  # Basic noise

    # Modify noise to be more negative in the first half and more positive in the second half
    noise[:1500] -= np.linspace(0.5, 0.8, 1500)  # Shift noise downward in the first half
    noise[1500:] += np.linspace(0.5, 0.8, 1500)  # Shift noise upward in the second half

    # Add noise to the normal pattern
    normal_pattern += noise_amplitude * noise

    # Introduce a collective anomaly: a sequence of decreasing values
    anomaly_start = 1000
    anomaly_end = 2000
    anomaly_pattern = normal_pattern.copy()

    # Create a sequence with smaller random values to simulate the anomaly (flat-ish, decreasing)
    normal_pattern[anomaly_start:anomaly_end] = np.linspace(4,
                                                             4.6,
                                                             anomaly_end - anomaly_start)

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot the normal pattern in blue
    plt.plot(time, normal_pattern, label='Normal Pattern', color='blue')

    # Highlight the collective anomaly in red (decreasing values)
    plt.plot(time[anomaly_start:anomaly_end], normal_pattern[anomaly_start:anomaly_end],
             label='Collective Anomaly', color='red', linewidth=2)

    # Labels and title
    plt.xlabel('Time')
    plt.ylabel('Feature Y')
    plt.title('Collective Anomaly in Time Series Data')
    plt.legend()
    #plt.ylim(-5, 6)
    plt.tight_layout()
    plt.savefig("./exampleGraphs/collective_anomaly.png", format='png', dpi=300)  # You can adjust the format and dpi as needed
    plt.show()

def plot_point_anomaly():
    np.random.seed(42)

    # Generate a time series with a periodic pattern (sinusoidal waves)
    time = np.linspace(0, 3000, 3000)
    # normal_pattern = np.sin(time / 4) * 2
    normal_pattern = np.random.normal(3, 0.6, 3000)

    # Add more pronounced noise
    noise_amplitude = 1.5  # Larger noise amplitude for more pronounced effect
    noise = np.random.normal(0, 0.5, 3000)  # Basic noise

    # Modify noise to be more negative in the first half and more positive in the second half
    noise[::] += np.linspace(0.5, 0.8, 3000)  # Shift noise downward in the first half

    # Add noise to the normal pattern
    normal_pattern += noise_amplitude * noise

    # Introduce a collective anomaly: a sequence of decreasing values
    anomaly_start = 1500
    anomaly_end = anomaly_start + 1

    # Create a sequence with smaller random values to simulate the anomaly (flat-ish, decreasing)
    normal_pattern[anomaly_start:anomaly_end] = 12

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot the normal pattern in blue
    plt.plot(time, normal_pattern, label='Normal Pattern', color='blue')

    # Highlight the collective anomaly in red (decreasing values)
    plt.plot(time[anomaly_start:anomaly_end], normal_pattern[anomaly_start:anomaly_end], color='red', marker = 'o', markerfacecolor = 'none', markersize = 10, linewidth=2)

    # Labels and title
    legend_elements = [
        Line2D([0], [0], color='blue', label='Normal Pattern'),
        Line2D([0], [0], marker='o', color='r', label='Point Anomaly', markerfacecolor='none', markersize=10)
    ]

    plt.xlabel('Time')
    plt.ylabel('Feature Y')
    plt.title('Collective Anomaly in Time Series Data')
    plt.legend(handles=legend_elements, loc='upper right')
    # plt.ylim(-5, 6)
    plt.tight_layout()
    plt.savefig("./exampleGraphs/point_anomaly.png", format='png',
                dpi=300)
    plt.show()