import numpy as np
import matplotlib.pyplot as plt


def plot_normal_vs_noisy_data():
    # Generate normal data (a simple sine wave)
    x = np.linspace(0, 10, 100)
    y_normal = np.sin(x)

    # Generate normal data with added noise
    noise = np.random.normal(0, 0.2, size=x.shape)  # mean=0, std=0.2
    y_noisy = y_normal + noise

    # Create the figure and axes for two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot normal data
    ax1.plot(x, y_normal, color='blue', label='Normal Data')
    ax1.set_title('Normal Data')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.grid(True)
    ax1.legend()

    # Plot normal data with noise
    ax2.plot(x, y_noisy, color='red', label='Normal Data with Noise')
    ax2.set_title('Normal Data with Noise')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.grid(True)
    ax2.legend()

    # Adjust layout for a better look
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_scatter_normal_vs_noisy():
    # Generate normal data (a simple sine wave)
    x = np.linspace(0, 10, 100)
    y_normal = np.sin(x)

    # Generate normal data with added noise
    noise = np.random.normal(0, 0.2, size=x.shape)  # mean=0, std=0.2
    y_noisy = y_normal + noise

    # Create the figure and axes for two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Scatter plot for normal data
    ax1.scatter(x, y_normal, color='blue', label='Normal Data', alpha=0.7)
    ax1.set_title('Normal Data (Scatter)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.grid(True)
    ax1.legend()

    # Scatter plot for normal data with noise
    ax2.scatter(x, y_noisy, color='red', label='Normal Data with Noise', alpha=0.7)
    ax2.set_title('Normal Data with Noise (Scatter)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.grid(True)
    ax2.legend()

    # Adjust layout for a better look
    plt.tight_layout()

    # Show the plots
    plt.show()


def plot_clusters_with_noise():

    np.random.seed(42)

    # Generate two clusters of normal data (black)
    cluster1_x = np.random.normal(1, 2, 100)
    cluster1_y = np.random.normal(1, 2, 100)

    cluster2_x = np.random.normal(14, 2, 100)
    cluster2_y = np.random.normal(14, 2, 100)

    cluster1_x_n = np.random.normal(1, 2, 70)
    cluster1_y_n = np.random.normal(1, 2, 70)

    cluster2_x_n = np.random.normal(14, 2, 70)
    cluster2_y_n = np.random.normal(14, 2, 70)

    # noise_x1 = np.random.uniform(-4, 9, 30)
    # noise_y1 = np.random.uniform(-4, 8, 30)
    # noise_x2 = np.random.uniform(8, 16, 30)
    # noise_y2 = np.random.uniform(7, 15, 30)

    # Generate noise points (black)
    noise_x1 = np.random.uniform(-4, 9, 30)
    noise_y1 = np.random.uniform(-4, 8, 30)
    noise_x2 = np.random.uniform(6, 17, 30)
    noise_y2 = np.random.uniform(7, 15, 30)

    anomaly_x = [7, 5]
    anomaly_y = [7, 13]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.scatter(cluster1_x, cluster1_y, color='black', label='Cluster 1', alpha=0.7)
    ax1.scatter(cluster2_x, cluster2_y, color='black', label='Cluster 2', alpha=0.7)
    ax1.scatter(anomaly_x, anomaly_y, color='red', label='Anomaly', s=100, edgecolor='black')

    for i in range(len(anomaly_x)):
        ax1.text(anomaly_x[i] + 0.2, anomaly_y[i], f'Anomaly {i + 1}', color='red')

    ax1.set_title('Without Noise')
    ax1.set_xlabel('Feature X')
    ax1.set_ylabel('Feature Y')
    ax1.grid(True)

    # Scatter plot for the two clusters (black) and noise (black)
    ax2.scatter(cluster1_x_n, cluster1_y_n, color='black', label='Cluster 1', alpha=0.7)
    ax2.scatter(cluster2_x_n, cluster2_y_n, color='black', label='Cluster 2', alpha=0.7)
    ax2.scatter(noise_x1, noise_y1, color='black', label='Noise', alpha=0.7)
    ax2.scatter(noise_x2, noise_y2, color='black', label='Noise', alpha=0.7)

    ax2.scatter(anomaly_x, anomaly_y, color='red', label='Anomaly', s=100, edgecolor='black')
    for i in range(len(anomaly_x)):
        ax2.text(anomaly_x[i] + 0.2, anomaly_y[i], f'Anomaly {i + 1}', color='red')

    ax2.set_title('With Noise')
    ax2.set_xlabel('Feature X')
    ax2.set_ylabel('Feature Y')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("./exampleGraphs/scatterplot_with_and_without_noise.png", format='png', dpi=300)  # You can adjust the format and dpi as needed

    plt.show()
