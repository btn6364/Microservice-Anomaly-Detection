import numpy as np
from sklearn.feature_selection import mutual_info_regression
import statistics
import matplotlib.pyplot as plt


# Adds Gaussian noise to data
def gaussian_noise(data):
    """
    Adds Gaussian noise to the data.

    Parameters:
    - data: 1D numpy array of data points

    Returns:
    - data_noised: 1D numpy array with added noise
    """
    noise = np.random.normal(statistics.mean(data), statistics.stdev(data), len(data))
    data_noised = data + noise
    return data_noised


# Calculates Mutual Information
def calculate_mutual_information(anomaly_scores, metrics):
    """
    Calculates Mutual Information between anomaly scores and each metric.

    Parameters:
    - anomaly_scores: 1D numpy array of anomaly scores
    - metrics: 2D numpy array where each column is a metric series

    Returns:
    - mi_scores: List of Mutual Information scores for each metric
    """
    anomaly_scores = np.array(anomaly_scores)
    metrics = np.array(metrics)

    mi_scores = []
    for i in range(metrics.shape[1]):
        mi_score = mutual_info_regression(metrics[:, i].reshape(-1, 1), anomaly_scores)
        mi_scores.append(mi_score[0])

    return mi_scores


# Function to visualize original and noised anomaly scores
def plot_anomaly_scores(original_scores, noised_scores):
    """
    Plots original and noised anomaly scores for comparison.

    Parameters:
    - original_scores: 1D numpy array of original anomaly scores
    - noised_scores: 1D numpy array of anomaly scores with added noise
    """
    x = np.arange(len(original_scores))
    plt.figure(figsize=(10, 5))
    plt.plot(x, original_scores, color="blue", label="Original Anomaly Score")
    plt.plot(x, noised_scores, color="red", label="Anomaly Score with Gaussian Noise")
    plt.title("Anomaly Score Before and After Adding Gaussian Noise")
    plt.xlabel("Time")
    plt.ylabel("Anomaly Score")
    plt.legend()
    plt.savefig('gaussian_anomaly_score.png')
    plt.show()


# Main block for testing the functions
if __name__ == "__main__":
    # Sample data for demonstration
    # Replace with actual data as needed
    anomaly_scores = np.random.rand(100)  # Example anomaly scores
    metrics = np.random.rand(100, 5)  # Example metrics data with 5 different metrics

    # Apply Gaussian noise to anomaly scores
    anomaly_scores_noised = gaussian_noise(anomaly_scores)

    # Calculate Mutual Information scores with the noised anomaly scores
    mi_scores = calculate_mutual_information(anomaly_scores_noised, metrics)
    print("Mutual Information scores for each metric (with noised anomaly scores):")
    for i, score in enumerate(mi_scores):
        print(f"Metric {i + 1}: {score}")

    # Plot original vs. noised anomaly scores
    plot_anomaly_scores(anomaly_scores, anomaly_scores_noised)
