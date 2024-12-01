import numpy as np
from sklearn.feature_selection import mutual_info_regression
import statistics
import matplotlib.pyplot as plt
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler


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
def calculate_mutual_information(anomaly_scores, metrics, metric_names):
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
    mi_scores = {}
    for i in range(metrics.shape[1]):
        metric_name = metric_names[i]
        mi_score = mutual_info_regression(metrics[:, i].reshape(-1, 1), anomaly_scores, n_neighbors=2)
        mi_scores[metric_name] = mi_score[0]
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

def generate_metrics():
    # Collect node_cpu_seconds_total, node_memory_MemAvailable_bytes, node_memory_MemTotal_bytes
    filenames = [
        "anomalies_train_ticket/ts-admin-basic-info-service-sprintstarterweb_1.5.22/Monitoring_ts-admin-basic-info-service_springstarterweb_1.5.22.RELEASE.json_2022-07-08/ts-admin-basic-info-service_springstarterweb_1.5.22.RELEASE.json_node_cpu_seconds_total.json",
        "anomalies_train_ticket/ts-auth-mongo_4.4.15_2022-07-13/Monitoring_ts-auth-service_3_Mongo_4.4.15.json_2022-07-13/ts-auth-service_3_Mongo_4.4.15.json_node_cpu_seconds_total.json",
        "anomalies_train_ticket/ts-order-service_mongodb_4.2.2_2022-07-12/Monitoring_ts-order-service_mongodb_4.2.2.json_2022-07-12/ts-order-service_mongodb_4.2.2.json_node_cpu_seconds_total.json"
    ]
    metric_name_1, metric_values_1 = generate_metric_values(filenames)

    # Collect node_cpu_seconds_total, node_memory_MemAvailable_bytes, node_memory_MemTotal_bytes
    filenames = [
        "anomalies_train_ticket/ts-admin-basic-info-service-sprintstarterweb_1.5.22/Monitoring_ts-admin-basic-info-service_springstarterweb_1.5.22.RELEASE.json_2022-07-08/ts-admin-basic-info-service_springstarterweb_1.5.22.RELEASE.json_node_memory_MemAvailable_bytes.json",
        "anomalies_train_ticket/ts-auth-mongo_4.4.15_2022-07-13/Monitoring_ts-auth-service_3_Mongo_4.4.15.json_2022-07-13/ts-auth-service_3_Mongo_4.4.15.json_node_memory_MemAvailable_bytes.json",
        "anomalies_train_ticket/ts-order-service_mongodb_4.2.2_2022-07-12/Monitoring_ts-order-service_mongodb_4.2.2.json_2022-07-12/ts-order-service_mongodb_4.2.2.json_node_memory_MemAvailable_bytes.json"
    ]
    metric_name_2, metric_values_2 = generate_metric_values(filenames)

    # Collect node_cpu_seconds_total, node_memory_MemAvailable_bytes, node_memory_MemTotal_bytes
    filenames = [
        "anomalies_train_ticket/ts-admin-basic-info-service-sprintstarterweb_1.5.22/Monitoring_ts-admin-basic-info-service_springstarterweb_1.5.22.RELEASE.json_2022-07-08/ts-admin-basic-info-service_springstarterweb_1.5.22.RELEASE.json_node_memory_MemTotal_bytes.json",
        "anomalies_train_ticket/ts-auth-mongo_4.4.15_2022-07-13/Monitoring_ts-auth-service_3_Mongo_4.4.15.json_2022-07-13/ts-auth-service_3_Mongo_4.4.15.json_node_memory_MemTotal_bytes.json",
        "anomalies_train_ticket/ts-order-service_mongodb_4.2.2_2022-07-12/Monitoring_ts-order-service_mongodb_4.2.2.json_2022-07-12/ts-order-service_mongodb_4.2.2.json_node_memory_MemTotal_bytes.json"
    ]
    metric_name_3, metric_values_3 = generate_metric_values(filenames)

    filenames = [
        "anomalies_train_ticket/ts-admin-basic-info-service-sprintstarterweb_1.5.22/Monitoring_ts-admin-basic-info-service_springstarterweb_1.5.22.RELEASE.json_2022-07-08/ts-admin-basic-info-service_springstarterweb_1.5.22.RELEASE.json_node_network_transmit_packets_total.json",
        "anomalies_train_ticket/ts-auth-mongo_4.4.15_2022-07-13/Monitoring_ts-auth-service_3_Mongo_4.4.15.json_2022-07-13/ts-auth-service_3_Mongo_4.4.15.json_node_network_transmit_packets_total.json",
        "anomalies_train_ticket/ts-order-service_mongodb_4.2.2_2022-07-12/Monitoring_ts-order-service_mongodb_4.2.2.json_2022-07-12/ts-order-service_mongodb_4.2.2.json_node_network_transmit_packets_total.json"
    ]
    metric_name_4, metric_values_4 = generate_metric_values(filenames)

    filenames = [
        "anomalies_train_ticket/ts-admin-basic-info-service-sprintstarterweb_1.5.22/Monitoring_ts-admin-basic-info-service_springstarterweb_1.5.22.RELEASE.json_2022-07-08/ts-admin-basic-info-service_springstarterweb_1.5.22.RELEASE.json_container_cpu_usage_seconds_total.json",
        "anomalies_train_ticket/ts-auth-mongo_4.4.15_2022-07-13/Monitoring_ts-auth-service_3_Mongo_4.4.15.json_2022-07-13/ts-auth-service_3_Mongo_4.4.15.json_container_cpu_usage_seconds_total.json",
        "anomalies_train_ticket/ts-order-service_mongodb_4.2.2_2022-07-12/Monitoring_ts-order-service_mongodb_4.2.2.json_2022-07-12/ts-order-service_mongodb_4.2.2.json_container_cpu_usage_seconds_total.json"
    ]
    metric_name_5, metric_values_5 = generate_metric_values(filenames)

    filenames = [
        "anomalies_train_ticket/ts-admin-basic-info-service-sprintstarterweb_1.5.22/Monitoring_ts-admin-basic-info-service_springstarterweb_1.5.22.RELEASE.json_2022-07-08/ts-admin-basic-info-service_springstarterweb_1.5.22.RELEASE.json_container_memory_working_set_bytes.json",
        "anomalies_train_ticket/ts-auth-mongo_4.4.15_2022-07-13/Monitoring_ts-auth-service_3_Mongo_4.4.15.json_2022-07-13/ts-auth-service_3_Mongo_4.4.15.json_container_memory_working_set_bytes.json",
        "anomalies_train_ticket/ts-order-service_mongodb_4.2.2_2022-07-12/Monitoring_ts-order-service_mongodb_4.2.2.json_2022-07-12/ts-order-service_mongodb_4.2.2.json_container_memory_working_set_bytes.json"
    ]
    metric_name_6, metric_values_6 = generate_metric_values(filenames)

    filenames = [
        "anomalies_train_ticket/ts-admin-basic-info-service-sprintstarterweb_1.5.22/Monitoring_ts-admin-basic-info-service_springstarterweb_1.5.22.RELEASE.json_2022-07-08/ts-admin-basic-info-service_springstarterweb_1.5.22.RELEASE.json_container_network_transmit_packets_total.json",
        "anomalies_train_ticket/ts-auth-mongo_4.4.15_2022-07-13/Monitoring_ts-auth-service_3_Mongo_4.4.15.json_2022-07-13/ts-auth-service_3_Mongo_4.4.15.json_container_network_transmit_packets_total.json",
        "anomalies_train_ticket/ts-order-service_mongodb_4.2.2_2022-07-12/Monitoring_ts-order-service_mongodb_4.2.2.json_2022-07-12/ts-order-service_mongodb_4.2.2.json_container_network_transmit_packets_total.json"
    ]
    metric_name_7, metric_values_7 = generate_metric_values(filenames)

    metric_map = {
        metric_name_1: metric_values_1, 
        metric_name_2: metric_values_2, 
        metric_name_3: metric_values_3, 
        metric_name_4: metric_values_4,
        metric_name_5: metric_values_5,
        metric_name_6: metric_values_6,
        metric_name_7: metric_values_7
    }
    # The result with a 3x3 matrix
    #           metric1   metric2   metric3
    # sample1      -        -          -
    # sample2      -        -          -
    # sample3      -        -          -
    df = pd.DataFrame.from_dict(metric_map)
    return df.to_numpy(), df.columns

def generate_metric_values(filenames):
    metric_name = None
    metric_values_and_timestamp = []
    for filename in filenames:
        # Open and read the JSON file
        with open(filename, 'r') as file:
            data = json.load(file)["data"]

        # Print the data
        metric_name = data["result"][0]["metric"]["__name__"]
        microservice_metric_values_and_timestamp = data["result"][0]["values"]
        metric_values_and_timestamp.extend(microservice_metric_values_and_timestamp)
    
    # Sort based on timestamp
    metric_values_and_timestamp.sort(key=lambda x: x[0])

    # Generate 3 mean values
    metric_values = [float(metric_value) for _, metric_value in metric_values_and_timestamp]
    first_mean_metric_value = int(statistics.mean(metric_values[:len(metric_values) // 3]))
    second_mean_metric_value = int(statistics.mean(metric_values[len(metric_values) // 3: 2 * len(metric_values) // 3]))
    third_mean_metric_value = int(statistics.mean(metric_values[2 * len(metric_values) // 3:]))
    mean_metric_values = [
        first_mean_metric_value, 
        second_mean_metric_value, 
        third_mean_metric_value
    ]
    
    # Each metric can only get 3 values since we only have 3 anomaly scores
    return metric_name, mean_metric_values

def generate_anomaly_scores():
    """
    Number of sessions(hdfs_logs_admin_basic): 5607
    Microservice Anomaly Score = 0.999
    Number of sessions(hdfs_logs_auth): 9217
    Microservice Anomaly Score = 0.999
    Number of sessions(hdfs_logs_order): 8069
    Microservice Anomaly Score = 0.993
    """
    anomaly_scores = np.array([0.999, 0.999, 0.993])
    return anomaly_scores

# Main block for testing the functions
if __name__ == "__main__":
    # Sample data for demonstration
    # Replace with actual data as needed
    anomaly_scores = generate_anomaly_scores()
    metrics, metric_names = generate_metrics()
    # print(metrics)

    # Apply Gaussian noise to anomaly scores
    anomaly_scores_noised = gaussian_noise(anomaly_scores)

    # Calculate Mutual Information scores with the noised anomaly scores
    mi_scores = calculate_mutual_information(anomaly_scores_noised, metrics, metric_names)
    print("Mutual Information scores for each metric (with noised anomaly scores):")
    for metric, score in mi_scores.items():
        print(f"Metric {metric}: {score}")

    # Plot original vs. noised anomaly scores
    plot_anomaly_scores(anomaly_scores, anomaly_scores_noised)

    
