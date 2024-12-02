import numpy as np
import DeepLog.find_anomaly_score as AS
import pandas as pd
import json
from datetime import datetime


def epsilon_diagnosis(normal_set, abnormal_set):
    # normal set and abnormal set are lists of a given metric measurement in float format
    if len(normal_set) == 0 or len(abnormal_set) == 0:
        return 0
    ab_var = np.var(abnormal_set)
    norm_var = np.var(normal_set)
    if ab_var == 0 or norm_var == 0:
        return 0
    print(f"normal_size = {len(normal_set)}, abnormal_size = {len(abnormal_set)}")
    normal_set = normal_set[:min(len(normal_set), len(abnormal_set))]
    abnormal_set = abnormal_set[:min(len(normal_set), len(abnormal_set))]
    covariance = np.cov(normal_set, abnormal_set, ddof=1)[0, 1]
    result = (covariance ** 2) / np.sqrt(ab_var * norm_var)
    return result


def regression_based_analysis(anomaly_score_vector, metric_matrix):
    # the score vector is a list of the anomaly scores for a given service
    # metric matrix consists of all the metric values for the given service
    beta_vector = np.linalg.inv(metric_matrix.T @ metric_matrix) @ metric_matrix.T @ anomaly_score_vector
    return beta_vector[1:]  # beta[0] is the intercept (alpha) value


def correlating_with_time_series(normal_set, abnormal_set):
    # normal set and abnormal set are lists of a given metric measurement in float format
    combined_set = []
    for i in range(len(normal_set)):
        combined_set.append(("N", normal_set[i]))
    for i in range(len(abnormal_set)):
        combined_set.append(("A", abnormal_set[i]))

    p = len(combined_set)
    r = 3
    total_sum = 0
    for i in range(p):
        for j in range(1, r + 1):
            st = combined_set[i]
            same_set = rth_nearest_neighbor(st, combined_set, j)
            total_sum += same_set
    return total_sum / (p * r)


def rth_nearest_neighbor(item, combined_set, r):
    # calculates Ir function from paper
    item_class = item[0]
    # Calculate 1D euclidean distances
    distances = [(cls, abs(value - item)) for cls, value in combined_set]
    sorted_distances = sorted(distances, key=lambda x: x[1])

    class_str, distance = sorted_distances[r]  # would be r - 1 but item is in the set as well
    if class_str == item_class:
        return 1
    else:
        return 0


def generate_service_metrics():
    base_admin_basic_filename = "anomalies_train_ticket/ts-admin-basic-info-service-sprintstarterweb_1.5.22/Monitoring_ts-admin-basic-info-service_springstarterweb_1.5.22.RELEASE.json_2022-07-08/ts-admin-basic-info-service_springstarterweb_1.5.22.RELEASE."
    base_auth_filename = "anomalies_train_ticket/ts-auth-mongo_4.4.15_2022-07-13/Monitoring_ts-auth-service_3_Mongo_4.4.15.json_2022-07-13/ts-auth-service_3_Mongo_4.4.15."
    base_order_filename = "anomalies_train_ticket/ts-order-service_mongodb_4.2.2_2022-07-12/Monitoring_ts-order-service_mongodb_4.2.2.json_2022-07-12/ts-order-service_mongodb_4.2.2."

    # node_cpu_seconds_total
    filenames = [
        base_admin_basic_filename + "json_node_cpu_seconds_total.json",
        base_auth_filename + "json_node_cpu_seconds_total.json",
        base_order_filename + "json_node_cpu_seconds_total.json"
    ]
    metric_name_1, metric_values_1 = get_all_metric_values(filenames)

    # node_memory_MemAvailable_bytes
    filenames = [
        base_admin_basic_filename + "json_node_memory_MemAvailable_bytes.json",
        base_auth_filename + "json_node_memory_MemAvailable_bytes.json",
        base_order_filename + "json_node_memory_MemAvailable_bytes.json"
    ]
    metric_name_2, metric_values_2 = get_all_metric_values(filenames)

    # node_memory_MemTotal_bytes
    filenames = [
        base_admin_basic_filename + "json_node_memory_MemTotal_bytes.json",
        base_auth_filename + "json_node_memory_MemTotal_bytes.json",
        base_order_filename + "json_node_memory_MemTotal_bytes.json"
    ]
    metric_name_3, metric_values_3 = get_all_metric_values(filenames)

    # node_network_transmit_packets_total
    filenames = [
        base_admin_basic_filename + "json_node_network_transmit_packets_total.json",
        base_auth_filename + "json_node_network_transmit_packets_total.json",
        base_order_filename + "json_node_network_transmit_packets_total.json"
    ]
    metric_name_4, metric_values_4 = get_all_metric_values(filenames)

    # container_cpu_usage_seconds_total
    filenames = [
        base_admin_basic_filename + "json_container_cpu_usage_seconds_total.json",
        base_auth_filename + "json_container_cpu_usage_seconds_total.json",
        base_order_filename + "json_container_cpu_usage_seconds_total.json"
    ]
    metric_name_5, metric_values_5 = get_all_metric_values(filenames)

    # container_memory_working_set_bytes
    filenames = [
        base_admin_basic_filename + "json_container_memory_working_set_bytes.json",
        base_auth_filename + "json_container_memory_working_set_bytes.json",
        base_order_filename + "json_container_memory_working_set_bytes.json"
    ]
    metric_name_6, metric_values_6 = get_all_metric_values(filenames)

    # json_container_network_transmit_packets_total.json
    filenames = [
        base_admin_basic_filename + "json_container_network_transmit_packets_total.json",
        base_auth_filename + "json_container_network_transmit_packets_total.json",
        base_order_filename + "json_container_network_transmit_packets_total.json"
    ]
    metric_name_7, metric_values_7 = get_all_metric_values(filenames)

    metric_map = {
        metric_name_1: metric_values_1,
        metric_name_2: metric_values_2,
        metric_name_3: metric_values_3,
        metric_name_4: metric_values_4,
        metric_name_5: metric_values_5,
        metric_name_6: metric_values_6,
        metric_name_7: metric_values_7
    }

    df = pd.DataFrame.from_dict(metric_map, orient='index').transpose()
    return df


def get_all_metric_values(filenames):
    metric_name = None
    metric_values_and_timestamp = []
    for filename in filenames:
        # Open and read the JSON file
        with open(filename, 'r') as file:
            data = json.load(file)["data"]

        # Print the data
        metric_name = data["result"][0]["metric"]["__name__"]
        for i in range(len(data["result"])):
            vals = data["result"][i]["values"]
            metric_values_and_timestamp.extend(vals)
        # microservice_metric_values_and_timestamp = data["result"][0]["values"]
        # metric_values_and_timestamp.extend(microservice_metric_values_and_timestamp)

    # Sort based on timestamp
    metric_values_and_timestamp.sort(key=lambda x: x[0])
    metric_values_and_timestamp = [(float(t), float(m)) for t, m in metric_values_and_timestamp]

    # metric_values = [float(metric_value) for _, metric_value in metric_values_and_timestamp]
    return metric_name, metric_values_and_timestamp     # metric_values


def is_within_windows(timestamp, windows):
    return any(s <= timestamp <= e for s, e in windows)


if __name__ == '__main__':
    # all_anomaly_scores = AS.main(2, 64, 10, 9, False, 'DeepLog/model/Adam_batch_size=2048_epoch=300.pt')

    all_metrics_df = generate_service_metrics()

    # Manually sourced from potentialAnomalies files
    anomaly_time_windows = [("2022-07-08 13:49:15.159", "2022-07-08 13:49:21.266"),
                            ("2022-07-08 13:57:09.655", "2022-07-08 13:57:09.869"),
                            ("2022-07-08 13:59:04.954", "2022-07-08 13:59:30.768"),
                            ("2022-07-08 14:05:00.063", "2022-07-08 14:05:30.870"),
                            ("2022-07-08 14:48:54.161", "2022-07-08 14:49:07.758"),
                            ("2022-07-08 14:50:28.247", "2022-07-08 14:50:42.465"),
                            ("2022-07-08 15:29:46.760", "2022-07-08 15:29:46.861"),
                            ("2022-07-13 19:41:53.042", "2022-07-13 19:41:53.061"),
                            ("2022-07-13 19:45:23.861", "2022-07-13 19:45:23.864"),
                            ("2022-07-12 20:13:04.955", "2022-07-12 20:13:05.260")]
    int_anom_windows = []
    # convert all time windows into unix timestamps
    for window in anomaly_time_windows:
        s_dt = datetime.strptime(window[0], "%Y-%m-%d %H:%M:%S.%f")
        e_dt = datetime.strptime(window[1], "%Y-%m-%d %H:%M:%S.%f")
        start = int(s_dt.timestamp()) - 36000
        end = int(e_dt.timestamp()) + 36000
        print(f"start = {start}, end = {end}")
        # make windows +/- 8 hours of the actual start and end
        int_anom_windows.append((start, end))
    int_anom_windows = sorted(int_anom_windows, key=lambda x: x[0])

    epsilon_d_values = {}
    for metric in all_metrics_df.columns:
        values = all_metrics_df[metric].dropna()

        abnormal_values, normal_values = [], []
        for time, val in values:
            (abnormal_values if is_within_windows(time, int_anom_windows) else normal_values).append(val)
        epsilon_d_values[metric] = epsilon_diagnosis(normal_values, abnormal_values)

    # Normalize the epsilon values
    total_epsilon_values = sum(epsilon_d_values.values())
    for k in epsilon_d_values.keys():
        epsilon_d_values[k] /= total_epsilon_values

    print("Epsilon-Diagnosis Results:")
    for k in epsilon_d_values.keys():
        print(f"{k}: {epsilon_d_values[k]}")
