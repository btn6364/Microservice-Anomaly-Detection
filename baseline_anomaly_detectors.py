import numpy as np
import DeepLog.find_anomaly_score as AS
import pandas as pd
import json


def epsilon_diagnosis(normal_set, abnormal_set):
    # normal set and abnormal set are lists of a given metric measurement in float format
    ab_var = np.var(abnormal_set)
    norm_var = np.var(normal_set)
    if ab_var == 0 or norm_var == 0:
        return 0

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

    df = pd.DataFrame.from_dict(metric_map)
    return df.to_numpy(), df.columns


def get_all_metric_values(filenames):
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

    metric_values = [float(metric_value) for _, metric_value in metric_values_and_timestamp]
    return metric_name, metric_values


if __name__ == '__main__':
    all_anomaly_scores = AS.main(2, 64, 10, 9, False)

    all_metrics = generate_service_metrics()

    abnormal_threshold = .5
    for k in all_anomaly_scores.keys():
        serv_anomaly_score = all_anomaly_scores[k]
        if serv_anomaly_score > abnormal_threshold:
            pass
