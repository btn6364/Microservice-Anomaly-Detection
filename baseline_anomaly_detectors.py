import numpy as np


def epsilon_diagnosis(normal_set, abnormal_set):
    ab_var = np.var(abnormal_set)
    norm_var = np.var(normal_set)
    if ab_var == 0 or norm_var == 0:
        return 0

    covariance = np.cov(normal_set, abnormal_set, ddof=1)[0, 1]
    result = (covariance ** 2) / np.sqrt(ab_var * norm_var)
    return result


def regression_based_analysis():
    return


def correlating_with_time_series(normal_set, abnormal_set):
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
    item_class = item[0]
    # Calculate 1D euclidean distances
    distances = [(cls, abs(value - item)) for cls, value in combined_set]
    sorted_distances = sorted(distances, key=lambda x: x[1])

    class_str, distance = sorted_distances[r]  # would be r - 1 but item is in the set as well
    if class_str == item_class:
        return 1
    else:
        return 0


if __name__ == '__main__':
    pass
