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
    return


if __name__ == '__main__':
    pass
