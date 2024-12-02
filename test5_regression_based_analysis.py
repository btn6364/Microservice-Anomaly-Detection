from find_anomaly_score_modified import find_anomaly_scores
import numpy as np
import torch
import torch.nn as nn

# Define the device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def regression_based_analysis(log_file, metric_matrix, model_path, num_layers, hidden_size, window_size, num_candidates):
    """
    Perform regression-based analysis.

    Parameters:
        - log_file: Log file to process
        - metric_matrix: 2D numpy array of metric values
        - model_path: Path to the pre-trained model
        - num_layers: Number of LSTM layers
        - hidden_size: Hidden size of LSTM
        - window_size: Window size for sequences
        - num_candidates: Top-k predictions considered for anomaly detection

    Returns:
        - beta_vector: Coefficients from the regression analysis
    """
    # Step 1: Load the model
    class Model(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, num_classes):
            super(Model, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
            return out

    model = Model(1, hidden_size, num_layers, 28).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=False))
    model.eval()

    # Step 2: Call `find_anomaly_scores` to generate anomaly score vector
    anomaly_scores = find_anomaly_scores(log_file, model, window_size, num_candidates, return_mean=False)

    # Step 3: Perform regression analysis
    anomaly_scores = np.array(anomaly_scores)

    # Add these lines for debugging
    print(f"Metric matrix shape: {metric_matrix.shape}")
    print(f"Anomaly scores length: {len(anomaly_scores)}")

    # Ensure metric_matrix and anomaly_scores align
    if metric_matrix.shape[0] != len(anomaly_scores):
        raise ValueError("The number of rows in metric_matrix must equal the length of anomaly_scores.")

    # Perform regression using a numerically stable pseudo-inverse
    beta_vector = np.linalg.pinv(metric_matrix.T @ metric_matrix) @ metric_matrix.T @ anomaly_scores

    return beta_vector[1:]  # Exclude intercept (alpha) from results



# Testing Section
if __name__ == "__main__":
    # Define paths and parameters for the test
    # log_file = 'DeepLog/data/hdfs_logs_admin_basic'
    # model_path = 'DeepLog/model/Adam_batch_size=2048_epoch=300.pt'
    log_file = 'data/hdfs_logs_admin_basic.txt'
    model_path = 'model/Adam_batch_size=2048_epoch=300.pt'
    num_layers = 2
    hidden_size = 64
    window_size = 10
    num_candidates = 9

    def generate_metric_matrix(num_rows, num_columns=5):
        metrics = np.random.rand(num_rows, num_columns)  # Generate random metric values
        intercept = np.ones((num_rows, 1))  # Column of 1's for the intercept
        return np.hstack((intercept, metrics))


    class Model(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, num_classes):
            super(Model, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
            return out


    try:
        anomaly_scores = find_anomaly_scores(
            name=log_file,
            model=Model(1, hidden_size, num_layers, 28).to(device),
            window_size=window_size,
            num_candidates=num_candidates,
            return_mean=False
        )

        metric_matrix = generate_metric_matrix(len(anomaly_scores), num_columns=7)

        beta_vector = regression_based_analysis(
            log_file=log_file,
            metric_matrix=metric_matrix,
            model_path=model_path,
            num_layers=num_layers,
            hidden_size=hidden_size,
            window_size=window_size,
            num_candidates=num_candidates
        )

        print("Regression Analysis Coefficients (Beta Vector):")
        metric_names = ["node_cpu_seconds_total", "node_memory_MemAvailable_bytes", "node_memory_MemTotal_bytes",
                        "node_network_transmit_packets_total", "container_cpu_usage_seconds_total",
                        "container_memory_working_set_bytes", "json_container_network_transmit_packets_total"]
        for i in range(len(metric_names)):
            print(f"{metric_names[i]}: {beta_vector[i]}")
    except Exception as e:
        print(f"An error occurred during testing: {e}")
