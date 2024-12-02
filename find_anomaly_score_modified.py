import torch
import torch.nn as nn
import statistics

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Existing `generate` function
def generate(name, window_size):
    hdfs = set()
    with open(name, 'r') as f:
        for ln in f.readlines():
            ln = list(map(lambda n: n - 1, map(int, ln.strip().split())))
            ln = ln + [-1] * (window_size + 1 - len(ln))
            hdfs.add(tuple(ln))
    print('Number of sessions({}): {}'.format(name, len(hdfs)))
    return hdfs


# Existing `Model` class
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# New modular function: find_anomaly_scores
def find_anomaly_scores(name, model, window_size, num_candidates, return_mean=False):
    """
    Generates anomaly scores for a given log file.

    Parameters:
        - name: The log file name to process
        - model: Pre-trained LSTM model
        - window_size: The size of the window for sequences
        - num_candidates: Top-k predictions considered for anomaly detection
        - return_mean: If True, return the mean anomaly score; else, return all scores

    Returns:
        - List of anomaly scores or a single mean score
    """
    test_abnormal_loader = generate(name, window_size)
    anomaly_scores = []

    # Compute anomaly scores
    with torch.no_grad():
        for line in test_abnormal_loader:
            num_sequences = 0
            num_abnormal_sequences = 0
            for i in range(len(line) - window_size):
                num_sequences += 1

                seq = line[i:i + window_size]
                label = line[i + window_size]

                seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, 1).to(device)
                label = torch.tensor(label).view(-1).to(device)

                output = model(seq)
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                if label not in predicted:
                    num_abnormal_sequences += 1

            anomaly_score = round(num_abnormal_sequences / num_sequences, 2)
            anomaly_scores.append(anomaly_score)

    if return_mean:
        return round(statistics.mean(anomaly_scores), 3)
    return anomaly_scores


# Updated `main` function to use the modular function
def main(num_layers, hidden_size, window_size, num_candidates, return_mean):
    # Hyperparameters
    num_classes = 28
    input_size = 1
    model_path = 'model/Adam_batch_size=2048_epoch=300.pt'

    model = Model(input_size, hidden_size, num_layers, num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print('model_path: {}'.format(model_path))

    # Generate the anomaly scores for 3 microservices using the new modular function
    data_sources = [
        "hdfs_logs_admin_basic",
        "hdfs_logs_auth",
        "hdfs_logs_order"
    ]

    anomaly_scores = {}
    for data_source in data_sources:
        scores = find_anomaly_scores(data_source, model, window_size, num_candidates, return_mean)
        anomaly_scores[data_source] = scores
        print(f"Anomaly Scores for {data_source}: {scores}")

    return anomaly_scores


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_layers', default=2, type=int)
    parser.add_argument('-hidden_size', default=64, type=int)
    parser.add_argument('-window_size', default=10, type=int)
    parser.add_argument('-num_candidates', default=9, type=int)
    parser.add_argument('-return_mean', default=True, type=bool)
    args = parser.parse_args()

    main(
        args.num_layers,
        args.hidden_size,
        args.window_size,
        args.num_candidates,
        args.return_mean
    )
