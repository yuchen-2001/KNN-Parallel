import numpy as np
import pandas as pd
import os

# Define dataset sizes
dataset_configs = [
    {"name": "small", "NTRAIN": 135, "NTEST": 15},
    {"name": "medium", "NTRAIN": 1000, "NTEST": 100},
    {"name": "large", "NTRAIN": 10000, "NTEST": 1000}
]

# dataset_configs = [
#     {"name": "small", "NTRAIN": 135, "NTEST": 15},
#     {"name": "medium", "NTRAIN": 10000, "NTEST": 1000},
#     {"name": "large", "NTRAIN": 1000000, "NTEST": 10000}
# ]

# Shared settings
NFEATURES = 4
NCLASSES = 3

# Base folder for datasets
base_dir = "./datasets"
base_dir = "./datasets_cuda"
os.makedirs(base_dir, exist_ok=True)

if __name__ == "__main__":
    # Generate and save datasets
    for config in dataset_configs:
        name = config["name"]
        NTRAIN = config["NTRAIN"]
        NTEST = config["NTEST"]

        dataset_path = os.path.join(base_dir, name)
        os.makedirs(dataset_path, exist_ok=True)

        X_train = np.random.rand(NTRAIN, NFEATURES)
        y_train = np.random.randint(0, NCLASSES, size=(NTRAIN,))
        X_test = np.random.rand(NTEST, NFEATURES)
        y_test = np.random.randint(0, NCLASSES, size=(NTEST,))

        # Save without headers or index
        pd.DataFrame(X_train).to_csv(f"{dataset_path}/X_train.csv", index=False, header=False)
        pd.DataFrame(y_train).to_csv(f"{dataset_path}/y_train.csv", index=False, header=False)
        pd.DataFrame(X_test).to_csv(f"{dataset_path}/X_test.csv", index=False, header=False)
        pd.DataFrame(y_test).to_csv(f"{dataset_path}/y_test.csv", index=False, header=False)

    base_dir
