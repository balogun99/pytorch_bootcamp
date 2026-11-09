# \[
# y = sin(0.05t)+0.5sin(0.15t)+\epsilon
# \]

# Where:
# - \(t\) represents time.
# - sin() creates a repeating pattern.
# - \(\epsilon\) represents small random noise.
# Why use this dataset? It gives us a predictable pattern that an LSTM can learn, without requiring a large external dataset.

# import libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np

# generate synthetic data
def generate_series(length=300, seed=42):

    rng = np.random.default_rng(seed)

    time = np.arange(length, dtype=np.float32)

    signal = (
        np.sin(0.05 * time)
        + 0.5 * np.sin(0.15 * time)
    )

    noise = rng.normal(
        loc=0.0,
        scale=0.05,
        size=length
    )

    values = signal + noise

    return values.astype(np.float32)


# create a sequence dataset for LSTM
class TimeSeriesDataset(Dataset):

    def __init__(
        self,
        values,
        target_indices,
        sequence_length
    ):

        self.values = torch.tensor(
            values,
            dtype=torch.float32
        )

        self.target_indices = target_indices
        self.sequence_length = sequence_length

    def __len__(self):

        return len(self.target_indices)
    
    def __getitem__(self, index):

        target_index = self.target_indices[index]

        # previous observations
        x = self.values[
            target_index - self.sequence_length:
            target_index
        ]

        # next value to predict
        y = self.values[target_index]

        # # add feature dimension
        # x = x.unsqueeze(-1)

        return x.unsqueeze(-1), y

# prepare training and testing data
def prepare_datasets(
    sequence_length=20,
    train_ratio=0.8
):

    values = generate_series(length=300)

    split_index = int(
        len(values) * train_ratio
    )

    # calcuate normalization values using
    # training observations only
    train_values = values[:split_index]

    mean = train_values.mean()
    std = train_values.std()

    normalized_values = (
        values - mean
    ) / std

    # target indices start after the first
    # complete input sequence
    all_indices = np.arange(
        sequence_length,
        len(values)
    )

    # chronological split by target index
    # train_indices = all_indices[
    #     all_indices < split_index
    # ]

    # test_indices = all_indices[
    #     all_indices < split_index
    # ]

    all_target_indices = np.arange(sequence_length, len(values))

    train_indices = all_target_indices[
    all_target_indices < split_index
    ]

    test_indices = all_target_indices[
    all_target_indices >= split_index
    ]

    # train_dataset = TimeSeriesDataset(
    #     normalized_values,
    #     train_indices,
    #     sequence_length
    # )

    # test_dataset = TimeSeriesDataset(
    #     normalized_values,
    #     test_indices,
    #     sequence_length
    # )

    train_dataset = TimeSeriesDataset(
    normalized_values,
    train_indices,
    sequence_length
    )

    test_dataset = TimeSeriesDataset(
    normalized_values,
    test_indices,
    sequence_length
    )

    return (
        train_dataset,
        test_dataset,
        mean,
        std,
        values
    )

# test data preparation
if __name__ == "__main__":

    train_dataset, test_dataset, mean, std, values = (
        prepare_datasets()
    )

    print("\nTIME SERIES DATASET")
    print("Total Observations: ", len(values))
    print("Training Samples: ", len(train_dataset))
    print("Testing Samples: ", len(test_dataset))
    print("Sequence Length: ", 20)

    sample_x, sample_y = train_dataset[0]

    print("\nSAMPLE DATA")

    print("Input Shape: ", sample_x.shape)
    print("Target Shape: ", sample_y.shape)
    print("\nData Preparation Completed!")