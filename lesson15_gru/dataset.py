# import libraries
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from lesson15_gru.data import generate_sequences

# custom PyTorch Dataset
class TimeSeriesDataset(Dataset):

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        sequence = self.features[index]
        label = self.labels[index]

        return sequence, label

def create_dataloaders(
    batch_size=32,
    train_ratio=0.8
):

    # generate dataset
    X, y = generate_sequences()

    # create custom dataset
    dataset = TimeSeriesDataset(X, y)

    # calculate training and testing sizes
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size

    # split dataset reproducibility
    generator = torch.Generator().manual_seed(42)

    train_dataset, test_dataset = random_split(
        dataset,
        [train_size - test_size],
        generator=generator
    )

    # create DataLoaders