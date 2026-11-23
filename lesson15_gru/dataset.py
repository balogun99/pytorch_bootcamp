# import libraries
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from lesson15_gru.data import generate_sequences

# custom PyTorch Dataset
class TimeSeriesDataset(Dataset):

    def __init__(self, features, labels): # This initializes our Dataset and stores the input sequences and their corresponding labels.
        self.features = features
        self.labels = labels

    def __len__(self): # This tells PyTorch how many samples are available.
        return len(self.features)

    def __getitem__(self, index): # This retrieves one sequence and its corresponding label.
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
        [train_size, test_size],
        generator=generator
    )

    # create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, test_loader

if __name__ == "__main__":

    train_loader, test_loader = create_dataloaders()

    print("DATA LOADER INFORMATION")
    print("=================================")

    print("Training Samples: ", len(train_loader.dataset))
    print("Testing Samples: ", len(test_loader.dataset))

    print("Training Batches: ", len(train_loader))
    print("Testing Batches: ", len(test_loader))

    # inspect one training epoch
    sample_sequences, sample_labels = next(iter(train_loader))

    print("\nSAMPLE BATCH INFORMATION")
    print("==================================")

    print("Sequence Batch Shape: ", sample_sequences.shape)
    print("Label Batch Shape: ", sample_labels.shape)

    print("Sequence Data Type: ", sample_sequences.dtype)
    print("Label Data Type: ", sample_labels.dtype)

    print("\nDATASET AND DATALOADER COMPLETED!")