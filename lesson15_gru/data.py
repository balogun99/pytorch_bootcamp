# import libraries
import torch
import numpy as np

def generate_sequences(
    n_samples=1000,
    sequence_length=30,
    noise_std=0.15,
    seed=42
):

    """
    Generate synthetic time-series sequences for binary classification.

    Class 0: Low-frequency sine waves.
    Class 1: High-frequency sine waves.
    """

    # reproducibility
    rng = np.random.default_rng(seed)

    X = []
    y = []

    time_steps = np.linspace(0, 1, sequence_length)

    for i in range(n_samples):

        # alternate between the two classes
        label = i % 2

        # random amplitude and phase
        amplitude = rng.uniform(0.8, 1.2)
        phase = rng.uniform(0, 2 * np.pi)

        if label == 0:
            frequency = 2

        else:
            frequency = 5

        # generate the sine value
        signal = amplitude * np.sin(
            2 * np.pi * frequency * time_steps + phase
        )

        # add random noise
        noise = rng.normal(
            0,
            noise_std,
            sequence_length
        )

        signal = signal + noise

        X.append(signal)
        y.append(label)

    # convert the NumPy arrays
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    # shuffle the dataset
    indices = rng.permutation(n_samples)

    X = X[indices]
    y = y[indices]

    # add feature dimension
    X = X.reshape(n_samples, sequence_length, 1)

    # convert to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    return X, y

# launch the model
if __name__ == "__main__":
    X, y = generate_sequences()

    print("DATASET INFORMATION")
    print("============================")

    print("Features Shape: ", X.shape)
    print("Labels Shape: ", y.shape)

    print("Data Type: ", X.dtype)
    print("Label TYpe: ", y.dtype)

    print("Class 0 Samples: ", (y == 0).sum().item())
    print("Class 1 Samples: ", (y == 1).sum().item())

    print("\nDATA GENERATION COMPLETED!.")