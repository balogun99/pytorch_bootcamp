# import libraries
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from pathlib import Path

from lesson14_rnn_lstm.data import prepare_datasets
from lesson14_rnn_lstm.model import LSTMRegressor

# configuration
SEQUENCE_LENGTH = 20
HIDDEN_SIZE = 32

EPOCHS = 50
BATCH_SIZE = 16

LEARNING_RATE = 0.001

DEVICE = torch.device("cpu")

CHECKPOINT_PATH = Path(
    "lesson14_rnn_lstm/checkpoints/lstm_time_series.pth"
)

# reproducibility
torch.manual_seed(42)

# load datasets
train_dataset, test_dataset, mean, std, original_values = (
    prepare_datasets(
        sequence_length=SEQUENCE_LENGTH
    )
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nTEST DATASET DIAGNOSTICS")
print("Test dataset length:", len(test_dataset))

for batch_number, (inputs, targets) in enumerate(test_loader, start=1):

    print(
        f"Batch {batch_number}: "
        f"Inputs={inputs.shape}, "
        f"Targets={targets.shape}"
    )

# print dataset results
print("\nDATA LOADERS")
print("Training Batches: ", len(train_loader))
print("Testing Batches: ", len(test_loader))

# initialize LSTM Regressor
model = LSTMRegressor(
    input_size=1,
    hidden_size=HIDDEN_SIZE,
).to(DEVICE)

# Loss Function and Optimizer
criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# training loop
print("\nSTARTING LSTM TRAINING")

for epoch in range(EPOCHS):

    model.train()

    total_train_loss = 0.0

    for inputs, targets in train_loader:

        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)

        # clear previous gradients
        optimizer.zero_grad()

        # forward propagation
        predictions = model(inputs)

        # calculate loss
        loss = criterion(predictions, targets)

        # backpropagation
        loss.backward()

        # update model parameters
        optimizer.step()

        total_train_loss += loss.item() * inputs.size(0)

    average_train_loss = (
        total_train_loss / len(train_dataset)
    )

    if (epoch + 1) % 5 == 0:

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Training Loss: {average_train_loss:.6f}"
        )

# model evaluation
model.eval()

all_predictions = []
all_targets = []

with torch.no_grad():

    for inputs, targets in test_loader:

        inputs = inputs.to(DEVICE)

        batch_predictions = model(inputs)

        all_predictions.append(batch_predictions.cpu())
        all_targets.append(targets.cpu())

# combine batches
predictions = torch.cat(all_predictions)
targets = torch.cat(all_targets)

# denormalize predictions
mean = float(mean)
std = float(std)

predictions_original = predictions * std + mean
targets_original = targets * std + mean

# calculate evaluation metrics
mae = torch.mean(
    torch.abs(predictions_original - targets_original)
)

rmse = torch.sqrt(
    torch.mean(
        (predictions_original - targets_original) ** 2
    )
)

# print the evaluation metrics results
print("\nMODEL EVALUATION")
print("-------------------------------------")
print(f"MAE: {mae.item():.6f}")
print(f"RMSE: {rmse.item():.6f}")

# save model checkpoint
CHECKPOINT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

checkpoint = {
    "model_state_dict": model.state_dict(),
    "mean": mean,
    "std": std,
    "sequence_length": SEQUENCE_LENGTH,
    "hidden_size": HIDDEN_SIZE,
    "num_layers": 1,
    "epochs": EPOCHS
}

torch.save(
    checkpoint,
    CHECKPOINT_PATH
)

# print save status
print("\nMODEL SAVED SUCCESSFULLY")
print("Checkpoint: ", CHECKPOINT_PATH)

print("\nTRAINING COMPLETED")

# MAE = 0.055931: On average, the model's predictions differ from the actual values by approximately 0.056 units 

# in the original time-series scale.

# RMSE = 0.076627: This measures prediction error while penalizing larger errors more heavily.

# Training loss = 0.006108: The model's final average training MSE in normalized units.