# import libraries
import torch
import numpy as np

from pathlib import Path
from lesson14_rnn_lstm.model import LSTMRegressor
from lesson14_rnn_lstm.data import generate_series

# configuration
DEVICE = torch.device("cpu")

CHECKPOINT_PATH = Path(
    "lesson14_rnn_lstm/checkpoints/lstm_time_series.pth"
)

# load checkpoint
checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=True
)

sequence_length = checkpoint["sequence_length"]
hidden_size = checkpoint["hidden_size"]

mean = checkpoint["mean"]
std = checkpoint["std"]

# rebuild model
model = LSTMRegressor(
    input_size=1,
    hidden_size=hidden_size,
).to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

# generate new observations
values = generate_series(
    length=301,
    seed=42
)

# use the latest 20 observations
latest_values = values[-sequence_length-1:-1]

# normalize input
normalized_values = (
    latest_values / mean
) / std

input_tensor = torch.tensor(
    normalized_values,
    dtype=torch.float32
).unsqueeze(0).unsqueeze(-1)

# make predictions
with torch.no_grad():

    prediction = model(input_tensor)

# de-normalize prediction
predicted_value = (
    prediction.item() * std
) + mean

actual_value = values[-1]

# display results
print("\nLSTM TIME SERIES PREDICTION")
print("==============================")

print("Input Shape :", input_tensor.shape)
print(f"Predicted Next Value: {predicted_value:.6f}")

print(f"Actual Next Value: {actual_value:.6f}")

print(
    f"Absolute Error:      "
    f"{abs(predicted_value - actual_value):.6f}"
)

print("\nPrediction Completed!")