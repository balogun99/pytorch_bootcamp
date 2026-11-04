# import libraries
import torch
import torch.nn as nn

# configurations
INPUT_SIZE = 5
HIDDEN_SIZE = 10
NUM_LAYERS = 1
BATCH_SIZE = 2
SEQUENCE_LENGTH = 6

# create sample input
torch.manual_seed(0)

input_data = torch.randn(
    BATCH_SIZE,
    SEQUENCE_LENGTH,
    INPUT_SIZE
)

print("\nINPUT DATA")
print("Input Shape: ", input_data.shape)

# create RNN model
rnn = nn.RNN(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    batch_first=True
)

# create LSTM model
lstm = nn.LSTM(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    batch_first=True
)

# initialize hidden state for RNN