# import libraries
import torch
import torch.nn as nn

# configuration
INPUT_SIZE = 5
HIDDEN_SIZE = 16
NUM_LAYERS = 1
BATCH_SIZE = 2
SEQUENCE_LENGTH = 4

# create GRU model
gru = nn.GRU(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    batch_first=True
)

# create sample input
sample_input = torch.randn(
    BATCH_SIZE,
    SEQUENCE_LENGTH,
    INPUT_SIZE
)

# initialize hidden state
initial_hidden = torch.zeros(
    NUM_LAYERS,
    BATCH_SIZE,
    HIDDEN_SIZE
)

# forward propagation
output, final_hidden = gru(
    sample_input,
    initial_hidden
)

# display results
print("\nGRU FUNDAMENTALS")
print("==========================")

print("Input Shape: ", sample_input.shape)
print("Initial Hidden Shape: ", initial_hidden.shape)

print("Output Shape: ", output.shape)
print("Final Hidden Shape: ", final_hidden.shape)

print(
    "GRU Parameters: ",
    sum(parameter.numel() for parameter in gru.parameters())
)

print("\nGRU Forward Pass Completed!")