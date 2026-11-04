# import libraries
import torch
import torch.nn as nn

# configuration
INPUT_SIZE = 5
HIDDEN_SIZE = 10
NUM_LAYERS = 1
SEQUENCE_LENGTH = 4
BATCH_SIZE = 2

# create sample sequential data
# shape: (batch_size, sequence_length, input_size)

input_data = torch.randn(
    BATCH_SIZE,
    SEQUENCE_LENGTH,
    INPUT_SIZE
)

print("\nINPUT DATA")
print("Input Shape: ", input_data.shape)

# create an RNN model - traditional RNN
rnn = nn.RNN(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    batch_first=True
)

# initiate hidden state
# shape: (num_layers, batch_size, hidden_size)
hidden_state = torch.zeros(
    NUM_LAYERS,
    BATCH_SIZE,
    HIDDEN_SIZE
)

# pass data through the RNN model
output, final_hidden_state = rnn(
    input_data,
    hidden_state
)

# display results
print("\nOUTPUT RESULTS")
print("Output Shape: ", output.shape)
print(
    "Final Hidden State Shape: ",
    final_hidden_state.shape
)
print("\nFinal Hidden State: \n", final_hidden_state)
print(final_hidden_state)
print("\nRNN execution completed successfully!")

# Input shape: [2, 4, 5]
# Dimension	Value	Meaning
# First	2	Two sequences in the batch
# Second	4	Four time steps per sequence
# Third	5	Five features at each time step

# Output shape: [2, 4, 10]

# The RNN produces a hidden representation of size 10 for every time step in each sequence.

# Final hidden state shape: [1, 2, 10]
# - 1: One RNN layer.
# - 2: Two sequences.
# - 10: Ten hidden-state values per sequence.
# Notice that the output contains a hidden representation for every time step, whereas the final hidden state contains the last hidden representation for each sequence.