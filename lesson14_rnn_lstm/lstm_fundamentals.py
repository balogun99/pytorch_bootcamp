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

# create an LSTM model
lstm = nn.LSTM(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    batch_first=True
)

# initiate hidden state and cell state
# shape: (num_layers, batch_size, hidden_size)
hidden_state = torch.zeros(
    NUM_LAYERS,
    BATCH_SIZE,
    HIDDEN_SIZE
)

# initialize cell state
# same shape as hidden state
cell_state = torch.zeros(
    NUM_LAYERS,
    BATCH_SIZE,
    HIDDEN_SIZE
)

# pass data through the LSTM model
output, (final_hidden_state, final_cell_state) = lstm(
    input_data,
    (hidden_state, cell_state)
)

# display results
print("\nLSTM RESULTS")
print("Output Shape: ", output.shape)
print("Final Hidden State Shape: ", 
    final_hidden_state.shape
    )
print("Final Cell State Shape: ",
    final_cell_state.shape
    )

# display final states
print("\nFinal Hidden State:")
print(final_hidden_state)

print("\nFinal Cell State:")
print(final_cell_state)

print("\nLSTM execution completed successfully!")

# (pytorch_env) macbookpro@MACBOOKs-MBP pytorch_bootcamp % python -m lesson14_rnn_lstm.lstm
# _fundamentals

# INPUT DATA
# Input Shape:  torch.Size([2, 4, 5])

# LSTM RESULTS
# Output Shape:  torch.Size([2, 4, 10])
# Final Hidden State Shape:  torch.Size([1, 2, 10])
# Final Cell State Shape:  torch.Size([1, 2, 10])

# Final Hidden State:
# tensor([[[-0.0815,  0.1038, -0.0326, -0.1376, -0.0568, -0.0510,  0.0972,
#            0.1406, -0.1187, -0.0254],
#          [ 0.1442,  0.0022,  0.0212,  0.1541,  0.1125, -0.0293, -0.0027,
#           -0.0173, -0.0627, -0.0583]]], grad_fn=<StackBackward0>)

# Final Cell State:
# tensor([[[-0.1565,  0.2301, -0.1164, -0.2961, -0.1847, -0.1259,  0.1459,
#            0.2414, -0.3094, -0.0435],
#          [ 0.2267,  0.0040,  0.0592,  0.3990,  0.2953, -0.0510, -0.0042,
#           -0.0324, -0.1696, -0.1185]]], grad_fn=<StackBackward0>)

# LSTM execution completed successfully!
# (pytorch_env) macbookpro@MACBOOKs-MBP pytorch_bootcamp % 

# 14.7.6 — Understanding the LSTM output
# Let's break down the three outputs.

# 1. Output
# output



# Shape:
# [2, 4, 10]


# It contains the output representation for every time step in both sequences.

# 2. Final hidden state
# final_hidden_state



# Shape:
# [1, 2, 10]


# It contains the final hidden state for each sequence.

# 3. Final cell state
# final_cell_state



# Shape:
# [1, 2, 10]


# It contains the final cell state for each sequence.

# Here's an important distinction:
# - The hidden state is used to produce the LSTM's output.
# - The cell state provides an additional pathway for carrying information through the sequence.
# The cell state is not simply another prediction. It is part of the LSTM's internal memory.