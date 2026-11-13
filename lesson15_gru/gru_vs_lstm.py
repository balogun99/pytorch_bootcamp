# import libraries
import torch
import torch.nn as nn

# configuration
INPUT_SIZE = 5
HIDDEN_SIZE = 16
NUM_LAYERS = 1
BATCH_SIZE = 2
SEQUENCE_LENGTH = 4

# initialize models
rnn = nn.RNN(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    batch_first=True
)

lstm = nn.LSTM(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    batch_first=True
)

gru = nn.GRU(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    batch_first=True
)

# count parameters
def count_parameters(model):

    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )

# display comparison
print("\n RNN vs LSTM vs GRU")
print("=============================")

print("RNN Parameters: ", count_parameters(rnn))
print("LSTM Parameters: ", count_parameters(lstm))
print("GRU Parameters: ", count_parameters(gru))

# test forward pass
sample_input = torch.randn(
    BATCH_SIZE,
    SEQUENCE_LENGTH,
    INPUT_SIZE
)

rnn_output, _ = rnn(sample_input)
lstm_output, (lstm_hidden, lstm_cell) = lstm(sample_input)
gru_output, gru_hidden = gru(sample_input)

print("\nOUTPUT SHAPES")

print("RNN Output: ", rnn_output.shape)
print("LSTM Output: ", lstm_output.shape)
print("GRU Output: ", gru_output.shape)

print("\nMODEL COMPARISON COMPLETED")