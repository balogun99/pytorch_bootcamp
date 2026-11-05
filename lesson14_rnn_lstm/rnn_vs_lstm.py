# import libraries
import torch
import torch.nn as nn

# configurations
INPUT_SIZE = 5
HIDDEN_SIZE = 16
NUM_LAYERS = 1
BATCH_SIZE = 2
SEQUENCE_LENGTH = 12

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
rnn_hidden = torch.zeros(
    NUM_LAYERS,
    BATCH_SIZE,
    HIDDEN_SIZE
)

# initialize lstm states
lstm_hidden = torch.zeros(
    NUM_LAYERS,
    BATCH_SIZE,
    HIDDEN_SIZE
)

lstm_cell = torch.zeros(
    NUM_LAYERS,
    BATCH_SIZE,
    HIDDEN_SIZE
)

# initialize the rnn
rnn_output, rnn_final_hidden = rnn(
    input_data,
    rnn_hidden
)

# initialize lstm
lstm_output, (lstm_final_hidden, lstm_final_cell) = lstm(
    input_data,
    (lstm_hidden, lstm_cell)
)

# count model parameters
def count_parameters(model):

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

rnn_parameters = count_parameters(rnn)
lstm_parameters = count_parameters(lstm)

# display the results
print("\n" + "=" * 50)
print("RNN vs LSTM")
print("=" * 50)

print("\nRNN RESULTS")

print("Output Shape: ", rnn_output.shape)

print(
    "Final Hidden State Shape: ",
    rnn_final_hidden.shape
)

print("Trainable Parameters: ", rnn_parameters)

print("\nLSTM RESULTS")

print("Final Hidden State Shape: ",
    lstm_final_hidden.shape
    )

print("Final Cell State Shape: ",
    lstm_final_cell.shape
    )

print("Trainable Parameters: ", lstm_parameters)

# compare output shape
print("\nOUTPUT COMPARISON")

print(
    "RNN and LSTM output shapes match: ",
    rnn_output.shape ==  lstm_output.shape
)

print(
    "Are their numerical outputs identical? ",
    torch.allclose(rnn_output, lstm_output) # this checks whether the two output tensors are numerically close.
)

# parameter comparison
print("\nPARAMETER COMPARISON")

print("RNN Parameters: ", rnn_parameters)
print("LSTM Parameters: ", lstm_parameters)

print(
    "Additional LSTM Parameters: ",
    lstm_parameters - rnn_parameters
)

print("\nComparison Completed Successfully!")