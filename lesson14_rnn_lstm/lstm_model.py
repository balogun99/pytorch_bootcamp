# import libraries
import torch
import torch.nn as nn

# from lesson14_rnn_lstm.rnn_fundamentals import SEQUENCE_LENGTH

# class for LSTMClassifier
class LSTMClassifier(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_classes
    ):

        super().__init__()

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # fully connected layer
        self.fc = nn.Linear(
            hidden_size,
            num_classes
        )

    def forward(self, x):

        # pass input through LSTM
        output, (hidden_state, cell_state) = self.lstm(x)

        # extract the last hidden state
        final_hidden = hidden_state[-1]

        # generate classification scores
        predictions = self.fc(final_hidden)

        return predictions

# test the model
if __name__ == "__main__":
    # define model parameters
    INPUT_SIZE = 5
    HIDDEN_SIZE = 10
    NUM_LAYERS = 1
    NUM_CLASSES = 2

    BATCH_SIZE = 2
    SEQUENCE_LENGTH = 4

    # create sample input
    sample_input = torch.randn(
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        INPUT_SIZE
    )

    # create model instance
    model = LSTMClassifier(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES
    )

    # forward pass
    predictions = model(sample_input)

    print("\nLSTM CLASSIFIER RESULTS")

    print("Input Shape: ", sample_input.shape)
    print("Predictions Shape: ", predictions.shape)

    print("\nPredictions: ")
    print(predictions)

    print("\nLSTM Classifier Executed Successfully!")