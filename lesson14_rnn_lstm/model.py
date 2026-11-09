# import libraries
import torch
import torch.nn as nn

# LSTMRegressor
class LSTMRegressor(nn.Module):

    def __init__(
        self,
        input_size=1,
        hidden_size=32,
        num_layers=1
    ):

        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # fully connected output layer
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):

        # pass input through LSTM
        output, (hidden_state, cell_state) = self.lstm(x)

        # get the final hidden state
        final_hidden = hidden_state[-1]

        # predict the next numerical value
        prediction = self.fc(final_hidden)

        return prediction.squeeze(-1)

# run the program
if __name__ == "__main__":

    model = LSTMRegressor()

    sample_input = torch.randn(4, 20, 1)

    predictions = model(sample_input)

    # print the results
    print("LSTM REGRESSION MODEL")
    print("Input Shape: ", sample_input.shape)
    print("Prediction Shape: ", predictions.shape)
    print("Model Parameters: ", sum(p.numel() for p in model.parameters()))

