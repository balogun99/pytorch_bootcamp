# import libraries
import torch
import torch.nn as nn

class GRUClassifier(nn.Module):

    def __init__(
        self,
        input_size=5,
        hidden_size=32,
        num_layers=1,
        num_classes=2,
        dropout=0.0
    ):

        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # GRU layer
        self.gru = nn.GRU( # This is the GRU initialization, and it creates the recurrent layer
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # classification layer
        self.fc = nn.Linear(
            hidden_size,
            num_classes
        )

    def forward(self, x):

        # process sequence
        output, hidden_state = self.gru(x)

        # extract final hidden state representation from the last GRU layer
        final_hidden = hidden_state[-1]

        # generate class scores (logits)
        # the fully connected layer converts the 32 hidden features into two class scores
        logits = self.fc(final_hidden)

        return logits

# test GRU classifier
if __name__ == "__main__":

    model = GRUClassifier(
        input_size=5,
        hidden_size=32,
        num_layers=1,
        num_classes=2
    )

    # 8 sequences, 10 time steps, 5 features
    sample_input = torch.randn(8, 10, 5)

    predictions = model(sample_input)

    predicted_classes = torch.argmax(
        predictions,
        dim=1
    )

    print("\nGRU CLASSIFIER")

    print("Input Shape: ", sample_input.shape)
    print("Output Shape: ", predictions.shape)
    print("Predicted Classes: ", predicted_classes)

    print(
        "Model Parameters: ",
        sum(p.numel() for p in model.parameters())
    )

    print("\nGRU Classification Forward Pass Completed")