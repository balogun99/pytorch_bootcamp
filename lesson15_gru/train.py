# import libraries
import os
import torch
import torch.nn as nn
import torch.optim as optim
from lesson15_gru.dataset import create_dataloaders
from lesson15_gru.gru_model import GRUClassifier

# configuration
INPUT_SIZE = 1
HIDDEN_SIZE = 32
NUM_LAYERS = 1
NUM_CLASSES = 2

BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT_DIR = "lesson15_gru/checkpoints"
CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "gru_classifier.pth"
)

# load data
train_loader, test_loader = create_dataloaders(
    batch_size=BATCH_SIZE
)

# initialize the model
model = GRUClassifier(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    num_classes=NUM_CLASSES
).to(DEVICE)

# loss function and optimizer
criterion = nn.CrossEntropyLoss()

# loss = criterion(outputs, labels)

# CrossEntropyLoss measures how far the predictions are from the correct labels.

# A lower loss generally indicates better predictions.

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# training function
def train_model(model, train_loader, criterion, optimizer, device):

    model.train()

    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for sequences, labels in train_loader:

        # move data to device
        sequences = sequences.to(device)
        labels = labels.to(device)

        # clear gradients
        optimizer.zero_grad()

        # forward propagation
        outputs = model(sequences)

        # calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # PyTorch calculates gradients, 
        # which tell the optimizer how 
        # the model parameters should change to reduce the loss.

        # update model parameters
        optimizer.step()
        # Adam uses the calculated gradients to update the model's weights.

        # track loss
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size

        # calculate predictions
        predictions = torch.argmax(outputs, dim=1)

        correct_predictions += (
            predictions == labels
        ).sum().item()

        total_samples += batch_size

    # calculate average loss and accuracy
    epoch_loss = total_loss / total_samples

    epoch_accuracy = (
        correct_predictions / total_samples
    ) * 100

    return epoch_loss, epoch_accuracy

# training loop
def main():

    print("\nGRU CLASSIFIER")
    print("======================")

    print("Device: ", DEVICE)
    print("Training Sample: ", len(train_loader.dataset))
    print("Testing Samples: ", len(test_loader.dataset))
    print("Epochs: ", NUM_EPOCHS)
    print("Learning Rate: ", LEARNING_RATE)

    print("\nSTARTING TRAIING...\n")

    for epoch in range(NUM_EPOCHS):

        train_loss, train_accuracy = train_model(
            model,
            train_loader,
            criterion,
            optimizer,
            DEVICE
        )

        print(
            f"Epoch [{epoch + 1:02d} / {NUM_EPOCHS}] "
            f"Loss: {train_loss:.4f} "
            f"Accuracy: {train_accuracy:.2f}%"
        )

        # save model checkpoint
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

        checkpoint = {
            "epoch": NUM_EPOCHS,
            "model_state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": train_loss,
            "accuracy": train_accuracy,
            "input_size": INPUT_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "num_classes": NUM_CLASSES
        }

        torch.save(checkpoint, CHECKPOINT_PATH)
        #This saves the trained model parameters and 
        # optimizer state so we can reload the model later.

        print("\nMODEL CHECKPOINT SAVED")
        print("=============================")
        print("Checkpoint: ", CHECKPOINT_PATH)

        print("\nGRU TRAINING COMPLETED")

if __name__ == "__main__":
    main()