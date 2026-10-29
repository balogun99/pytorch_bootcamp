# import libraries
import os
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset
from lesson13_nlp_pytorch.data.sentiment_data import texts, labels
from lesson13_nlp_pytorch.data.text_dataset import (
    BATCH_SIZE,
    MAX_LENGTH,
    SentimentDataset,
    build_vocabulary,
    stratified_split
)
from lesson13_nlp_pytorch.models.sentiment_model import SentimentModel
from pytorch_lesson10.train import CHECKPOINT_PATH, LEARNING_RATE

# configuration
RANDOM_SEED = 42
EPOCHS = 10
LEARNING_RATE = 0.01
EMBEDDING_DIM = 32

CHECKPOINT_DIR = "lesson13_nlp_pytorch/checkpoints"
CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "sentiment_model.pth"
)

torch.manual_seed(RANDOM_SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else
    "cpu"
)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# prepare the dataset
train_indices, test_indices = stratified_split(
    texts,
    labels,
    test_ratio=0.2,
    seed=RANDOM_SEED,
)

training_texts = [texts[i] for i in train_indices]

# build the vocabulary using training text only
vocabulary = build_vocabulary(training_texts)

dataset = SentimentDataset(
    texts,
    labels,
    vocabulary,
    max_length=MAX_LENGTH,
)

train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

# load the dataloader
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

print("=" * 50)
print("SENTIMENT CLASSIFIER TRAINING")
print("=" * 50)
print("Device: ", device)
print("Total Examples: ", len(dataset))
print("Training Examples: ", len(train_dataset))
print("Testing Examples: ", len(test_dataset))
print("Vocabulary Size: ", len(vocabulary))
print("Batch Size: ", BATCH_SIZE)
print("Epochs: ", EPOCHS)
print("Learning Rate: ", LEARNING_RATE)

# create the model
model = SentimentModel(
    vocab_size=len(vocabulary),
    embedding_dim=EMBEDDING_DIM,
    padding_idx=0,
).to(device)

# Combines sigmoid and binary cross-entropy in a
# numerically stable way.
criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

print("\nModel created successfully")
print("Model Parameters: ", sum(
    parameter.numel()
    for parameter in model.parameters()
))

# train the model
best_train_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()

    total_loss = 0
    total_correct = 0
    total_examples = 0

    for input_ids, batch_labels in train_loader:
        input_ids = input_ids.to(device)
        batch_labels = batch_labels.to(device)

        # Forward pass: generate raw prediction scores.
        logits = model(input_ids)

        # Calculate the difference between predictions
        # and the correct labels.
        loss = criterion(logits, batch_labels)

        # clear gradients from the previous batch
        optimizer.zero_grad()

        # Backpropagation: calculate gradients.
        loss.backwards()

        # update model parameters
        optimizer.step()

        # accumulate metrics
        batch_size = batch_labels.size(0)

        total_loss += loss.item() * batch_size

        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).float()

        total_correct += (
            predictions == batch_labels
        ).sum().item()

        total_examples += batch_size

    epoch_loss = total_loss / total_examples
    epoch_accuracy = total_correct / total_examples

    # Save the checkpoint with the lowest training loss.
    # The held-out test set is not used to select it.
    if epoch_loss < best_train_loss:
        best_train_loss = epoch_loss

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "vocabulary": vocabulary,
                "max_length": MAX_LENGTH,
                "embedding_dim": EMBEDDING_DIM,
                "pad_token": "<PAD>",
                "unk_token": "<UNK>",
                "best_train_loss": best_train_loss,
                "epoch": epoch + 1,
            },
            CHECKPOINT_PATH,
        )

    # Print progress every five epochs and on the last.
    if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == EPOCHS - 1:
        print(
            f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
            f"| Loss: {epoch_loss:.4f} "
            f"| Accuracy: {epoch_accuracy:.2%}"
        )


# --------------------------------------------------
# 5. Load the saved checkpoint
# --------------------------------------------------

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("\nBest checkpoint saved at:")
print(CHECKPOINT_PATH)
print("Checkpoint epoch:", checkpoint["epoch"])
print("Best training loss:", f"{checkpoint['best_train_loss']:.4f}")


# --------------------------------------------------
# 6. Evaluate once on the held-out test set
# --------------------------------------------------

test_loss_total = 0.0
test_correct = 0
test_examples = 0

with torch.no_grad():
    for input_ids, batch_labels in test_loader:
        input_ids = input_ids.to(device)
        batch_labels = batch_labels.to(device)

        logits = model(input_ids)
        loss = criterion(logits, batch_labels)

        batch_size = batch_labels.size(0)

        test_loss_total += loss.item() * batch_size

        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).float()

        test_correct += (
            predictions == batch_labels
        ).sum().item()

        test_examples += batch_size

test_loss = test_loss_total / test_examples
test_accuracy = test_correct / test_examples

print("\n" + "=" * 50)
print("FINAL TEST RESULTS")
print("=" * 50)
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.2%}")
print(f"Correct predictions: {test_correct}/{test_examples}")
print("\nTraining completed successfully!")