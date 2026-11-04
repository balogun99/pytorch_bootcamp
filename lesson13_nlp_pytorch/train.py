
# lesson13_nlp_pytorch/train.py

import os
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from lesson13_nlp_pytorch.data.sentiment_data import texts, labels
from lesson13_nlp_pytorch.data.text_dataset import (
    BATCH_SIZE,
    MAX_LENGTH,
    SentimentDataset,
    build_vocabulary,
    stratified_split,
)
from lesson13_nlp_pytorch.models.sentiment_model import SentimentModel


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

RANDOM_SEED = 42

EPOCHS = 100
LEARNING_RATE = 0.001
EMBEDDING_DIM = 32
WEIGHT_DECAY = 0.0001

PATIENCE = 10
MIN_DELTA = 0.0001

CHECKPOINT_DIR = "lesson13_nlp_pytorch/checkpoints"
CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "sentiment_model.pth"
)

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# --------------------------------------------------
# 2. CREATE TRAIN / VALIDATION / TEST SPLITS
# --------------------------------------------------

# First, separate 20% of the data for final testing.
train_val_indices, test_indices = stratified_split(
    texts,
    labels,
    test_ratio=0.2,
    seed=RANDOM_SEED,
)


def stratified_train_val_split(
    indices,
    labels,
    validation_ratio=0.2,
    seed=42,
):
    """Split training indices into stratified train and validation sets."""

    rng = random.Random(seed)

    train_indices = []
    validation_indices = []

    for class_label in sorted(set(labels)):
        class_indices = [
            i for i in indices
            if labels[i] == class_label
        ]

        rng.shuffle(class_indices)

        validation_count = max(
            1,
            round(len(class_indices) * validation_ratio)
        )

        validation_indices.extend(
            class_indices[:validation_count]
        )

        train_indices.extend(
            class_indices[validation_count:]
        )

    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)

    return train_indices, validation_indices


train_indices, validation_indices = stratified_train_val_split(
    train_val_indices,
    labels,
    validation_ratio=0.2,
    seed=RANDOM_SEED,
)

# --------------------------------------------------
# CHECK CLASS DISTRIBUTION
# --------------------------------------------------

def show_class_distribution(name, indices):

    positive_count = sum(
        labels[i] == 1 for i in indices
    )

    negative_count = sum(
        labels[i] == 0 for i in indices
    )

    print(f"\n{name}")
    print("Total:", len(indices))
    print("Positive:", positive_count)
    print("Negative:", negative_count)


show_class_distribution("TRAINING SET", train_indices)

show_class_distribution(
    "VALIDATION SET",
    validation_indices
)

show_class_distribution("TEST SET", test_indices)


# --------------------------------------------------
# 3. BUILD VOCABULARY FROM TRAINING TEXT ONLY
# --------------------------------------------------

training_texts = [
    texts[i] for i in train_indices
]

vocabulary = build_vocabulary(training_texts)

dataset = SentimentDataset(
    texts,
    labels,
    vocabulary,
    max_length=MAX_LENGTH,
)

train_dataset = Subset(dataset, train_indices)
validation_dataset = Subset(dataset, validation_indices)
test_dataset = Subset(dataset, test_indices)


# --------------------------------------------------
# 4. CREATE DATALOADERS
# --------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# --------------------------------------------------
# 5. DISPLAY DATASET INFORMATION
# --------------------------------------------------

print("=" * 55)
print("SENTIMENT CLASSIFIER TRAINING")
print("=" * 55)

print("Device:", device)
print("Total Examples:", len(dataset))
print("Training Examples:", len(train_dataset))
print("Validation Examples:", len(validation_dataset))
print("Testing Examples:", len(test_dataset))
print("Vocabulary Size:", len(vocabulary))
print("Batch Size:", BATCH_SIZE)
print("Maximum Epochs:", EPOCHS)
print("Learning Rate:", LEARNING_RATE)
print("Early Stopping Patience:", PATIENCE)


# --------------------------------------------------
# 6. CREATE MODEL, LOSS FUNCTION AND OPTIMIZER
# --------------------------------------------------

model = SentimentModel(
    vocab_size=len(vocabulary),
    embedding_dim=EMBEDDING_DIM,
    padding_idx=0,
).to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

print("\nModel created successfully")

print(
    "Model Parameters:",
    sum(parameter.numel() for parameter in model.parameters())
)


# --------------------------------------------------
# 7. VALIDATION FUNCTION
# --------------------------------------------------

def evaluate_loader(model, loader, criterion, device):

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():

        for input_ids, batch_labels in loader:

            input_ids = input_ids.to(device)
            batch_labels = batch_labels.to(device)

            logits = model(input_ids)

            loss = criterion(logits, batch_labels)

            batch_size = batch_labels.size(0)

            total_loss += loss.item() * batch_size

            probabilities = torch.sigmoid(logits)

            predictions = (probabilities >= 0.5).float()

            total_correct += (
                predictions == batch_labels
            ).sum().item()

            total_examples += batch_size

    average_loss = total_loss / total_examples
    accuracy = total_correct / total_examples

    return average_loss, accuracy


# --------------------------------------------------
# 8. TRAINING WITH EARLY STOPPING
# --------------------------------------------------

best_validation_loss = float("inf")
epochs_without_improvement = 0

for epoch in range(EPOCHS):

    model.train()

    training_loss_total = 0.0
    training_correct = 0
    training_examples = 0

    for input_ids, batch_labels in train_loader:

        input_ids = input_ids.to(device)
        batch_labels = batch_labels.to(device)

        # Forward pass
        logits = model(input_ids)

        # Calculate loss
        loss = criterion(logits, batch_labels)

        # Clear previous gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update parameters
        optimizer.step()

        batch_size = batch_labels.size(0)

        training_loss_total += loss.item() * batch_size

        probabilities = torch.sigmoid(logits)

        predictions = (probabilities >= 0.5).float()

        training_correct += (
            predictions == batch_labels
        ).sum().item()

        training_examples += batch_size

    training_loss = (
        training_loss_total / training_examples
    )

    training_accuracy = (
        training_correct / training_examples
    )

    # Evaluate on validation data
    validation_loss, validation_accuracy = evaluate_loader(
        model,
        validation_loader,
        criterion,
        device,
    )

    # Check whether validation loss improved
    if validation_loss < best_validation_loss - MIN_DELTA:

        best_validation_loss = validation_loss

        epochs_without_improvement = 0

        # Save the best checkpoint
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "vocabulary": vocabulary,
                "max_length": MAX_LENGTH,
                "embedding_dim": EMBEDDING_DIM,
                "pad_token": "<PAD>",
                "unk_token": "<UNK>",
                "best_validation_loss": best_validation_loss,
                "epoch": epoch + 1,
                "train_indices": train_indices,
                "validation_indices": validation_indices,
                "test_indices": test_indices,
            },
            CHECKPOINT_PATH,
        )

        checkpoint_status = "BEST MODEL SAVED"

    else:

        epochs_without_improvement += 1

        checkpoint_status = (
            f"No improvement: {epochs_without_improvement}/{PATIENCE}"
        )

    # Display progress
    if (epoch + 1) % 5 == 0 or epoch == 0:

        print(
            f"Epoch [{epoch + 1:03d}/{EPOCHS}] "
            f"| Train Loss: {training_loss:.4f} "
            f"| Train Acc: {training_accuracy:.2%} "
            f"| Val Loss: {validation_loss:.4f} "
            f"| Val Acc: {validation_accuracy:.2%}"
        )

    # Early stopping
    if epochs_without_improvement >= PATIENCE:

        print(
            f"\nEarly stopping triggered at epoch {epoch + 1}."
        )

        break


# --------------------------------------------------
# 9. LOAD BEST VALIDATION CHECKPOINT
# --------------------------------------------------

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("\n" + "=" * 55)
print("BEST CHECKPOINT INFORMATION")
print("=" * 55)

print("Checkpoint:", CHECKPOINT_PATH)
print("Best Epoch:", checkpoint["epoch"])

print(
    "Best Validation Loss:",
    f"{checkpoint['best_validation_loss']:.4f}"
)


# --------------------------------------------------
# 10. FINAL TEST EVALUATION
# --------------------------------------------------

test_loss, test_accuracy = evaluate_loader(
    model,
    test_loader,
    criterion,
    device,
)

print("\n" + "=" * 55)
print("FINAL TEST RESULTS")
print("=" * 55)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2%}")
print(
    f"Correct Predictions: "
    f"{round(test_accuracy * len(test_dataset))}/{len(test_dataset)}"
)

print("\nTraining completed successfully!")


# python -m lesson13_nlp_pytorch.train.py
# python -m lesson13_nlp_pytorch.train