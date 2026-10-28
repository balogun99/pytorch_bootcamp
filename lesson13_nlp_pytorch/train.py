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

