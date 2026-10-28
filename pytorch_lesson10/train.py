# import libraries

import os
import torch
import torch.nn as nn
import json

from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from models.cnn import CIFAR10CNN
from engine.engine import (
    train_one_epoch,
    validate,
    save_checkpoint,
    load_checkpoint
)

# tensorboard writer configuration constants
# TensorBoard experiment name
EXPERIMENT_NAME = "pytorch_lesson10/cifar10_baseline_lr001_bs64"

writer = SummaryWriter(
    log_dir=f"pytorch_lesson10/runs/{EXPERIMENT_NAME}"
)

# configure the device
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Device in use: {device}")

# configure hyperparameters
BATCH_SIZE = 64

NUM_EPOCHS = 2

LEARNING_RATE = 0.001

WEIGHT_DECAY = 1e-4

PATIENCE = 5

# CHECKPOINT_PATH = (
#     "pytorch_lesson10/checkpoints/checkpoint.pth"
# )

# BEST_MODEL_PATH = (
#     "pytorch_lesson10/checkpoints/best_model.pth"
# )

CHECKPOINT_PATH = (
    "pytorch_lesson10/checkpoints/baseline_checkpoint.pth"
)

BEST_MODEL_PATH = (
    "pytorch_lesson10/checkpoints/baseline_best_model.pth"
)

# CIFAR-10 Dataset Normalization Parameters

CIFAR10_MEAN = [
    0.4914,
    0.4822,
    0.4465
]

CIFAR10_STD = [
    0.2470,
    0.2435,
    0.2616
]

# configure training transform
train_transform = transforms.Compose([

    transforms.RandomCrop(
        32,
        padding=4
    ),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(
        10
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        CIFAR10_MEAN,
        CIFAR10_STD
    )
])

# configure validation transform
test_transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(
        CIFAR10_MEAN,
        CIFAR10_STD
    )
])

# load CIFAR-10 dataset
train_dataset_full = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=train_transform
)

# for validation transform
val_dataset_full = datasets.CIFAR10(
    root="./data",
    train=True,
    download=False,
    transform=test_transform
)

# # create train and validation datasets
# train_size = int(
#     0.9 * len(train_dataset_full)
# )

# val_size = (
#     len(train_dataset_full)
#     - train_size
# )

# # generator
# generator = torch.Generator().manual_seed(42)

# # split the dataset into train and validation sets
# train_indices, val_indices = random_split(
#     range(len(train_dataset_full)),
#     [train_size, val_size],
#     generator=generator
# )

num_samples = len(train_dataset_full)

train_size = int(
    0.9 * num_samples
)

val_size = (
    num_samples - train_size
)

# generator for reproducibility
generator = torch.Generator().manual_seed(42)

indices = torch.randperm(
    num_samples,
    generator=generator
).tolist()

train_indices = indices[:train_size]

val_indices = indices[train_size:]

# create train and validation datasets
train_dataset = torch.utils.data.Subset(
    train_dataset_full,
    train_indices
)

val_dataset = torch.utils.data.Subset(
    val_dataset_full,
    val_indices
)

# create dataloaders for train and validation datasets
# train dataloader
train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

# validation dataloader
val_dataloader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0 # using 0 for OS Compatibility
)

# create the model
model = CIFAR10CNN(
    num_classes=10
)

model = model.to(device)

# loss function
criterion = nn.CrossEntropyLoss(
    label_smoothing=0.1
)

# optimizer using AdamW
optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

# learning rate scheduler
scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3
)

# training history
history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": [],
    "learning_rate": []
}

start_epoch = 0

best_val_loss = float("inf")

if os.path.exists(CHECKPOINT_PATH):

    print(
        f"Checkpoint found."
        "Resuming training loading."
    )

    start_epoch, best_val_loss = load_checkpoint(
        CHECKPOINT_PATH,
        model,
        optimizer,
        scheduler,
        device
    )

    print(
        f"Resuming training from epoch"
        f"{start_epoch + 1}."
    )

else:

    print(
        "No checkpoint found."
        "Starting Fresh retraining."
    )

# early-stopping variables
epochs_without_improvement = 0

# main training loop
for epoch in range(
    start_epoch,
    NUM_EPOCHS
):

    print(
        f"\nEpoch "
        f"{epoch + 1}/{NUM_EPOCHS}"
    )

    train_loss, train_acc = train_one_epoch(
        model=model,
        dataloader=train_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        max_grad_norm=1.0
    )

    val_loss, val_acc = validate(
        model=model,
        dataloader=val_dataloader,
        criterion=criterion,
        device=device
    )

    # Record metrics for the current epoch
    writer.add_scalar("Loss/Train", train_loss, epoch + 1)
    writer.add_scalar("Loss/Validation", val_loss, epoch + 1)

    writer.add_scalar("Accuracy/Train", train_acc, epoch + 1)
    writer.add_scalar("Accuracy/Validation", val_acc, epoch + 1)

    writer.add_scalar(
        "Learning_Rate",
        optimizer.param_groups[0]["lr"],
        epoch + 1
    )

    # update learning rate scheduler
    scheduler.step(val_loss)

    # get current learning rate
    current_lr = optimizer.param_groups[0]["lr"]

    # save training history
    history["train_loss"].append(
        train_loss
    )

    history['train_acc'].append(
        train_acc
    )

    history["val_loss"].append(
        val_loss
    )

    history["val_acc"].append(
        val_acc
    )

    history["learning_rate"].append(
        current_lr
    )

# print training and validation metrics
    print(
        f"Train Loss: {train_loss:.4f} "
    )

    print(
        f"Train Accuracy: "
        f"{train_acc:.2f}%"
    )

    print(
        f"Validation Loss: {val_loss:.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{val_acc:.2f}%"
    )

    print(
        f"Learning Rate: "
        f"{current_lr:.6f}"
    )

# save best model if validation loss improves
    if val_loss < best_val_loss:

        best_val_loss = val_loss

        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            BEST_MODEL_PATH
        )

        print(
            "Best model saved."
        )

    else:

        epochs_without_improvement += 1

# save lastest checkpoint
    save_checkpoint(
    path=CHECKPOINT_PATH,
    epoch=epoch,
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    best_val_loss=best_val_loss
)

    print(
        "✓ Checkpoint saved."
)

# Early stopping check
    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print(
            "Early stopping triggered."
        )

        break

Path("pytorch_lesson10/plots").mkdir(parents=True, exist_ok=True)

# pytorch_lesson10
# Save training history
with open("pytorch_lesson10/plots/history.json", "w") as f:
    json.dump(history, f, indent=4)


    print(
        "\nTraining history saved."
    )

writer.close()
print("TensorBoard logs saved.")

# to run tensorboard
# tensorboard --logdir=pytorch_lesson10/runs --port=6006