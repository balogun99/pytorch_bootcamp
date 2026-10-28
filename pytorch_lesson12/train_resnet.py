
import os
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from models.resnet_model import (
    build_resnet18,
    fine_tune_last_block,
    set_feature_extraction_mode,
)

# configuration

LESSON_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(LESSON_DIR)

DATA_DIR = os.path.join(PROJECT_DIR, "data")
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "checkpoints")
LOG_DIR = os.path.join(LESSON_DIR, "runs", "resnet18_finetune")

BATCH_SIZE = 32
NUM_EPOCHS = 3
NUM_CLASSES = 10
TRAIN_SAMPLES = 10000
VAL_SAMPLES = 2000
IMAGE_SIZE = 64
RANDOM_SEED = 42

BACKBONE_LR = 0.0001
CLASSIFIER_LR = 0.001
WEIGHT_DECAY = 0.0001

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

torch.manual_seed(RANDOM_SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

# data transforms

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# load cifar-10

train_full = datasets.CIFAR10(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=train_transform,
)

val_full = datasets.CIFAR10(
    root=DATA_DIR,
    train=True,
    download=False,
    transform=eval_transform,
)

# reproduce the same training/validation split.
generator = torch.Generator().manual_seed(RANDOM_SEED)

indices = torch.randperm(
    len(train_full),
    generator=generator,
).tolist()

train_indices = indices[:TRAIN_SAMPLES]

val_indices = indices[
    TRAIN_SAMPLES:TRAIN_SAMPLES + VAL_SAMPLES
]

train_dataset = torch.utils.data.Subset(
    train_full,
    train_indices,
)

val_dataset = torch.utils.data.Subset(
    val_full,
    val_indices,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Image size:", IMAGE_SIZE, "x", IMAGE_SIZE)

# build model for fine tuning

model = build_resnet18(
    num_classes=NUM_CLASSES,
    pretrained=True,
)

model = fine_tune_last_block(model)
model = model.to(device)

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print("Trainable parameters:", trainable_parameters)
print("Total parameters:", total_parameters)

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    [
        {
            "params": model.layer4.parameters(),
            "lr": BACKBONE_LR,
        },
        {
            "params": model.fc.parameters(),
            "lr": CLASSIFIER_LR,
        },
    ],
    weight_decay=WEIGHT_DECAY,
)

writer = SummaryWriter(LOG_DIR)

# model training and validation

best_val_accuracy = 0.0

checkpoint_path = os.path.join(
    CHECKPOINT_DIR,
    "resnet18_finetuned_best.pth",
)

try:
    for epoch in range(NUM_EPOCHS):

        # train mode, while keeping BatchNorm statistics fixed.
        set_feature_extraction_mode(model)

        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            train_correct += (
                predictions == labels
            ).sum().item()

            train_total += labels.size(0)

        train_loss = running_loss / train_total
        train_accuracy = train_correct / train_total

        # model validation

        model.eval()

        val_loss_total = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss_total += (
                    loss.item() * images.size(0)
                )

                predictions = outputs.argmax(dim=1)

                val_correct += (
                    predictions == labels
                ).sum().item()

                val_total += labels.size(0)

        val_loss = val_loss_total / val_total
        val_accuracy = val_correct / val_total

        # tensorboard logging

        writer.add_scalar(
            "Loss/Train", train_loss, epoch + 1
        )
        writer.add_scalar(
            "Loss/Validation", val_loss, epoch + 1
        )
        writer.add_scalar(
            "Accuracy/Train", train_accuracy, epoch + 1
        )
        writer.add_scalar(
            "Accuracy/Validation", val_accuracy, epoch + 1
        )

        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Train Acc: {train_accuracy:.4f} "
            f"| Val Loss: {val_loss:.4f} "
            f"| Val Acc: {val_accuracy:.4f}"
        )

        # Save the best fine-tuned checkpoint separately.
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_accuracy,
                    "class_names": train_full.classes,
                    "image_size": IMAGE_SIZE,
                    "backbone_lr": BACKBONE_LR,
                    "classifier_lr": CLASSIFIER_LR,
                },
                checkpoint_path,
            )

            print("Fine-tuned model saved:", checkpoint_path)

finally:
    writer.close()

print("\nFine-tuning complete!")
print(f"Best validation accuracy: {best_val_accuracy:.4f}")
print("Checkpoint:", checkpoint_path)
