# import libraries
import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.cnn import CIFAR10CNN

# configure device
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# dataset transform
transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(
        [0.4914, 0.4822, 0.4465],
        [0.2470, 0.2435, 0.2616]
    )
])

# load dataset
# evaluate the model using test data
test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# create Dataloader
test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    num_workers=0,
    shuffle=False
)

# load the CIFAR10 saved model
model = CIFAR10CNN(
    num_classes=10
)

model.load_state_dict(
    torch.load(
        "pytorch_lesson10/checkpoints/best_model.pth",
        map_location=device,
        weights_only=True
    )
)

model = model.to(device)
model.eval()

# model loss
criterion = nn.CrossEntropyLoss()

# model evaluation
correct = 0
total = 0
test_loss = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        test_loss += loss.item()

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        average_loss = (
            test_loss / len(test_loader)
        )

        accuracy = correct / total

        print(
            f"Test Loss: {average_loss:.4f}"
        )

        print(
            f"Test Accuracy: "
            f"{accuracy * 100:.2f}%"
        )