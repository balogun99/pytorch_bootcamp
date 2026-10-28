# import libr
import os
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.resnet_model import build_resnet18


# configure the path
LESSON_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(LESSON_DIR)

CHECKPOINT_PATH = os.path.join(
    PROJECT_DIR,
    "checkpoints",
    "resnet18_best.pth",
)

DATA_DIR = os.path.join(PROJECT_DIR, "data")
OUTPUT_DIR = os.path.join(LESSON_DIR, "plots")

os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_SIZE = 64
BATCH_SIZE = 64
MAX_IMAGES = 20

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# load test dataset
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

test_dataset = datasets.CIFAR10(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=transform,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

class_names = test_dataset.classes

# load the model
if not os.path.isfile(CHECKPOINT_PATH):
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT_PATH}"
    )

model = build_resnet18(
    num_classes=len(class_names),
    pretrained=False,
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

# collect incorrect predictions
wrong_images = []
wrong_labels = []
wrong_predictions = []
wrong_confidences = []
wrong_top3 = []

with torch.no_grad():
    for images, labels in test_loader:
        images_device = images.to(device)

        logits = model(images_device)
        probabilities = torch.softmax(logits, dim=1)

        confidences, predictions = probabilities.max(dim=1)

        top_probs, top_indices = torch.topk(
        probabilities, k=3, dim=1
)

        wrong_mask = predictions.cpu() != labels

        for index in torch.where(wrong_mask)[0].tolist():
            wrong_images.append(images[index].clone())
            wrong_labels.append(labels[index].item())
            wrong_predictions.append(predictions[index].item())
            wrong_confidences.append(confidences[index].item())

            wrong_top3.append([
            (
            class_names[top_indices[index, rank].item()],
            top_probs[index, rank].item() * 100,
            )
            for rank in range(3)
            ])

            if len(wrong_images) >= MAX_IMAGES:

                break

        if len(wrong_images) >= MAX_IMAGES:
            break


# display and save the results
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

columns = 4
rows = (len(wrong_images) + columns - 1) // columns

fig, axes = plt.subplots(
    rows,
    columns,
    figsize=(14, 3.5 * rows),
)

axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

for index, ax in enumerate(axes):
    if index >= len(wrong_images):
        ax.axis("off")
        continue

    image = wrong_images[index] * std + mean
    image = image.clamp(0, 1).permute(1, 2, 0).numpy()

    actual = class_names[wrong_labels[index]]
    predicted = class_names[wrong_predictions[index]]
    confidence = wrong_confidences[index] * 100

    ax.imshow(image)


    top3_text = "\n".join(
    f"{name}: {score:.1f}%"
    for name, score in wrong_top3[index]
)

    ax.set_title(
    f"Actual: {actual}\n"
    f"Predicted: {predicted} ({confidence:.1f}%)\n"
    f"Alternatives:\n{top3_text}",
    fontsize=8,
)

    # ax.set_title(
    #     f"Actual: {actual}\n"
    #     f"Predicted: {predicted}\n"
    #     f"Confidence: {confidence:.1f}%",
    #     fontsize=9,
    # )
    ax.axis("off")

fig.suptitle(
    "ResNet-18: Incorrect CIFAR-10 Predictions",
    fontsize=15,
)

fig.tight_layout()

output_path = os.path.join(
    OUTPUT_DIR,
    "resnet18_misclassified_examples.png",
)

fig.savefig(output_path, dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig)

print(f"Displayed {len(wrong_images)} misclassified images.")
print("Saved visualization:", output_path)