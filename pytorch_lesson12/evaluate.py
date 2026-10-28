
import os
import csv
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.resnet_model import build_resnet18
from evaluation.metrics import (
    calculate_metrics,
    plot_confusion_matrix,
)

# configuration

LESSON_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(LESSON_DIR)

DATA_DIR = os.path.join(PROJECT_DIR, "data")
PLOTS_DIR = os.path.join(LESSON_DIR, "plots")

# Choose which model to evaluate:
# "finetuned" = fine-tuned model
# "baseline"  = original classifier-only model

# MODEL_VERSION = "finetuned"
MODEL_VERSION = "baseline"

CHECKPOINTS = {
    "baseline": os.path.join(
        PROJECT_DIR,
        "checkpoints",
        "resnet18_best.pth",
    ),
    "finetuned": os.path.join(
        PROJECT_DIR,
        "checkpoints",
        "resnet18_finetuned_best.pth",
    ),
}

if MODEL_VERSION not in CHECKPOINTS:
    raise ValueError(
        "MODEL_VERSION must be 'baseline' or 'finetuned'."
    )

CHECKPOINT_PATH = CHECKPOINTS[MODEL_VERSION]

os.makedirs(PLOTS_DIR, exist_ok=True)

BATCH_SIZE = 64
IMAGE_SIZE = 64

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)
print("Model version:", MODEL_VERSION)

# load checkpoint

if not os.path.isfile(CHECKPOINT_PATH):
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT_PATH}\n"
        "Make sure the selected training experiment has completed."
    )

print("Loading checkpoint:", CHECKPOINT_PATH)

# This checkpoint is your own trusted model file.
checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
)

saved_image_size = checkpoint.get("image_size", IMAGE_SIZE)

if saved_image_size != IMAGE_SIZE:
    raise ValueError(
        f"Checkpoint image size is {saved_image_size}, "
        f"but evaluation uses {IMAGE_SIZE}."
    )

# load cifar-10 test data

test_transform = transforms.Compose([
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
    transform=test_transform,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

class_names = test_dataset.classes

print("Test images:", len(test_dataset))

# build and load model

model = build_resnet18(
    num_classes=len(class_names),
    pretrained=False,
)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

print(
    "Best validation accuracy: "
    f"{checkpoint['val_accuracy']:.4f}"
)

# collect predictions

all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)

        outputs = model(images)
        predictions = outputs.argmax(dim=1).cpu()

        all_predictions.extend(predictions.tolist())
        all_labels.extend(labels.tolist())

# classification metrics

results = calculate_metrics(
    all_labels,
    all_predictions,
    class_names,
)

print("\n" + "=" * 60)
print(f"CIFAR-10 TEST RESULTS — {MODEL_VERSION.upper()}")
print("=" * 60)

print(f"Test accuracy: {results['accuracy']:.4f}")
print(f"Test accuracy: {results['accuracy'] * 100:.2f}%")

print("\nCLASSIFICATION REPORT")
print(results["classification_report"])

# save confusion matrix

matrix = results["confusion_matrix"]

confusion_path = os.path.join(
    PLOTS_DIR,
    f"resnet18_{MODEL_VERSION}_confusion_matrix.png",
)

plot_confusion_matrix(
    matrix,
    class_names,
    confusion_path,
)

# save per class metrics

print("\nPER-CLASS ERROR ANALYSIS")
print("-" * 75)

print(
    f"{'Class':<15}"
    f"{'Precision':>12}"
    f"{'Recall':>12}"
    f"{'F1-score':>12}"
    f"{'Errors':>10}"
)

for item in results["per_class"]:
    print(
        f"{item['class']:<15}"
        f"{item['precision']:>12.4f}"
        f"{item['recall']:>12.4f}"
        f"{item['f1_score']:>12.4f}"
        f"{item['errors']:>10}"
    )

csv_path = os.path.join(
    PLOTS_DIR,
    f"resnet18_{MODEL_VERSION}_per_class_metrics.csv",
)

with open(csv_path, "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "class",
            "precision",
            "recall",
            "f1_score",
            "support",
            "errors",
        ],
    )

    writer.writeheader()
    writer.writerows(results["per_class"])

print("\nPer-class metrics saved:", csv_path)


# top-class confusion

confusions = []

for true_index in range(len(class_names)):
    for pred_index in range(len(class_names)):
        if true_index == pred_index:
            continue

        count = int(matrix[true_index, pred_index])

        if count > 0:
            confusions.append((
                count,
                class_names[true_index],
                class_names[pred_index],
            ))

confusions.sort(reverse=True)

print("\nTOP 10 CLASS CONFUSIONS")
print("-" * 50)

for count, true_class, predicted_class in confusions[:10]:
    print(
        f"Actual: {true_class:<12} "
        f"Predicted: {predicted_class:<12} "
        f"Count: {count}"
    )


# save summary for comparison

summary_path = os.path.join(
    PLOTS_DIR,
    "resnet18_comparison_summary.csv",
)

summary_row = {
    "model": MODEL_VERSION,
    "validation_accuracy": checkpoint["val_accuracy"],
    "test_accuracy": results["accuracy"],
    "macro_precision": sum(
        item["precision"] for item in results["per_class"]
    ) / len(class_names),
    "macro_recall": sum(
        item["recall"] for item in results["per_class"]
    ) / len(class_names),
    "macro_f1": sum(
        item["f1_score"] for item in results["per_class"]
    ) / len(class_names),
}

write_header = not os.path.exists(summary_path)

with open(summary_path, "a", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(summary_row.keys()),
    )

    if write_header:
        writer.writeheader()

    writer.writerow(summary_row)

print("\nComparison summary saved:", summary_path)
print("\nEvaluation complete!")


# Step 3 — Compare with your baseline
# After the fine-tuned evaluation completes, change:
# MODEL_VERSION = "finetuned"



# to:
# MODEL_VERSION = "baseline"



# Run the script again. Both runs use the same CIFAR-10 test set, image size, and evaluation metrics.

# Your output files will be stored in pytorch_lesson12/plots/, including separate confusion matrices 
# and per-class CSV files. The summary CSV will append one row per run, allowing you to compare the two models.

# Your original benchmark remains 71.56% test accuracy. Let's see whether fine-tuning improves it, chief. 
# Send me the fine-tuned test results when you're done, and we'll interpret the comparison.