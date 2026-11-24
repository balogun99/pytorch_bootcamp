# import pytorch
import torch
import numpy as np
from sklearn.metrics import(
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from lesson15_gru.dataset import create_dataloaders
from lesson15_gru.gru_model import GRUClassifier

# configuration
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT_PATH = (
    "lesson15_gru/checkpoints/gru_classifier.pth"
)

BATCH_SIZE = 32

# load test data
_, test_loader = create_dataloaders(
    batch_size=BATCH_SIZE
)

# load trained model
checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)

model = GRUClassifier(
    input_size=checkpoint["input_size"],
    hidden_size=checkpoint["hidden_size"],
    num_layers=checkpoint["num_layers"],
    num_classes=checkpoint["num_classes"]
).to(DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])

# switch to evaluation mode
model.eval()

# evaluation function
def evaluate_model(model, test_loader, device):

    all_predictions = []
    all_labels = []

    # disable gradient calculations
    with torch.no_grad():

        for sequences, labels in test_loader:

            sequences = sequences.to(device)

            # forward propagation
            outputs = model(sequences)

            # get predicted classes
            predictions = torch.argmax(
                outputs,
                dim=1
            )

            # collect predictions and actual labels
            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.numpy()
            )

    # convert to NumPy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    return all_predictions, all_labels


# calculate evaluation metrics
predictions, actual_labels = evaluate_model(
    model,
    test_loader,
    DEVICE
)

accuracy = accuracy_score(
    actual_labels,
    predictions
)

precision = precision_score(
    actual_labels,
    predictions,
    zero_division=0
)

recall = recall_score(
    actual_labels,
    predictions,
    zero_division=0
)

f1 = f1_score(
    actual_labels,
    predictions,
    zero_division=0
)

cm = confusion_matrix(
    actual_labels,
    predictions,
    labels=[0,1]
)

# display results
print("\nGRU MODEL EVALUATION")
print("==========================")

print("Device: ", DEVICE)
print("Testing Samples: ", len(actual_labels))

print("\nPERFORMANCE METRICS")
print("==========================")

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score : {f1:.4f}")

# confusion matrix
print("\nCONFUSION MATRIX")
print("========================")

print(cm)

print("\nCLASSIFICATION REPORT")
print("========================")


print(
    classification_report(
        actual_labels,
        predictions,
        labels=[0, 1],
        target_names=[
            "Low Frequency (Class 0)",
            "High Frequency (Class 1)"
        ],
        zero_division=0
    )
)

print("\nGRU MODEL EVALUATION COMPLETED")

# Accuracy: The model correctly classified all 200 samples.
# Precision: Every sequence predicted as high-frequency actually belonged to the high-frequency class.
# Recall: The model identified every actual high-frequency sequence.
# F1-score: The harmonic mean of precision and recall. A score of 1.0 means both are perfect.

