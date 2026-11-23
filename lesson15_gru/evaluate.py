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
from lesson15_gru.model import GRUClassifier

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