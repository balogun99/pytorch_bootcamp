from matplotlib.pyplot import cla
import torch
from sklearn.metrics import(
    classification_report,
    confusion_matrix,
    accuracy_score
)

@torch.no_grad()
def collect_prediction(model, dataloader, device):
    model.eval()

    all_predictions = []
    all_labels = []

    for images, labels in dataloader:
        images = images.to(device)

        outputs = model(images)
        predictions = outputs.argmax(
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().tolist()
        )
        all_labels.extend(labels.tolist())

    return all_predictions, all_labels

def evaluate_classification(y_true, y_pred, classes):
    print(
       f"Overall Accuracy: "
       f"{accuracy_score(y_true, y_pred) * 100:.2f}%" 
    )

    print("Classification Report: ")
    print(
        classification_report(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        zero_division=0
        )
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
    )

    print(
        "\nConfusion Matrix: "
    )
    print(matrix)

    return matrix

# This separates prediction collection from metric calculation, 
# making the functions reusable across different models.