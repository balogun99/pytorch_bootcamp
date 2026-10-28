# import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)
from torch import threshold

def calculate_metrics(y_true, y_pred, class_names):
    """Calculate overall and per-class classification metrics."""

    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=list(range(len(class_names))),
            zero_division=0
        )
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        digits=4,
        target_names=class_names,
        zero_division=0
    )

    per_class = []

    for index, class_name in enumerate(class_names):
        per_class.append({
            "class": class_name,
            "precision": precision[index],
            "recall": recall[index],
            "f1_score": f1[index],
            "support": int(support[index]),
            "errors": int(matrix[index].sum() - matrix[index, index]),
        })

    return {
        "accuracy": accuracy,
        "confusion_matrix": matrix,
        "classification_report": report,
        "per_class": per_class
    }

def plot_confusion_matrix(matrix, class_names, save_path):
    """Plot and save a confusion matrix."""

    fig, ax = plt.subplots(figsize=(11,9))

    image = ax.imshow(matrix, interpolation="nearest")
    fig.colorbar(image, ax=ax, label="Number of Images")

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title="CIFAR-10 Confusion Matrix",
    )

    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )

    threshold = matrix.max() / 2 if matrix.size else 0

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center",
                color=(
                    "white"
                    if matrix[row, col] > threshold
                    else "black"
                ),
                fontsize=8,
            )


    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Confusion matrix saved:", save_path)

# This module calculates precision, recall, F1-score, support, and misclassification counts for each class.
# - Support: number of actual test images belonging to a class.
# - Errors: number of images from that actual class predicted as another class.
# - Confusion matrix: rows represent actual classes; columns represent predicted classes.