import matplotlib.pyplot as plt
import os
import json

with open(
    "pytorch_lesson10/plots/history.json",
    "r"
) as file:

    history = json.load(file)

epochs = range(
    1,
    len(history["train_loss"]) + 1
)

# Loss Curve
plt.figure()

plt.plot(
    epochs,
    history["train_loss"],
    label="Training Loss"
)

plt.plot(
    epochs,
    history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title(
    "Training and Validation Loss"
)

plt.legend()

plt.grid()

plt.savefig(
    "pytorch_lesson10/plots/loss_curve.png"
)

plt.show()

# Accuracy Curve
plt.figure()

plt.plot(
    epochs,
    history["train_acc"],
    label="Training Accuracy"
)

plt.plot(
    epochs,
    history["val_acc"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title(
    "Training and Validation Accuracy"
)

plt.legend()

plt.grid()

plt.savefig(
    "pytorch_lesson10/plots/accuracy_curve.png"
)

plt.show()