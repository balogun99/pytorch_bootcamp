# import libraries
import torch
import numpy as np

from lesson15_gru.gru_model import GRUClassifier

# configuration

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT_PATH = "lesson15_gru/checkpoints/gru_classifier.pth"

SEQUENCE_LENGTH = 30

# load trained model


def load_model():

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    # Reconstruct the GRU architecture
    model = GRUClassifier(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        num_classes=checkpoint["num_classes"]
    )

    # Load trained parameters
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)

    # Switch to evaluation mode
    model.eval()

    print("GRU MODEL LOADED SUCCESSFULLY")
    print("Device:", DEVICE)
    print("Training Epoch:", checkpoint["epoch"])
    print("Training Accuracy:", checkpoint["accuracy"])

    return model


# def load_model():

#     checkpoint = torch.load( #This retrieves the saved model parameters and configuration from your training stage.
#         CHECKPOINT_PATH,
#         map_location=DEVICE
#     )

#     config = checkpoint["config"]

#     model = GRUClassifier(
#         input_size=config["input_size"],
#         hidden_size=config["hidden_size"],
#         num_layers=config["num_layers"],
#         num_classes=config["num_classes"]
#     )

#     model.load_state_dict(checkpoint["model_state_dict"])

#     model.to(DEVICE)

#     model.eval()

#     print("GRU MODEL LOADED SUCCESSFULLY")
#     print("Device:", DEVICE)

#     return model

# generate a new time-series sequence

def generate_new_sequence(
    frequency,
    sequence_length=30,
    noise_std=0.15,
    seed=100
):

    rng = np.random.default_rng(seed)

    time_steps = np.linspace(
        0, 1, sequence_length
    )

    amplitude = rng.uniform(0.8, 1.2)

    phase = rng.uniform(0, 2 * np.pi)

    signal = amplitude * np.sin(
        2 * np.pi * frequency * time_steps + phase
    )

    noise = rng.normal(
        0,
        noise_std,
        sequence_length
    )

    signal = signal + noise

    signal = signal.astype(np.float32)

    # Convert to PyTorch tensor
    signal_tensor = torch.tensor(
        signal,
        dtype=torch.float32
    )

    # Shape: [sequence_length, input_size]
    signal_tensor = signal_tensor.unsqueeze(-1)

    # Add batch dimension
    # Shape: [1, sequence_length, input_size]
    signal_tensor = signal_tensor.unsqueeze(0)

    return signal_tensor, signal


# model prediction

def predict(model, sequence):

    sequence = sequence.to(DEVICE)

    with torch.no_grad():

        logits = model(sequence)

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

    confidence = probabilities[0][predicted_class].item()

    return predicted_class, confidence, probabilities

# main function 

def main():

    print("\nGRU TIME-SERIES PREDICTION")
    print("=" * 40)

    model = load_model()

    # Generate a new low-frequency signal
    sequence, original_signal = generate_new_sequence(
        frequency=5,
        seed=100
    ) # This generates a new low-frequency sine wave with random amplitude, phase and noise.

    print("Actual Frequency: 5")
    print("Expected Class: 1")
    print("Expected Category: High Frequency")

    # print("Actual Frequency: 2")
    # print("Expected Class: 0")
    # print("Expected Category: Low Frequency")

    print("\nINPUT INFORMATION")
    print("=" * 40)

    print("Sequence Shape:", sequence.shape)
    print("Sequence Length:", SEQUENCE_LENGTH)

    # Make prediction
    predicted_class, confidence, probabilities = predict(
        model,
        sequence
    )

    # Map predicted class to category
    class_names = {
        0: "Low Frequency",
        1: "High Frequency"
    }

    predicted_label = class_names[predicted_class]

    print("\nPREDICTION RESULTS")
    print("=" * 40)

    print("Predicted Class:", predicted_class)
    print("Predicted Category:", predicted_label)

    print(
        f"Prediction Confidence: {confidence * 100:.2f}%"
    )

    print("\nCLASS PROBABILITIES")
    print("=" * 40)

    print(
        f"Low Frequency: {probabilities[0][0].item() * 100:.2f}%"
    )

    print(
        f"High Frequency: {probabilities[0][1].item() * 100:.2f}%"
    )

    print("\nPREDICTION COMPLETED!")

# run the file

if __name__ == "__main__":
    main()
