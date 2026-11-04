
# lesson13_nlp_pytorch/evaluate.py


# lesson13_nlp_pytorch/evaluate.py

import torch
from torch.utils.data import DataLoader, Subset

from lesson13_nlp_pytorch.data.sentiment_data import texts, labels

from lesson13_nlp_pytorch.data.text_dataset import (
    BATCH_SIZE,
    SentimentDataset,
    tokenize,
)

from lesson13_nlp_pytorch.models.sentiment_model import SentimentModel


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

CHECKPOINT_PATH = (
    "lesson13_nlp_pytorch/checkpoints/sentiment_model.pth"
)

device = torch.device("cpu")


# --------------------------------------------------
# 2. LOAD CHECKPOINT
# --------------------------------------------------

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
)

vocabulary = checkpoint["vocabulary"]
max_length = checkpoint["max_length"]
embedding_dim = checkpoint["embedding_dim"]

test_indices = checkpoint["test_indices"]


# --------------------------------------------------
# 3. RECREATE TEST DATASET
# --------------------------------------------------

dataset = SentimentDataset(
    texts,
    labels,
    vocabulary,
    max_length=max_length,
)

test_dataset = Subset(
    dataset,
    test_indices,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# --------------------------------------------------
# 4. LOAD TRAINED MODEL
# --------------------------------------------------

model = SentimentModel(
    vocab_size=len(vocabulary),
    embedding_dim=embedding_dim,
    padding_idx=0,
).to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# --------------------------------------------------
# 5. DISPLAY CHECKPOINT INFORMATION
# --------------------------------------------------

print("=" * 55)
print("SENTIMENT MODEL EVALUATION")
print("=" * 55)

print("Checkpoint Epoch:", checkpoint["epoch"])

print(
    "Best Validation Loss:",
    f"{checkpoint['best_validation_loss']:.4f}"
)

print("Vocabulary Size:", len(vocabulary))
print("Test Examples:", len(test_dataset))


# --------------------------------------------------
# 6. UNKNOWN WORD ANALYSIS
# --------------------------------------------------

print("\nUNKNOWN WORD ANALYSIS")
print("=" * 55)

for index in test_indices:

    sentence = texts[index]

    unknown_words = [
        word
        for word in tokenize(sentence)
        if word not in vocabulary
    ]

    print("\nSentence:", sentence)
    print("Unknown words:", unknown_words)


# --------------------------------------------------
# 7. PREDICT TEST EXAMPLES
# --------------------------------------------------

actual_labels = []
predicted_labels = []
positive_probabilities = []

print("\nTEST EXAMPLE PREDICTIONS")
print("=" * 55)

with torch.no_grad():

    for batch_ids, batch_labels in test_loader:

        logits = model(batch_ids)

        probabilities = torch.sigmoid(logits).view(-1)

        predictions = (
            probabilities >= 0.5
        ).int()

        batch_labels = batch_labels.view(-1).int()

        for probability, prediction, actual in zip(
            probabilities,
            predictions,
            batch_labels,
        ):

            index = len(actual_labels)

            sentence = texts[test_indices[index]]

            predicted_label = prediction.item()
            actual_label = actual.item()

            predicted_sentiment = (
                "Positive"
                if predicted_label == 1
                else "Negative"
            )

            actual_sentiment = (
                "Positive"
                if actual_label == 1
                else "Negative"
            )

            status = (
                "CORRECT"
                if predicted_label == actual_label
                else "INCORRECT"
            )

            print("\nSentence:", sentence)
            print("Actual:", actual_sentiment)
            print("Predicted:", predicted_sentiment)

            print(
                "Positive Probability:",
                f"{probability.item():.4f}"
            )

            print("Result:", status)

            # Store predictions for metric calculations
            actual_labels.append(actual_label)
            predicted_labels.append(predicted_label)

            positive_probabilities.append(
                probability.item()
            )


# --------------------------------------------------
# 8. CALCULATE CONFUSION MATRIX
# --------------------------------------------------

TP = 0
TN = 0
FP = 0
FN = 0

for actual, predicted in zip(
    actual_labels,
    predicted_labels,
):

    if actual == 1 and predicted == 1:
        TP += 1

    elif actual == 0 and predicted == 0:
        TN += 1

    elif actual == 0 and predicted == 1:
        FP += 1

    elif actual == 1 and predicted == 0:
        FN += 1


# --------------------------------------------------
# 9. CALCULATE EVALUATION METRICS
# --------------------------------------------------

total = len(actual_labels)

correct = TP + TN
incorrect = FP + FN

accuracy = correct / total if total > 0 else 0

precision = (
    TP / (TP + FP)
    if (TP + FP) > 0
    else 0
)

recall = (
    TP / (TP + FN)
    if (TP + FN) > 0
    else 0
)

f1_score = (
    2 * precision * recall / (precision + recall)
    if (precision + recall) > 0
    else 0
)


# --------------------------------------------------
# 10. DISPLAY CONFUSION MATRIX
# --------------------------------------------------

print("\n" + "=" * 55)
print("CONFUSION MATRIX")
print("=" * 55)

print("\n                  PREDICTED")
print("                Positive  Negative")

print(f"ACTUAL Positive     {TP:2d}        {FN:2d}")
print(f"       Negative     {FP:2d}        {TN:2d}")

print("\nTrue Positive (TP):", TP)
print("True Negative (TN):", TN)
print("False Positive (FP):", FP)
print("False Negative (FN):", FN)


# --------------------------------------------------
# 11. DISPLAY PRECISION, RECALL AND F1-SCORE
# --------------------------------------------------

print("\n" + "=" * 55)
print("CLASSIFICATION METRICS")
print("=" * 55)

print(f"Accuracy:  {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall:    {recall:.2%}")
print(f"F1-score:  {f1_score:.2%}")


# --------------------------------------------------
# 12. FINAL EVALUATION SUMMARY
# --------------------------------------------------

print("\n" + "=" * 55)
print("FINAL EVALUATION RESULTS")
print("=" * 55)

print(f"Correct Predictions: {correct}/{total}")
print(f"Incorrect Predictions: {incorrect}/{total}")

print(f"Accuracy: {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1-score: {f1_score:.2%}")

print("\nEvaluation completed successfully!")


# python -m lesson13_nlp_pytorch.evaluate

# Proceed to Lesson 13.6 — Making Predictions with Your Trained Sentiment Model