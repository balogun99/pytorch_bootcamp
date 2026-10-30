# import libraries
import torch

from torch.utils.data import DataLoader, Subset
from lesson13_nlp_pytorch.data.sentiment_data import texts, labels
from lesson13_nlp_pytorch.models.sentiment_model import SentimentModel
from lesson13_nlp_pytorch.data.text_dataset import (
    BATCH_SIZE,
    MAX_LENGTH,
    SentimentDataset,
    build_vocabulary,
    stratified_split
)

# load the saved checkpoint path
CHECKPOINT_PATH = (
    "lesson13_nlp_pytorch/checkpoints/sentiment_model.pth"
)

# configure the device
device = torch.device("cpu")

# reproduce same split used during training
train_indices, test_indices = stratified_split(
    texts,
    labels,
    test_ratio=0.2,
    seed=42,
)

training_texts = [texts[i] for i in train_indices]
vocabulary = build_vocabulary(training_texts)

# load the dataset
dataset = SentimentDataset(
    texts,
    labels,
    vocabulary,
    max_length=MAX_LENGTH,
)

test_dataset = Subset(dataset, test_indices)

# load the dataloader
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# load the checkpoint path
checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
)

# load the model
model = SentimentModel(
    vocab_size=len(checkpoint['vocabulary']),
    embedding_dim=(checkpoint['embedding_dim']),
    padding_idx=0,
).to(device)

# load the saved model
model.load_state_dict(checkpoint['model_state_dict'])

# evaluate the model
model.eval()

correct = 0
total = 0

print("\nTEST EXAMPLE PREDICTIONS")
print("=" * 60)

with torch.no_grad():
    for batch_ids, batch_labels in test_loader:
        logits = model(batch_ids)
        probabilities = torch.sigmoid(logits)
        predicitions = (probabilities >= 0.5).int()

        for probability, prediction, actual in zip(
            probabilities, predicitions, batch_labels
        ):

            index = test_indices[total]
            sentence = texts[index]

            predicted_label = prediction.item()
            actual_label = int(actual.item())

            predicted_sentiment = (
                "Positive" if predicted_label == 1 else "Negative"
            )

            actual_sentiment = (
                "Positive" if actual_label == 1 else "Negative"
            )

            status = (
                "CORRECT"
                if predicted_label == actual_label
                else "INCORRECT"
            )

            print(f"\nSentence: ", {sentence})
            print(f"Actual: ", {actual_sentiment})
            print(f"Predicted: ", {predicted_sentiment})
            print(f"Positive Probability: {probability.item():.4f}")
            print(f"Result: {status}")

            correct += int(predicted_label == actual_label)
            total += 1

print("\n" + "=" * 60)
print(f"Accuracy: {correct}/{total} = {correct / total:.2%}")
# python -m lesson13_nlp_pytorch.evaluate

# Proceed to Lesson 13.6 — Making Predictions with Your Trained Sentiment Model