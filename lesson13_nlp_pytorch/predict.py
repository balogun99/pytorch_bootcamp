# import libraries
import torch

from lesson13_nlp_pytorch.data.text_dataset import tokenize
from lesson13_nlp_pytorch.models.sentiment_model import SentimentModel

# configuration
CHECKPOINT_PATH = (
    "lesson13_nlp_pytorch/checkpoints/sentiment_model.pth"
)

device = torch.device("cpu")

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

PAD_ID = 0
UNK_ID = 1

print("\nLoading trained sentiment model...")

# load trained model checkpoint
checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
)

vocabulary = checkpoint["vocabulary"]
max_length = checkpoint["max_length"]
embedding_dim = checkpoint["embedding_dim"]

# recreate the model architecture
model = SentimentModel(
    vocab_size=len(vocabulary),
    embedding_dim=embedding_dim,
    padding_idx=PAD_ID,
).to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Model loaded successfully!\n")

print("Vocabulary Size: ", len(vocabulary))
print("Max Sequence Length: ", max_length)

# text preprocessing function
def preprocess_text(text):
    # tokenize the input text
    tokens = tokenize(text)

    # convert tokens into numerical IDs
    input_ids = [
        vocabulary.get(
            token, UNK_ID
        )
        for token in tokens
    ]

    # truncate sequences that are too long
    input_ids = input_ids[:max_length]

    # add padding when sequences are too short
    padding_length = max_length - len(input_ids)

    input_ids.extend(
        [PAD_ID] * padding_length
    )

    # convert numerical IDs into a PyTorch tensor
    input_tensor = torch.tensor(
        [input_ids],
        dtype=torch.long,
    )

    return input_tensor, tokens

# sentiment prediction function
def predict_sentiment(sentence):

    # reject empty input
    if not sentence.strip():

        print("\nPlease enter a valid sentence.")

        return

    # preprocess the sentence
    input_tensor, tokens = preprocess_text(sentence)

    # identify words not present in the vocabulary
    unknown_words = [
        token
        for token in tokens
        if token not in vocabulary
    ]

    # perform an inference (generate a predictions)
    with torch.no_grad():

        logits = model(input_tensor)

        positive_probability = torch.sigmoid(
            logits
        ).item()

    # calculate negative probability
    negative_probability = (
        1 - positive_probability
    )

    # classify the sentiment
    if positive_probability >= 0.5:

        sentiment = "Positive"

    else:

        sentiment = "Negative"

    # calculate prediction confidence
    confidence = max(
        positive_probability,
        negative_probability,
    )

    # display results
    print("\n" + "=" * 50)

    print("SENTIMENT PREDICTION RESULTS")

    print("=" * 50)

    print("Input Sentence: ", sentence)

    print("Tokens: ", tokens)

    print("Unknown Words: ", unknown_words)

    print("\nPredicted Sentiment: ", sentiment)

    print(
        f"Positive Probability: {positive_probability:.2%}"
    )

    print(
        f"Negative Probability: {negative_probability:.2%}"
    )

    print(
        f"Prediction Confidence: {confidence:.2%}"
    )

    print("=" * 50)

# interactive prediction interface
def main():

    print("=" + "=" * 50)

    print ("SENTIMENT PREDICTION INTERFACE")

    print("=" * 50)

    print("\nEnter a sentence to analyse its sentiment.")

    print("Type 'exit' to quit.")

    while True:

        sentence = input(
            "\nEnter a Sentence: "
        )

        if sentence.strip().lower() == "exit":

            print("\nExiting the Sentiment Prediction Interface. Goodbye!")

            break

        predict_sentiment(sentence)

# execute the program
if __name__ == "__main__":
    main()