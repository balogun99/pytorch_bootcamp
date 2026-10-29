# import librarie
import re
import random
import torch

from torch.utils.data import Dataset, DataLoader, Subset
if __package__:
    from .sentiment_data import texts, labels
else:
    from sentiment_data import texts, labels

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
MAX_LENGTH = 12
BATCH_SIZE = 4

def tokenize(text):
    """Build a word-to-integer mapping from training text only."""
    return re.findall(r"\b\w+\b", text.lower())

def build_vocabulary(training_texts):
    """Build a word-to-integer mapping from training text only."""
    vocabulary = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1
    }

    for text in training_texts:
        for token in tokenize(text):
            if token not in vocabulary:
                vocabulary[token] = len(vocabulary)

    return vocabulary

def encode_text(text, vocabulary, max_length=MAX_LENGTH):
    """Convert a sentence into a fixed-length tensor of token IDs."""
    token_ids = [
        vocabulary.get(token, vocabulary[UNK_TOKEN])
        for token in tokenize(text)
    ]

    # truncate sentences that are too long
    token_ids = token_ids[:max_length]
    
    # Pad shorter sentences with the PAD token ID (0).
    token_ids += [vocabulary[PAD_TOKEN]] * (
        max_length - len (token_ids)
    )

    return torch.tensor(token_ids, dtype=torch.long)

class SentimentDataset(Dataset):
    def __init__(
        self, 
        texts,
        labels,
        vocabulary,
        max_length=MAX_LENGTH,
    ):

        if len(texts) != len(labels):
            raise ValueError("Texts and Labels must have equal length!")

        self.texts = texts
        self.labels = labels
        self.vocabulary = vocabulary
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        input_ids = encode_text(
            self.texts[index],
            self.vocabulary,
            self.max_length,
        )

        label = torch.tensor(
            self.labels[index],
            dtype=torch.float32,
        )

        return input_ids, label

def stratified_split(texts, labels, test_ratio=0.2, seed=42):
    """Split each class separately to preserve class proportions."""
    rng = random.Random(seed)
    train_indices = []
    test_indices = []

    for class_label in sorted(set(labels)):
        class_indices = [
            i for i, label in enumerate(labels)
            if label == class_label
        ]

        rng.shuffle(class_indices)
        test_count = max(1, round(len(class_indices) * test_ratio))

        test_indices.extend(class_indices[:test_count])
        train_indices.extend(class_indices[test_count:])

        rng.shuffle(train_indices)
        rng.shuffle(test_indices)

        return train_indices, test_indices

def main():
    train_indices, test_indices = stratified_split(texts, labels)

    training_texts = [texts[i] for i in train_indices]

    # Never use test text to build the vocabulary.
    vocabulary = build_vocabulary(training_texts)

    dataset = SentimentDataset(texts, labels, vocabulary)

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size = BATCH_SIZE,
        shuffle = True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size = BATCH_SIZE,
        shuffle = False,
    )

    print("Total Examples: ", len(dataset))
    print("Training Examples: ", len(train_dataset))
    print("Testing Examples: ", len(test_dataset))
    print("Vocabulary Size: ", len(vocabulary))
    print("First Vocabulary Entries: ", list(vocabulary.items())[:12])

    # inspect a single-encoded sentence
    sample_text = texts[train_indices[0]]
    sample_ids = encode_text(sample_text, vocabulary)

    print("Sample Text: ", sample_text)
    print("Encoded IDS: ", sample_ids.tolist())
    print("Tensor Shape: ", tuple(sample_ids.shape))

    # inspect one batch
    input_ids, batch_labels = next(iter(train_loader))

    print("\nBatch Input Shape: ", tuple(input_ids.shape))
    print("Batch Label Shape: ", tuple(batch_labels.shape))
    print("Batch Input dtype: ", input_ids.dtype)
    print("Batch Label dtype: ", batch_labels.dtype)
    print("Batch Labels : ", batch_labels.tolist())

    # Check that the inputs and labels are compatible with our model.
    assert input_ids.shape == (BATCH_SIZE, MAX_LENGTH)
    assert input_ids.dtype == torch.long
    assert batch_labels.dtype == torch.float32

    print("\nData Pipeline checks passed!")

if __name__ == "__main__":
    main()