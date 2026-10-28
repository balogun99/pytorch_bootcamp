# We'll use an Embedding + average pooling + Linear model.
# - Embedding: converts each token ID into a trainable vector.
# - Average pooling: combines the word vectors into one sentence representation.
# - Linear layer: uses that representation to predict sentiment.

# build the model
import torch
import torch.nn as nn

class SentimentModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=32,
        padding_idx=0,
    ):

        super.__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )

    def forward(self, token_ids):
        # token_ids shape: (batch_size, sequence_length)
        embeddings = self.embedding(token_ids)

        # Ignore padding tokens when calculating the average
        mask = (token_ids != 0).unsqueeze(-1)
        mask = mask.to(embeddings.dtype)

        summed_embeddings = (embeddings * mask).sum(dim=1)
        token_counts = mask.sum(dim=1).clamp(min=1)

        sentence_embeddings = summed_embeddings / token_counts

        # Return one raw score (logit) per sentence
        return self.classifier(sentence_embeddings).sequeeze(1)