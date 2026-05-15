"""
BOW dataset only — pipes 20 Newsgroups into PyTorch DataLoaders.
The model, loss, optimizer, and training loop are yours to write.

Run: uv run python scripts/bow.py
Requires: scikit-learn (uv add scikit-learn)

Each batch yields:
    tokens:  LongTensor (batch, max_seq_len)   token IDs, padded with 0
    lengths: LongTensor (batch,)               real (unpadded) length per row
    labels:  LongTensor (batch,)               class index in [0, len(CATEGORIES))
"""

import re
from collections import Counter

import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.datasets import fetch_20newsgroups

CATEGORIES = ["rec.sport.hockey", "sci.med", "comp.graphics", "talk.politics.guns"]
VOCAB_SIZE = 10_000
BATCH_SIZE = 64
PAD, UNK = 0, 1

train_raw = fetch_20newsgroups(subset="train", categories=CATEGORIES,
                                remove=("headers", "footers", "quotes"))
test_raw = fetch_20newsgroups(subset="test", categories=CATEGORIES,
                               remove=("headers", "footers", "quotes"))


def tokenize(text):
    return re.findall(r"[a-z]+", text.lower())


# Build vocab from the most common training words. Indices 0/1 are reserved.
counts = Counter()
for text in train_raw.data:
    counts.update(tokenize(text))
vocab = {w: i + 2 for i, (w, _) in enumerate(counts.most_common(VOCAB_SIZE - 2))}


def encode(text):
    ids = [vocab.get(tok, UNK) for tok in tokenize(text)]
    return ids or [UNK]


class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.items = [(encode(t), int(l)) for t, l in zip(texts, labels)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch):
    # Variable-length sequences -> pad to the longest in the batch.
    seqs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs])
    padded = torch.full((len(seqs), int(lengths.max())), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, : len(s)] = torch.tensor(s)
    return padded, lengths, torch.tensor(labels)


train_ds = TextDataset(train_raw.data, train_raw.target)
test_ds = TextDataset(test_raw.data, test_raw.target)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)


if __name__ == "__main__":
    print(f"train docs: {len(train_ds)}  test docs: {len(test_ds)}")
    print(f"vocab size: {len(vocab) + 2} (incl. PAD=0, UNK=1)")
    tokens, lengths, labels = next(iter(train_loader))
    print(f"batch shapes: tokens={tuple(tokens.shape)} lengths={tuple(lengths.shape)} labels={tuple(labels.shape)}")
    print(f"first row, first 20 token IDs: {tokens[0, :20].tolist()}")
    print(f"first row label: {labels[0].item()} ({CATEGORIES[labels[0].item()]})")

# --- your model goes here ---

from torch import nn
class BadBoW(nn.Module):
    def __init__(self, embeddingsize, vocabsize):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(embeddingsize, vocabsize))
    def forward(self, x):
        return self.embedding[x]


m = BadBoW(embeddingsize=64, vocabsize=VOCAB_SIZE)
print(m.embedding.shape, m.embedding.shape[0])

        



