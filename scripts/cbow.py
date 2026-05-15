"""
CBOW dataset only — produces (context, target) pairs for word2vec-style training.

The supervision signal: given the words around a position, predict the missing
center word. No human labels needed — the text itself provides the targets.

Run: uv run python scripts/cbow.py
Requires: scikit-learn (uv add scikit-learn)

Each batch yields:
    context:  LongTensor (batch, 2*WINDOW)   token IDs of surrounding words
    target:   LongTensor (batch,)            token ID of the center word
"""

import re
from collections import Counter

import torch
from torch.nn.functional import cross_entropy
from torch.onnx.symbolic_opset12 import cross_entropy_loss
from torch.utils.data import DataLoader, Dataset
from torch import nn
from sklearn.datasets import fetch_20newsgroups

VOCAB_SIZE = 10_000
WINDOW = 2          # 2 words on each side → context size = 4
BATCH_SIZE = 256
PAD, UNK = 0, 1


def tokenize(text):
    return re.findall(r"[a-z]+", text.lower())


# Use 20 Newsgroups train split as a text corpus (we don't care about its labels).
raw = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))

counts = Counter()
for text in raw.data:
    counts.update(tokenize(text))
vocab = {w: i + 2 for i, (w, _) in enumerate(counts.most_common(VOCAB_SIZE - 2))}
id_to_word = {i: w for w, i in vocab.items()}
id_to_word[PAD] = "<PAD>"
id_to_word[UNK] = "<UNK>"


def encode(text):
    return [vocab.get(tok, UNK) for tok in tokenize(text)]


# Flatten the whole corpus into one long token stream.
stream = []
for text in raw.data:
    stream.extend(encode(text))
stream = torch.tensor(stream, dtype=torch.long)


class CBOWDataset(Dataset):
    """Each training example: (window of W context tokens on each side) → center token."""
    def __init__(self, stream, window):
        self.stream = stream
        self.window = window

    def __len__(self):
        # Skip positions where a full context window doesn't fit at either end.
        return len(self.stream) - 2 * self.window

    def __getitem__(self, i):
        center = i + self.window
        context = torch.cat([
            self.stream[center - self.window : center],
            self.stream[center + 1 : center + self.window + 1],
        ])
        target = self.stream[center]
        return context, target


train_ds = CBOWDataset(stream, WINDOW)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)


if __name__ == "__main__":
    print(f"corpus tokens: {len(stream):,}")
    print(f"vocab size:    {len(vocab) + 2} (incl. PAD=0, UNK=1)")
    print(f"training pairs: {len(train_ds):,}")
    context, target = next(iter(train_loader))
    print(f"batch shapes:  context={tuple(context.shape)}  target={tuple(target.shape)}")

    ctx, tgt = train_ds[1000]
    ctx_words = [id_to_word[i.item()] for i in ctx]
    print(f"\nexample pair:")
    print(f"  context words: {ctx_words}")
    print(f"  target word:   {id_to_word[tgt.item()]!r}")

# --- your model goes here ---


class M(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedder = nn.Parameter(torch.randn(vocab_size, embedding_dim))
        self.proj = nn.Parameter(torch.randn(embedding_dim,vocab_size))
    def forward(self, x):
        lookup =  self.embedder[x]
        mean = lookup.mean(dim=1)
        logits = mean @ self.proj 
        return logits 

m = M(vocab_size=VOCAB_SIZE, embedding_dim=100)

context, target = next(iter(train_loader))
print(context.shape, target.shape)   # (256, 4) and (256,)
out = m(context)
print(out.shape)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(m.parameters(), lr=1e-3)

for epoch in range(2):
    for i, (context, target) in enumerate(train_loader):
        res = m(context)
        loss = loss_fn(res, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i % 200 == 0:
            print(f"step {i}: loss={loss.item():.3f}")


import torch.nn.functional as F

embeddings = m.embedder.detach()


def nearest(word, k=10):
    if word not in vocab:
        print(f"  {word!r} not in vocab")
        return
    query = embeddings[vocab[word]]
    sims = F.cosine_similarity(embeddings, query.unsqueeze(0), dim=1)
    top = sims.topk(k + 1).indices[1:]
    for idx in top:
        print(f"  {id_to_word[idx.item()]:20s} {sims[idx].item():+.3f}")


def analogy(a, b, c, k=5):
    for w in (a, b, c):
        if w not in vocab:
            print(f"  {w!r} not in vocab")
            return
    vec = embeddings[vocab[b]] - embeddings[vocab[a]] + embeddings[vocab[c]]
    sims = F.cosine_similarity(embeddings, vec.unsqueeze(0), dim=1)
    for idx in sims.topk(k).indices:
        print(f"  {id_to_word[idx.item()]:20s} {sims[idx].item():+.3f}")


for word in ["hockey", "computer", "doctor", "god", "car"]:
    print(f"\nNearest to {word!r}:")
    nearest(word)

print("\nman : woman :: king : ?")
analogy("man", "woman", "king")




ctx, tgt = train_ds[1000]
print([id_to_word[i.item()] for i in ctx], "→", id_to_word[tgt.item()])



        