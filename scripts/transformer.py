"""
Decoder-only transformer — fill-in-the-blanks template.

Goal: implement a tiny GPT-style transformer from scratch, train it on a small
text corpus, and sample from it. Character-level tokenization keeps the data
pipeline trivial so you can focus on the model.

Shape conventions (used throughout):
    B = batch size
    T = sequence length (a.k.a. context / block size)
    C = embedding dim (a.k.a. n_embd, d_model)
    H = number of attention heads
    D = per-head dim = C // H
    V = vocab size
"""

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 0. Config
# ---------------------------------------------------------------------------
# Keep these small at first — you want a training run to finish in minutes
# on your laptop so you can iterate. Scale up once it learns.

BLOCK_SIZE = 128      # T: max context length
N_EMBD     = 192      # C: embedding dim
N_HEAD     = 6        # H: must divide N_EMBD
N_LAYER    = 4        # number of transformer blocks
DROPOUT    = 0.1

BATCH_SIZE = 32
LR         = 3e-4
MAX_ITERS  = 2000
EVAL_EVERY = 200

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


# ---------------------------------------------------------------------------
# 1. Data: character-level tokenizer + train/val split
# ---------------------------------------------------------------------------
# Grab any plain-text file (Shakespeare is the classic). Build a char->int
# vocabulary, encode the whole corpus into one long LongTensor, then slice
# random windows of length BLOCK_SIZE+1 for batches.

DATA_PATH = Path(__file__).parent / "input.txt"
# If the file doesn't exist yet, download tinyshakespeare:
#   curl -o scripts/input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

def load_data():
    """
    Returns:
        train_data: LongTensor of token ids
        val_data:   LongTensor of token ids
        vocab_size: int
        encode: Callable[[str], list[int]]
        decode: Callable[[list[int]], str]
    """
    stoi = {}
    itos = {}
    content = '' 
    ids = None
    with open(DATA_PATH, 'r') as f:
        content = f.read()
        chars = sorted(set(content))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        ids = torch.tensor([stoi[c] for c in content], dtype=torch.long)
    return ids[0: int(len(content)*.9)], ids[int(len(content)*.9) + 1:], len(chars), lambda s: [stoi[c] for c in s], lambda ints: ''.join([itos[x] for x in ints])

    # TODO:
    #   - read the file
    #   - sorted(set(text)) -> chars
    #   - stoi / itos dicts
    #   - encode/decode lambdas
    #   - 90/10 split
    ...


def get_batch(data: torch.Tensor):
    """
    Sample BATCH_SIZE random windows of length BLOCK_SIZE.

    Returns:
        x: LongTensor (B, T) — input tokens
        y: LongTensor (B, T) — targets, shifted by one
    """
    # TODO:
    #   - random start indices in [0, len(data) - BLOCK_SIZE - 1]
    #   - stack windows into (B, T) for x and (B, T) for y (shifted by 1)
    #   - move to device
    starts = torch.randint(0, len(data) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([data[s : s + BLOCK_SIZE]         for s in starts])
    y = torch.stack([data[s + 1 : s + 1 + BLOCK_SIZE] for s in starts])
    return x.to(device), y.to(device)


# ---------------------------------------------------------------------------
# 2. Multi-head causal self-attention
# ---------------------------------------------------------------------------
# The core idea: each token produces a query, key, and value vector. Attention
# weight from token i to token j is softmax(q_i · k_j / sqrt(D)). Causal mask
# zeros out j > i so the model can't peek at the future.
#
# Multi-head: split C into H heads of size D = C // H, run attention per head
# in parallel, then concat back to C and project.

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        # x : B,T,C
        # Linear, T, C
        self.Wq = nn.Linear(n_embd, n_embd)
        self.Wk = nn.Linear(n_embd, n_embd)
        self.Wv = nn.Linear(n_embd, n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        triangle = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", triangle)

        # TODO:
        #   - one nn.Linear(n_embd, 3 * n_embd) to produce q, k, v in one matmul
        #     (then split), OR three separate Linears — your call
        #   - output projection nn.Linear(n_embd, n_embd)
        #   - attn_dropout, resid_dropout (nn.Dropout)
        #   - register a causal mask as a buffer:
        #       self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C)  ->  (B, T, C)
        """
        B, T, C = x.shape
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        S = (Q @ K.transpose(-2, -1)) / math.sqrt(C // self.n_head)
        sm = F.softmax(S, dim=-1)
        out = sm @ V
        return self.proj @ out
        # TODO:
        #   1. project to q, k, v   -> each (B, T, C)
        #   2. reshape to heads     -> (B, H, T, D)   where D = C // H
        #   3. scores = q @ k.transpose(-2, -1) * (1 / sqrt(D))  -> (B, H, T, T)
        #   4. apply causal mask: scores.masked_fill(mask[:T,:T] == 0, -inf)
        #   5. att = softmax(scores, dim=-1); attn_dropout
        #   6. out = att @ v        -> (B, H, T, D)
        #   7. transpose+contiguous+view back to (B, T, C)
        #   8. output projection + resid_dropout
        ...


# ---------------------------------------------------------------------------
# 3. Position-wise feed-forward (MLP)
# ---------------------------------------------------------------------------
# Standard recipe: Linear(C -> 4C) -> GELU -> Linear(4C -> C) -> Dropout.
# Applied independently to each position.

class FeedForward(nn.Module):
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        # TODO: build the 2-layer MLP described above
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...


# ---------------------------------------------------------------------------
# 4. Transformer block
# ---------------------------------------------------------------------------
# Pre-norm variant (more stable than post-norm at small scale):
#     x = x + attn(layernorm(x))
#     x = x + mlp(layernorm(x))

class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        # TODO:
        #   - self.ln1, self.ln2 = nn.LayerNorm(n_embd)
        #   - self.attn = CausalSelfAttention(...)
        #   - self.mlp  = FeedForward(...)
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: residual + ln + sublayer, twice
        ...


# ---------------------------------------------------------------------------
# 5. Full model: token embed + positional embed + N blocks + head
# ---------------------------------------------------------------------------
# Token embedding: nn.Embedding(V, C)
# Positional embedding: a learned nn.Embedding(T, C) is fine for a tiny model.
#   (Sinusoidal or RoPE are alternatives — try one after you have it working.)
# Final layernorm before the unembedding projection (C -> V).
# Common trick: weight-tie the unembedding to the token embedding.

class GPT(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        # TODO:
        #   - self.tok_emb = nn.Embedding(vocab_size, N_EMBD)
        #   - self.pos_emb = nn.Embedding(BLOCK_SIZE, N_EMBD)
        #   - self.drop    = nn.Dropout(DROPOUT)
        #   - self.blocks  = nn.ModuleList([Block(...) for _ in range(N_LAYER)])
        #   - self.ln_f    = nn.LayerNorm(N_EMBD)
        #   - self.head    = nn.Linear(N_EMBD, vocab_size, bias=False)
        #   - (optional) weight tying: self.head.weight = self.tok_emb.weight
        #   - apply weight init (small normal for Linear/Embedding)
        ...

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """
        idx:     LongTensor (B, T)
        targets: LongTensor (B, T) or None

        Returns:
            logits: (B, T, V)
            loss:   scalar tensor or None
        """
        B, T = idx.shape
        # TODO:
        #   1. tok = self.tok_emb(idx)                  # (B, T, C)
        #   2. pos = self.pos_emb(arange(T, device=...))  # (T, C), broadcasts
        #   3. x = self.drop(tok + pos)
        #   4. run through blocks
        #   5. x = self.ln_f(x); logits = self.head(x)
        #   6. if targets is None: loss = None
        #      else: F.cross_entropy(logits.view(B*T, V), targets.view(B*T))
        ...

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0):
        """
        Greedy/sampled autoregressive generation.

        idx: (B, T_start)  ->  (B, T_start + max_new_tokens)
        """
        # TODO loop:
        #   - crop idx to last BLOCK_SIZE tokens (model only sees that much)
        #   - logits, _ = self(idx_cond)
        #   - take logits[:, -1, :] / temperature
        #   - probs = softmax(...)
        #   - next_tok = torch.multinomial(probs, num_samples=1)
        #   - idx = torch.cat([idx, next_tok], dim=1)
        ...


# ---------------------------------------------------------------------------
# 6. Training loop
# ---------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss(model, train_data, val_data, iters: int = 50):
    """Average loss over a few batches on each split — gives a less noisy signal."""
    # TODO:
    #   - model.eval()
    #   - for split in [train, val]: average loss over `iters` batches
    #   - model.train()
    ...


def train():
    train_data, val_data, vocab_size, encode, decode = load_data()
    model = GPT(vocab_size).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=LR)

    for it in range(MAX_ITERS):
        if it % EVAL_EVERY == 0:
            # TODO: log estimate_loss(...)
            ...
        pass


    return model, encode, decode


# ---------------------------------------------------------------------------
# 7. Sanity checks
# ---------------------------------------------------------------------------
# - Untrained model should have loss ~ ln(vocab_size). For 65 chars, ~4.17.
#   If yours is way off at step 0, your init or loss reduction is wrong.
# - Overfit a single batch: training loss should go to ~0 within a few hundred
#   steps. If not, your forward pass / gradient flow is broken.
# - Watch val loss — it should drop with train loss and then plateau. If train
#   keeps falling but val climbs, you're overfitting (expected on tiny data).
# - Generation from an untrained model = noise; after training on Shakespeare,
#   you should see word-shaped gibberish within a few hundred steps and
#   recognizable English within a couple thousand.

if __name__ == "__main__":
    model, encode, decode = train()
    start = torch.zeros((1, 1), dtype=torch.long, device=device)  # token 0 as BOS-ish
    out = model.generate(start, max_new_tokens=500)
    print(decode(out[0].tolist()))
