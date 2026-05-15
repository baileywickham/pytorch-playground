"""
Speculative decoding — fill-in-the-blanks template.

Goal: use a small "draft" model to propose K tokens, then verify them in a
single forward pass of a larger "target" model. Accepted tokens are kept;
the first rejected token is resampled from a corrected distribution.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# 1. Models
# ---------------------------------------------------------------------------
# Pick a draft/target pair that share a tokenizer (e.g. two sizes of the
# same family). The draft should be meaningfully smaller/faster.

DRAFT_MODEL_NAME = "gpt2"
TARGET_MODEL_NAME = "gpt2-xl"

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_NAME)
draft = AutoModelForCausalLM.from_pretrained(DRAFT_MODEL_NAME).to(device).eval()
target = AutoModelForCausalLM.from_pretrained(TARGET_MODEL_NAME).to(device).eval()


# ---------------------------------------------------------------------------
# 2. Draft step: propose K tokens autoregressively
# ---------------------------------------------------------------------------
@torch.no_grad()
def draft_tokens(input_ids: torch.Tensor, k: int):
    """
    Run the draft model k times, sampling one token per step.

    Returns:
        proposed_ids: LongTensor of shape (k,) — the k proposed tokens
        draft_probs:  FloatTensor of shape (k, vocab) — q(x) at each step,
                      conditioned on input_ids + previously proposed tokens
    """
    output = torch.zeros(k, 50257, device=device)
    proposed_ids = torch.zeros(1, 0, dtype=int, device=device)
    newinput = input_ids.clone()
    for i in range(k):
        # print(input_ids)
        logits = draft(newinput).logits
        softmaxxed = F.softmax(logits[:, -1, :], dim=-1)
        token = torch.multinomial(softmaxxed, num_samples=1)
        output[i] = softmaxxed
        print('shape', newinput.shape, token.shape)
        newinput = torch.concat([newinput, token], dim=1)
        proposed_ids = torch.concat([proposed_ids, token], dim=1)

    return output, proposed_ids
    # TODO: loop k times
    #   - forward pass on the growing sequence
    #   - take logits for the last position
    #   - convert to a probability distribution (your sampling temperature here)
    #   - sample a token, append it, record the distribution
    ...


# ---------------------------------------------------------------------------
# 3. Target verification: ONE forward pass over the whole proposed block
# ---------------------------------------------------------------------------
@torch.no_grad()
def target_probs(input_ids: torch.Tensor, proposed_ids: torch.Tensor):
    print(input_ids.shape, proposed_ids.shape)
    k=proposed_ids.shape[1]
    cat = torch.concat([input_ids, proposed_ids], dim=1)
    print(cat.shape)
    logits = target(cat).logits
    print(logits[:, -k::, :].shape)
    sm = F.softmax(logits[:, -(k + 1):-1, :], dim=-1)
    sm = sm.squeeze(0)
    print(sm.shape)
    return sm
    """
    Run the target model once on [input_ids ++ proposed_ids] and return
    the target distribution p(x) at each of the k proposal positions
    (i.e. the position that *predicts* each proposed token).

    Returns:
        p: FloatTensor of shape (k, vocab)
    """
    # TODO:
    #   - concatenate input_ids and proposed_ids
    #   - single forward pass
    #   - slice out the k logits that correspond to predicting each proposal
    #   - softmax with the same temperature you used in the draft
    ...


# ---------------------------------------------------------------------------
# 4. Acceptance rule
# ---------------------------------------------------------------------------
# For each proposed token x_i with draft prob q(x_i) and target prob p(x_i):
#   accept with probability min(1, p(x_i) / q(x_i))
# On the first rejection at position i, sample a replacement token from the
# "residual" distribution: normalize( max(0, p - q) ) at that position.
# If ALL k are accepted, you also get one "bonus" token sampled from p at
# position k (the target's next-token distribution after the last proposal).

def accept_or_resample(proposed_ids: torch.Tensor,
                       q: torch.Tensor,   # (k, vocab) draft probs
                       p: torch.Tensor, k:int):  # (k, vocab) target probs
    """
    Returns:
        accepted_ids: LongTensor of shape (n,), where 0 <= n <= k+1
    """
    accepted = torch.empty(0, dtype=torch.long, device=device)
    for i in range(k):
        r = torch.rand((), device=device)
        # this is the ith prediction, and the x_ith token
        x_i = proposed_ids[0, i].item()
        p_xi = p[i, x_i]
        q_xi = q[i, x_i]
        x_i_tensor = torch.tensor([x_i], dtype=torch.long, device=device)
        if p_xi > q_xi:
            accepted = torch.cat([accepted, x_i_tensor], dim=0)
        else:
            if r < (p_xi / q_xi):
                accepted = torch.cat([accepted, x_i_tensor], dim=0)
            else:
                diff = torch.relu(p[i] - q[i])
                res = diff / diff.sum()
                instead_token = torch.multinomial(res, num_samples=1)
                accepted = torch.cat([accepted, instead_token], dim=0)
                break


    return accepted

    # TODO:
    #   for i in range(k):
    #       r = uniform(0,1)
    #       if r < p[i, x_i] / q[i, x_i]:  accept x_i
    #       else: sample replacement from normalize(relu(p[i] - q[i])); STOP
    #   if all accepted: sample bonus token from p[-1] ... wait — you also
    #     need p at the position AFTER the last accepted token; think about
    #     where that comes from in your target forward pass.
    ...


# ---------------------------------------------------------------------------
# 5. Main loop
# ---------------------------------------------------------------------------
def speculative_generate(prompt: str, max_new_tokens: int, k: int = 4):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    generated = 0
    while generated < max_new_tokens:
        q, proposed_ids = draft_tokens(input_ids, k)
        p = target_probs(input_ids, proposed_ids)
        accepted = accept_or_resample(proposed_ids, q,p, k)
        input_ids = torch.cat([input_ids, accepted.unsqueeze(0)], dim=1)
        print(tokenizer.decode(input_ids[0], skip_special_tokens=True))
        generated += 1
        print('generated', generated)
        # TODO:
        #   1. proposed_ids, q = draft_tokens(input_ids, k)
        #   2. p = target_probs(input_ids, proposed_ids)
        #   3. accepted = accept_or_resample(proposed_ids, q, p)
        #   4. append accepted to input_ids; bump `generated`
        #   5. handle EOS
        ...

    print(generated)
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# 6. Sanity checks to convince yourself it's correct
# ---------------------------------------------------------------------------
# - With temperature -> 0 (greedy), accepted tokens should match what the
#   target would have produced on its own. Run both and diff.
# - Track acceptance rate; it should rise as draft/target get more similar.
# - Time per token vs. plain target.generate() — that's your speedup.

if __name__ == "__main__":
    prompt = "The capital of France is"
    print(speculative_generate(prompt, max_new_tokens=64, k=4))
