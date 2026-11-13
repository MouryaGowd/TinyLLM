# TinyLLM: presentation notes

## 30-second pitch

"I built a small language model from scratch in PyTorch, the same kind of transformer that powers GPT,
and trained it on my laptop. It learned to write Shakespeare-style text by repeatedly predicting the next
character. It has 10.76 million parameters and trained in about 27 minutes on an Apple M1 Pro."

## 2-minute walkthrough

**1. The goal.** A language model does one thing: given some text, predict what comes next. ChatGPT is
this idea at huge scale, plus extra training to follow instructions. I built the core idea at small scale.

**2. The data.** Tiny Shakespeare, about 1.1 million characters of plays. The model reads one character
at a time, with 65 distinct characters (letters, punctuation, newline).

**3. The model.** A decoder-only transformer, written by hand in `model.py`:
- 6 layers, 6 attention heads, 384-dimensional embeddings, 256-character context window.
- The key part is **self-attention**: each character looks back at the earlier characters and decides which
  ones matter for predicting the next one. A mask stops it from looking ahead, which would be cheating.

**4. The training.** Show the model 256 characters, ask for the next character, measure the error (loss),
and nudge the weights to reduce it. Repeat 5000 times.
- Loss started at 4.20, which is about what random guessing gives (ln 65 = 4.17).
- Validation loss on text the model never trained on fell to **1.48**.
- I kept the checkpoint with the best validation loss, not the last one.

**5. The result.** Prompt `ROMEO:` produces:

```
ROMEO:
And so much to fill the people's carse, we will not.

JULIET:
And thus the head of strike and misfortune
```

It learned the format (speaker names, colons, line breaks) and archaic vocabulary, but the sentences don't
make sense. That is expected for a model this small trained this briefly.

## Live demo

```bash
cd /Users/mourya/Rubrik/TinyLLM
./.venv/bin/python generate.py --prompt "ROMEO:" --tokens 400
./.venv/bin/python generate.py --prompt "JULIET:" --temperature 0.5   # safer, more repetitive
./.venv/bin/python generate.py --prompt "JULIET:" --temperature 1.2   # more creative, more errors
```

Temperature controls randomness: low gives predictable text, high gives more varied but messier text.

## Likely questions

**Is this ChatGPT?** No. It uses the same architecture family but is about 10 million parameters, versus
billions for modern models. It also only continues text; it was never trained to follow instructions.

**Why doesn't it make sense?** It has learned patterns and style, not meaning. Capability comes from scale
(more parameters, data and compute). This model has very little of each.

**Why character-level?** It keeps the vocabulary tiny (65) and the code simple. Real models use subword
tokenizers (BPE), which are more efficient.

**What is overfitting?** Training loss kept falling (down to 1.13) while validation loss stopped improving
(about 1.48). The model began memorizing the training text. I saved the best-validation checkpoint to
guard against this.

**Did you use a library for the model?** PyTorch provides tensors, layers like Linear and LayerNorm, and
automatic differentiation. The transformer itself (attention, blocks, generation) is written by hand.

**How would you improve it?** Subword tokenization, learning-rate scheduling, a larger and more varied
dataset, a bigger model on a stronger GPU, and fine-tuning for instruction following.

## Technical questions

**Where are the 10.76M parameters?** Each of the 6 blocks has about 1.77M (attention 590K, MLP 1.18M). The
embeddings are small (25K token + 98K position) because the vocabulary is only 65 characters.

**What does the loss of 1.48 mean?** About 2.14 bits per character, or a perplexity of about 4.4: the model is
as uncertain as picking among 4 to 5 equally likely characters. Random guessing over 65 characters is 4.17 nats.

**How does the causal mask work?** Before the softmax, attention scores for future positions are set to
negative infinity, so each position only sees itself and earlier characters.

**Why is it slow at generating?** There is no KV cache, so every new character recomputes attention over the
whole 256-character window.

**What does temperature do mathematically?** The logits are divided by the temperature before the softmax.
Below 1 this sharpens the distribution; above 1 it flattens it. Top-k then removes all but the 40 most likely
characters.

**Why does it overfit?** 5000 steps of 8,192 tokens is about 41M tokens, around 40 passes over a 1M-character
training set. A 10M-parameter model can start memorizing that.

**What is weight tying?** The output layer reuses the token embedding matrix, so the same weights embed input
characters and score output characters.

**Weakest point of the evaluation?** Validation loss picks the checkpoint and is also the number reported, so it
is slightly optimistic. A separate test set would fix that.

**How does the chat work?** Your message is written into a play script as a speech, the model continues as the
other character, and generation stops at the blank line that ends a speech. Its memory is limited to the
256-character context window.

## Numbers to remember

| Item | Value |
|------|-------|
| Parameters | 10.76M |
| Architecture | 6 layers, 6 heads, 384 dim, 256 context |
| Data | ~1.1M characters, vocab 65 |
| Training | 5000 iterations, ~27 min, M1 Pro (MPS) |
| Loss | 4.20 at start, best validation 1.48 |
