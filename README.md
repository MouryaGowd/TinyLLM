# TinyLLM: a GPT built from scratch in PyTorch

A small decoder-only transformer language model (GPT-style), written from scratch and trained
character-by-character on the Tiny Shakespeare dataset. It runs on a laptop (Apple Silicon GPU via MPS).

## Project structure

| File          | Purpose                                                           |
|---------------|-------------------------------------------------------------------|
| `model.py`    | The GPT architecture (attention, MLP, transformer blocks, sampling) |
| `train.py`    | Training loop with validation, gradient clipping and checkpointing |
| `generate.py` | Load a checkpoint and generate text from a prompt                 |
| `chat.py`     | Interactive terminal chat (you and the model trade lines of a play script) |
| `evaluate.py` | Full-validation loss, bigram baseline and generated-word statistics |
| `prompts/`    | 467 test prompts (famous lines, old-style questions, test lines, modern lines) |
| `run_prompts.py` | Run a whole prompt file through the chat model and save the replies |
| `ckpt.pt`     | Trained weights (best validation checkpoint, 44.6 MB)             |
| `*.bat`, `run.sh` | One-click launchers for Windows and macOS/Linux              |
| `paper/`      | IEEE-format paper (PDF/HTML/LaTeX), architecture diagrams, figure script |
| `input.txt`   | Training text (Tiny Shakespeare, ~1.1M characters)                |

## Quick start

### Windows (no command line needed)
1. Install Python 3.10+ from https://www.python.org/downloads/ and tick **Add python.exe to PATH**.
2. Download or clone this repository (the trained model `ckpt.pt` is included).
3. Double-click **`run_chat.bat`** to chat, or **`run_generate.bat`** to generate text.

The first launch creates a virtual environment and installs PyTorch automatically (a few hundred MB, several
minutes). If an NVIDIA GPU is detected the CUDA build is installed; otherwise the CPU build is used, which is fast
enough for chatting and generating. **`train.bat`** retrains the model (slow on CPU; try `train.bat --max_iters 300`).
Status: the Windows launchers were written on macOS and have not been run on Windows by the author, so please
report any problem.

### macOS / Linux
```bash
bash run.sh            # chat (default)
bash run.sh generate --prompt "ROMEO:"
bash run.sh train
bash run.sh evaluate
```

### Manual setup
```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
# only needed for retraining:
curl -L -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```
On Windows use `.venv\Scripts\python.exe` instead of `./.venv/bin/python` in the commands below.

## Paper

`paper/paper.pdf` is a 4-page IEEE-format paper describing the model and its evaluation (`paper/paper.tex` is an
IEEEtran version for Overleaf, not compiled by the author). Figures are regenerated with
`python paper/make_figures.py` (needs `matplotlib`), and `python evaluate.py` reproduces the numbers in
`paper/eval_results.txt`.

## Usage

```bash
# Train (saves the best-validation checkpoint to ckpt.pt)
./.venv/bin/python train.py

# Quick smoke test
./.venv/bin/python train.py --max_iters 200 --eval_interval 50

# Generate text
./.venv/bin/python generate.py --prompt "ROMEO:" --tokens 500 --temperature 0.8

# Interactive chat (you play ROMEO, the model replies as JULIET)
./.venv/bin/python chat.py
./.venv/bin/python chat.py --you HAMLET --bot OPHELIA --temperature 0.7
```

In `chat.py` your message is written into a play script as a speech by your character, and the model
continues the script as the other character, stopping at the blank line that ends a speech. Type `/reset`
to clear the history and `/quit` to exit. The model imitates Shakespeare's style and does not understand
questions.

Useful `train.py` flags: `--data`, `--max_iters`, `--batch_size`, `--block_size`, `--n_layer`,
`--n_head`, `--n_embd`, `--dropout`, `--lr`. Train on any text file with `--data yourfile.txt`.

## How it works

### 1. Tokenization
Character-level: every distinct character in the corpus becomes one token (65 for Tiny Shakespeare).
This keeps the vocabulary tiny and the code simple, at the cost of longer sequences than subword tokenizers.

### 2. Architecture (`model.py`)
- **Embeddings:** token embedding + learned positional embedding.
- **Transformer block (x6):** pre-LayerNorm, multi-head causal self-attention, then a 4x-wide GELU MLP,
  each wrapped in a residual connection.
- **Causal self-attention:** queries, keys and values come from one fused linear layer. Scores are
  scaled by `1/sqrt(head_dim)` and a lower-triangular mask stops a position from seeing future tokens.
- **Output head:** final LayerNorm, then a linear layer to vocabulary logits. Its weights are tied to the
  token embedding, which reduces parameters.
- **Initialization:** normal(0, 0.02) for linear and embedding weights.

Default configuration (about 10M parameters):

| Hyperparameter | Value |
|----------------|-------|
| Layers         | 6     |
| Heads          | 6     |
| Embedding dim  | 384   |
| Context length | 256   |
| Dropout        | 0.2   |

### 3. Training (`train.py`)
- Objective: next-character prediction with cross-entropy loss.
- Optimizer: AdamW, learning rate 3e-4, gradient clipping at 1.0.
- Data: 90% train / 10% validation split. Random windows of `block_size` characters are sampled each step.
- The checkpoint is saved only when validation loss improves, so `ckpt.pt` is the best model seen.
- Device: MPS (Apple GPU) if available, otherwise CPU.

### 4. Generation (`generate.py`)
Autoregressive sampling: feed the prompt, sample the next character from the softmax over logits,
append it, repeat. Two knobs control output quality:
- **temperature:** lower is more predictable, higher is more random.
- **top-k:** sample only from the k most likely characters.

## Technical details

### Parameter count (10,761,600)

| Component | Parameters |
|---|---|
| Token embedding (shared with output head) | 24,960 |
| Position embedding | 98,304 |
| Per block: QKV 442,368 + attention proj 147,456 + MLP 1,181,568 + 2 LayerNorms 1,536 | 1,772,928 |
| 6 blocks | 10,637,568 |
| Final LayerNorm | 768 |

About 99% of the parameters are in the transformer blocks, because the 65-character vocabulary makes the
embedding tiny.

### Attention
One fused linear layer produces Q, K, V (384 to 1152 dimensions), split into 6 heads of 64. Scores are
`QK^T / sqrt(64)`; a lower-triangular mask sets future positions to `-inf` before the softmax, so position
t only attends to positions up to t. The `T x T` score matrix is built explicitly (O(T^2) memory), which is
clearer than a fused kernel but slower.

### Training details
- Targets are the inputs shifted by one character, so one 256-character window yields 256 training examples.
- Batch of 32 windows = 8,192 tokens per step; 5000 steps = about 41M tokens seen.
- The train split is about 1M characters, so training made roughly 40 passes over the data. That explains
  the overfitting: train loss kept falling to 1.13 while validation loss bottomed out near 1.48.
- Constant learning rate 3e-4 with AdamW (default weight decay 0.01). There is no warmup and no decay schedule.
- Dropout 0.2 is applied to the embeddings, the attention weights, the attention output and the MLP output.
- The split is contiguous (first 90% / last 10%), not shuffled.
- fp32 on the MPS backend, about 0.33 s per step.

### Interpreting the loss
| Quantity | Value |
|---|---|
| Uniform-guess baseline | ln 65 = 4.17 nats/char |
| Initial loss | 4.20 |
| Best validation loss | 1.48 nats/char |
| In bits per character | 1.48 / ln 2 = about 2.14 |
| Perplexity | e^1.48 = about 4.4 |

A perplexity of about 4.4 means the model is on average as uncertain as choosing among 4 to 5 equally
likely characters, down from 65 at initialization.

Caveat: the checkpoint is chosen by validation loss and the same number is reported, so 1.48 is slightly
optimistic. A clean evaluation would use a separate held-out test set.

### Inference
Sampling crops the context to the last 256 characters, runs a forward pass, divides the last-position logits
by the temperature, keeps the top-k (k=40), applies softmax and draws one character with `multinomial`.
There is no KV cache, so each new character recomputes attention over the whole window.

`chat.py` is a framing trick on top of the same sampler: the user's message is appended to a script as
`NAME:\n<message>\n\nBOT:\n`, and generation stops at the first blank line (`\n\n`), which ends a speech in
the training data. Because the context window is 256 characters, the model forgets earlier turns quickly.

### Comparison with large models
| | This project | Modern LLM |
|---|---|---|
| Architecture | Decoder-only transformer | Same family, with refinements (RoPE, GQA, RMSNorm) |
| Parameters | 10.76M | Billions or more |
| Tokenizer | 65 characters | Subword BPE, tens of thousands of tokens |
| Training stages | Pretraining only | Pretraining, then instruction tuning and RLHF |

## Results

Measured on an Apple M1 Pro (MPS), 5000 iterations, batch size 32:

- Parameters: 10.76M
- Training time: about 27 minutes (1640s)
- Best validation loss: **1.48** (iteration 4000, the saved checkpoint)
- Final train loss: 1.13, final validation loss: 1.49 (iteration 5000). The train-validation gap at the end shows some overfitting, which the best-checkpoint selection guards against.

Validation loss over training:

| Iteration | Train loss | Val loss |
|-----------|-----------|----------|
| 0         | 4.20      | 4.21     |
| 1000      | 1.66      | 1.83     |
| 2000      | 1.39      | 1.61     |
| 3000      | 1.27      | 1.54     |
| 4000      | 1.20      | **1.48** |
| 5000      | 1.13      | 1.49     |

Sample (`generate.py --prompt "ROMEO:" --tokens 400 --temperature 0.8 --top_k 40`):

```
ROMEO:
And so much to fill the people's carse, we will not.

JULIET:
And thus the head of strike and misfortune
I could be ink in the general fury of our grievous made
In mercy; the people that such a word.
...
```

The output has the play's structure (speaker names, line breaks, colons) and vocabulary, but the sentences
are not coherent. That is expected for a 10M-parameter character model trained for about 27 minutes.

## Limitations

- Character-level modelling with a small model and small dataset: output imitates the style and
  structure of the training text (names, line breaks, archaic phrasing) but is not coherent over long spans.
- It is not instruction-tuned and cannot answer questions. It only continues text.
- The model memorizes and imitates its single training corpus, so it knows nothing outside it.

## Possible extensions

- Subword tokenization (BPE) for better efficiency.
- Learning-rate warmup and cosine decay.
- Training on a larger or custom corpus.
- Fine-tuning a pretrained model for instruction following.

## References

- Vaswani et al., *Attention Is All You Need* (2017)
- Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2, 2019)
- Andrej Karpathy, nanoGPT and "Let's build GPT from scratch" (the main inspiration for this design)
