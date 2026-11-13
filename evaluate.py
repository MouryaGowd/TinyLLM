"""Evaluate the trained model for the paper. Usage: python evaluate.py [--ckpt ckpt.pt] [--data input.txt]

Reports:
  1. loss on the whole validation split (deterministic, non-overlapping 256-char windows)
  2. a character-bigram baseline (add-one smoothing) on the same split
  3. share of generated words that occur in the training text, per sampling temperature
"""
import argparse
import math
import re

import torch

from model import GPT, GPTConfig, pick_device

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default="ckpt.pt")
p.add_argument("--data", default="input.txt")
p.add_argument("--samples", type=int, default=8, help="sequences per temperature")
p.add_argument("--length", type=int, default=1000, help="characters per sequence")
args = p.parse_args()

device = pick_device()
ck = torch.load(args.ckpt, map_location=device)
chars = ck["chars"]
stoi = {c: i for i, c in enumerate(chars)}
V = len(chars)
model = GPT(GPTConfig(**ck["cfg"])).to(device)
model.load_state_dict(ck["model"])
model.eval()
T = model.cfg.block_size

text = open(args.data, encoding="utf-8").read()
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(0.9 * len(data))
train_ids, val_ids = data[:n], data[n:]
print(f"device {device} | train {len(train_ids):,} chars | val {len(val_ids):,} chars | vocab {V}")

# 1. full validation loss: tile the validation split into non-overlapping windows of T characters
usable = (len(val_ids) - 1) // T * T
x = val_ids[:usable].view(-1, T)
y = val_ids[1:usable + 1].view(-1, T)
total, count = 0.0, 0
with torch.no_grad():
    for i in range(0, len(x), 32):
        _, loss = model(x[i:i + 32].to(device), y[i:i + 32].to(device))
        total += loss.item() * x[i:i + 32].numel()
        count += x[i:i + 32].numel()
val_loss = total / count
print(f"model full-val loss: {val_loss:.4f} nats/char = {val_loss / math.log(2):.3f} bits/char, "
      f"perplexity {math.exp(val_loss):.2f}  ({count:,} chars scored)")

# 2. bigram baseline with add-one smoothing, trained on the same train split
counts = torch.ones(V, V)
counts.index_put_((train_ids[:-1], train_ids[1:]), torch.ones(len(train_ids) - 1), accumulate=True)
probs = counts / counts.sum(1, keepdim=True)
bg = -torch.log(probs[val_ids[:-1], val_ids[1:]]).mean().item()
print(f"bigram baseline val loss: {bg:.4f} nats/char = {bg / math.log(2):.3f} bits/char, "
      f"perplexity {math.exp(bg):.2f}")
print(f"uniform baseline: {math.log(V):.4f} nats/char = {math.log2(V):.3f} bits/char")

# 3. generated-word validity per temperature (top-k 40, as in generate.py/chat.py)
word = re.compile(r"[A-Za-z']+")
train_words = set(w.lower() for w in word.findall(text[:n]))
val_words = [w.lower() for w in word.findall(text[n:])]
ctrl = sum(w in train_words for w in val_words) / len(val_words)
print(f"control: {ctrl:.1%} of real held-out words occur in the training split "
      f"({len(train_words):,} distinct training words)")

torch.manual_seed(0)
start = torch.tensor([[stoi["\n"]]] * args.samples, device=device)
for temp in (0.5, 0.8, 1.0, 1.2):
    out = model.generate(start.clone(), args.length, temperature=temp, top_k=40)
    gen = "".join(chars[i] for row in out.tolist() for i in row[1:] + [stoi["\n"]])
    words = [w.lower() for w in word.findall(gen)]
    valid = sum(w in train_words for w in words) / len(words)
    distinct = len(set(words)) / len(words)
    print(f"temperature {temp}: {len(words):,} words, {valid:.1%} in training vocabulary, "
          f"{distinct:.1%} distinct")
