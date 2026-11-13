"""Train the tiny GPT on a text file (character-level). Usage: python train.py [--data input.txt]"""
import argparse
import time

import torch

from model import GPT, GPTConfig, pick_device

p = argparse.ArgumentParser()
p.add_argument("--data", default="input.txt")
p.add_argument("--out", default="ckpt.pt")
p.add_argument("--max_iters", type=int, default=5000)
p.add_argument("--batch_size", type=int, default=32)
p.add_argument("--block_size", type=int, default=256)
p.add_argument("--n_layer", type=int, default=6)
p.add_argument("--n_head", type=int, default=6)
p.add_argument("--n_embd", type=int, default=384)
p.add_argument("--dropout", type=float, default=0.2)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--eval_interval", type=int, default=250)
p.add_argument("--eval_iters", type=int, default=50)
args = p.parse_args()

torch.manual_seed(1337)
device = pick_device()
print(f"device: {device}")

text = open(args.data, encoding="utf-8").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
print(f"{len(text):,} chars, vocab {len(chars)}")


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - args.block_size - 1, (args.batch_size,))
    x = torch.stack([d[i:i + args.block_size] for i in ix])
    y = torch.stack([d[i + 1:i + args.block_size + 1] for i in ix])
    return x.to(device), y.to(device)


cfg = GPTConfig(vocab_size=len(chars), block_size=args.block_size, n_layer=args.n_layer,
                n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout)
model = GPT(cfg).to(device)
print(f"{model.num_params() / 1e6:.2f}M parameters")
opt = torch.optim.AdamW(model.parameters(), lr=args.lr)


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            _, loss = model(*get_batch(split))
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def save():
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "chars": chars}, args.out)


t0 = time.time()
best_val = float("inf")
for it in range(args.max_iters + 1):
    if it % args.eval_interval == 0:
        l = estimate_loss()
        print(f"iter {it:5d} | train {l['train']:.4f} | val {l['val']:.4f} | {time.time() - t0:.0f}s")
        if l["val"] < best_val:
            best_val = l["val"]
            save()
    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

print(f"done. best val loss {best_val:.4f}, saved to {args.out}")
