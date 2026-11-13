"""Run every line of a prompt file through the chat model and save the replies.

Each prompt is answered independently (fresh context), formatted as a play script exactly like chat.py.

Usage:
  python run_prompts.py prompts/test_lines.txt
  python run_prompts.py prompts/questions.txt --limit 20 --temperature 0.5 --out replies.txt
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from model import GPT, GPTConfig, pick_device

p = argparse.ArgumentParser()
p.add_argument("file", help="text file with one prompt per line")
p.add_argument("--ckpt", default="ckpt.pt")
p.add_argument("--you", default="ROMEO")
p.add_argument("--bot", default="JULIET")
p.add_argument("--temperature", type=float, default=0.5)
p.add_argument("--top_k", type=int, default=40)
p.add_argument("--max_chars", type=int, default=300)
p.add_argument("--limit", type=int, default=0, help="only run the first N prompts (0 = all)")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default="", help="write 'prompt -> reply' pairs to this file")
args = p.parse_args()

device = pick_device()
ck = torch.load(args.ckpt, map_location=device)
chars = ck["chars"]
stoi = {c: i for i, c in enumerate(chars)}
model = GPT(GPTConfig(**ck["cfg"])).to(device)
model.load_state_dict(ck["model"])
model.eval()
block = model.cfg.block_size
torch.manual_seed(args.seed)


@torch.no_grad()
def reply(prompt):
    ids = [stoi[c] for c in f"{args.you}:\n{prompt}\n\n{args.bot}:\n" if c in stoi]
    out = ""
    for _ in range(args.max_chars):
        x = torch.tensor([ids[-block:]], dtype=torch.long, device=device)
        logits, _ = model(x)
        logits = logits[:, -1, :] / max(args.temperature, 1e-6)
        v, _ = torch.topk(logits, min(args.top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float("-inf")
        nxt = torch.multinomial(F.softmax(logits, dim=-1), 1).item()
        ids.append(nxt)
        out += chars[nxt]
        if out.endswith("\n\n"):
            break
    return out.strip()


prompts = [l.strip() for l in Path(args.file).read_text(encoding="utf-8").splitlines() if l.strip()]
if args.limit:
    prompts = prompts[:args.limit]

lines = []
for i, prompt in enumerate(prompts, 1):
    r = reply(prompt)
    block_text = f"[{i}/{len(prompts)}] {prompt}\n  -> {r.replace(chr(10), chr(10) + '     ')}\n"
    print(block_text)
    lines.append(block_text)

if args.out:
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"saved {len(lines)} replies to {args.out}")
