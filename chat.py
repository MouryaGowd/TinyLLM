"""Interactive chat with the trained model.

The conversation is formatted as a play script. Your message becomes a speech by YOU_NAME and the
model writes the reply as BOT_NAME, stopping at the blank line that ends a speech in the training data.

Usage: python chat.py [--you ROMEO] [--bot JULIET]
Commands inside the chat: /reset (clear history), /quit
"""
import argparse

import torch
import torch.nn.functional as F

from model import GPT, GPTConfig, pick_device

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default="ckpt.pt")
p.add_argument("--you", default="ROMEO", help="speaker name for your lines")
p.add_argument("--bot", default="JULIET", help="speaker name for the model's lines")
p.add_argument("--temperature", type=float, default=0.5)
p.add_argument("--top_k", type=int, default=40)
p.add_argument("--max_chars", type=int, default=400, help="max length of one reply")
args = p.parse_args()

device = pick_device()
ck = torch.load(args.ckpt, map_location=device)
chars = ck["chars"]
stoi = {c: i for i, c in enumerate(chars)}
model = GPT(GPTConfig(**ck["cfg"])).to(device)
model.load_state_dict(ck["model"])
model.eval()
block_size = model.cfg.block_size


def encode(s):
    # characters the model never saw in training are dropped
    return [stoi[c] for c in s if c in stoi]


@torch.no_grad()
def reply(history):
    """Generate the bot's speech given the script so far. Returns the reply text."""
    ids = encode(history)
    out = ""
    for _ in range(args.max_chars):
        x = torch.tensor([ids[-block_size:]], dtype=torch.long, device=device)
        logits, _ = model(x)
        logits = logits[:, -1, :] / max(args.temperature, 1e-6)
        v, _ = torch.topk(logits, min(args.top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float("-inf")
        nxt = torch.multinomial(F.softmax(logits, dim=-1), 1).item()
        ids.append(nxt)
        out += chars[nxt]
        if out.endswith("\n\n"):  # a blank line ends a speech
            break
    return out.strip()


def new_script():
    return ""


print(f"Chatting as {args.you}. The model replies as {args.bot}. (/reset to clear, /quit to exit)")
print("Note: this is a small Shakespeare model. It imitates the style; it does not understand questions.\n")

script = new_script()
while True:
    try:
        user = input(f"{args.you}> ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break
    if not user:
        continue
    if user == "/quit":
        break
    if user == "/reset":
        script = new_script()
        print("(history cleared)\n")
        continue
    script += f"{args.you}:\n{user}\n\n{args.bot}:\n"
    text = reply(script)
    print(f"{args.bot}> {text}\n")
    script += text + "\n\n"
    script = script[-1500:]  # keep the prompt bounded
