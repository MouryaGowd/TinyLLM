"""Sample text from a trained checkpoint. Usage: python generate.py --prompt "ROMEO:" """
import argparse

import torch

from model import GPT, GPTConfig, pick_device

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default="ckpt.pt")
p.add_argument("--prompt", default="\n")
p.add_argument("--tokens", type=int, default=500)
p.add_argument("--temperature", type=float, default=0.8)
p.add_argument("--top_k", type=int, default=40)
args = p.parse_args()

device = pick_device()
ck = torch.load(args.ckpt, map_location=device)
chars = ck["chars"]
stoi = {c: i for i, c in enumerate(chars)}
model = GPT(GPTConfig(**ck["cfg"])).to(device)
model.load_state_dict(ck["model"])
model.eval()

idx = torch.tensor([[stoi[c] for c in args.prompt if c in stoi]], dtype=torch.long, device=device)
out = model.generate(idx, args.tokens, temperature=args.temperature, top_k=args.top_k)
print("".join(chars[i] for i in out[0].tolist()))
