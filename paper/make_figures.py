"""Generate the paper figures: architecture diagram and training curves (read from ../train.log).

Usage: python paper/make_figures.py   (needs matplotlib)
Outputs: paper/architecture.png/.pdf, paper/loss_curve.png/.pdf
"""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
LOG = HERE.parent / "train.log"

INK = "#1f2933"
BLUE, BLUE_BG = "#2563eb", "#dbeafe"
GREEN, GREEN_BG = "#15803d", "#dcfce7"
ORANGE, ORANGE_BG = "#c2410c", "#ffedd5"
PURPLE, PURPLE_BG = "#7e22ce", "#f3e8ff"
GREY, GREY_BG = "#475569", "#f1f5f9"

plt.rcParams["font.family"] = "DejaVu Sans"


def box(ax, x, y, w, h, text, fc, ec, fs=9, bold=False, lw=1.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal", linespacing=1.3)


def arrow(ax, p0, p1, color=INK, lw=1.3, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=11, color=color,
                                 lw=lw, connectionstyle=f"arc3,rad={rad}", linestyle=ls))


def architecture():
    fig, ax = plt.subplots(figsize=(7.2, 9.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13.2)
    ax.axis("off")

    cx = 3.7   # centre line of the main stack
    w = 4.2
    rx = 1.3   # x of the residual (skip) lines
    tx = 6.75  # x of the weight-tying line
    px, pw = 7.45, 2.5  # side panels

    # input / output
    box(ax, cx - w / 2, 0.15, w, 0.7, "Input characters  (B, T)   T \u2264 256", GREY_BG, GREY, 9)
    box(ax, cx - w / 2, 12.2, w, 0.7, "Next-character logits  (B, T, 65)", GREY_BG, GREY, 9)

    # embeddings: position on the left, token on the right (token is the tied matrix)
    box(ax, cx - 2.2, 1.35, 1.95, 0.85, "Position\nembedding\n256\u2192384", BLUE_BG, BLUE, 8.5)
    box(ax, cx + 0.25, 1.35, 1.95, 0.85, "Token\nembedding\n65\u2192384", BLUE_BG, BLUE, 8.5)
    ax.text(cx, 2.45, "+", ha="center", va="center", fontsize=15, color=INK, fontweight="bold")
    box(ax, cx - 1.1, 2.85, 2.2, 0.5, "Dropout 0.2", GREY_BG, GREY, 8.5)
    arrow(ax, (cx, 0.85), (cx - 1.2, 1.35))
    arrow(ax, (cx, 0.85), (cx + 1.2, 1.35))
    arrow(ax, (cx - 1.2, 2.2), (cx - 0.15, 2.3))
    arrow(ax, (cx + 1.2, 2.2), (cx + 0.15, 2.3))
    arrow(ax, (cx, 2.62), (cx, 2.85))

    # transformer block (x6)
    bx, by, bw, bh = cx - 2.65, 3.75, 5.3, 6.9
    ax.add_patch(FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc="#fafafa", ec=INK, lw=1.8, linestyle=(0, (5, 3))))
    ax.text(bx + bw - 0.15, by + bh - 0.28, "Transformer block  \u00d7 6", ha="right", va="center",
            fontsize=9, fontweight="bold", color=INK)
    arrow(ax, (cx, 3.35), (cx, 4.3))

    # attention sub-layer
    box(ax, cx - 1.5, 4.3, 3.0, 0.5, "LayerNorm", GREY_BG, GREY, 9)
    box(ax, cx - 1.9, 5.05, 3.8, 1.9,
        "Causal multi-head\nself-attention\n6 heads \u00d7 64 dim\nQ,K,V from one linear layer\nmask: position t sees \u2264 t",
        GREEN_BG, GREEN, 8.5)
    arrow(ax, (cx, 4.8), (cx, 5.05))
    arrow(ax, (cx, 6.95), (cx, 7.07))
    ax.text(cx, 7.2, "\u2295", ha="center", va="center", fontsize=17, color=INK, fontweight="bold")

    # residual 1: branches off the block input, rejoins at the first (+)
    ax.plot([cx, rx], [3.95, 3.95], color=ORANGE, lw=1.3, ls="--")
    ax.plot([rx, rx], [3.95, 7.2], color=ORANGE, lw=1.3, ls="--")
    arrow(ax, (rx, 7.2), (cx - 0.22, 7.2), color=ORANGE, ls="--")
    ax.text(rx + 0.17, 5.6, "residual", rotation=90, ha="center", va="center", fontsize=8, color=ORANGE)

    # MLP sub-layer
    arrow(ax, (cx, 7.4), (cx, 7.65))
    box(ax, cx - 1.5, 7.65, 3.0, 0.5, "LayerNorm", GREY_BG, GREY, 9)
    box(ax, cx - 1.9, 8.4, 3.8, 1.35, "MLP\n384 \u2192 1536 (GELU) \u2192 384", PURPLE_BG, PURPLE, 8.5)
    arrow(ax, (cx, 8.15), (cx, 8.4))
    arrow(ax, (cx, 9.75), (cx, 9.87))
    ax.text(cx, 10.05, "\u2295", ha="center", va="center", fontsize=17, color=INK, fontweight="bold")

    # residual 2: branches off the first (+) output, rejoins at the second (+)
    ax.plot([cx, rx + 0.3], [7.52, 7.52], color=ORANGE, lw=1.3, ls="--")
    ax.plot([rx + 0.3, rx + 0.3], [7.52, 10.05], color=ORANGE, lw=1.3, ls="--")
    arrow(ax, (rx + 0.3, 10.05), (cx - 0.22, 10.05), color=ORANGE, ls="--")

    # head
    arrow(ax, (cx, 10.27), (cx, 10.95))
    box(ax, cx - 1.5, 10.95, 3.0, 0.5, "Final LayerNorm", GREY_BG, GREY, 9)
    box(ax, cx - 1.5, 11.6, 3.0, 0.5, "Linear 384\u219265", BLUE_BG, BLUE, 9)
    arrow(ax, (cx, 11.45), (cx, 11.6))
    arrow(ax, (cx, 12.1), (cx, 12.2))

    # weight tying: output Linear <-> TOKEN embedding (right-hand embedding box)
    ax.plot([cx + 1.5, tx], [11.85, 11.85], color=BLUE, lw=1.3, ls=(0, (2, 2)))
    ax.plot([tx, tx], [11.85, 1.78], color=BLUE, lw=1.3, ls=(0, (2, 2)))
    arrow(ax, (tx, 1.78), (cx + 2.2, 1.78), color=BLUE, ls=(0, (2, 2)))
    ax.text(tx + 0.26, 6.9, "weight tying: output layer shares the token-embedding matrix (65\u00d7384)",
            rotation=90, ha="center", va="center", fontsize=8, color=BLUE)

    # side panel: sampling
    ax.add_patch(FancyBboxPatch((px, 9.0), pw, 3.1, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=ORANGE_BG, ec=ORANGE, lw=1.4))
    ax.text(px + pw / 2, 11.78, "Sampling", ha="center", va="center", fontsize=10, fontweight="bold",
            color=INK)
    ax.text(px + 0.15, 11.4, "1. last-position logits\n2. \u00f7 temperature\n3. keep top-k (k=40)\n"
            "4. softmax\n5. draw 1 character\n6. append, repeat\n\n(chat: stop at blank line)",
            ha="left", va="top", fontsize=8.2, color=INK, linespacing=1.4)
    arrow(ax, (cx + w / 2, 12.55), (px, 11.95), color=ORANGE)

    # side panel: configuration
    ax.add_patch(FancyBboxPatch((px, 5.0), pw, 2.8, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=GREY_BG, ec=GREY, lw=1.2))
    ax.text(px + pw / 2, 7.48, "Configuration", ha="center", va="center", fontsize=10,
            fontweight="bold", color=INK)
    ax.text(px + 0.15, 7.12, "vocabulary   65 chars\ncontext      256\nlayers       6\nheads        6\n"
            "embedding    384\ndropout      0.2\nparameters   10.76 M",
            ha="left", va="top", fontsize=8.2, color=INK, family="DejaVu Sans Mono",
            linespacing=1.45)

    fig.tight_layout(pad=0.3)
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"architecture.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def architecture_wide():
    """Horizontal pipeline version sized for a two-column paper (full page width)."""
    fig, ax = plt.subplots(figsize=(7.16, 2.9))
    ax.set_xlim(0, 20.7)
    ax.set_ylim(0, 7.5)
    ax.axis("off")
    yc, h = 3.6, 1.5
    top, bot = yc + h / 2, yc - h / 2
    fs = 6.3

    def blk(x, w, text, fc, ec):
        box(ax, x, bot, w, h, text, fc, ec, fs)

    def link(x0, x1, **kw):
        arrow(ax, (x0, yc), (x1, yc), lw=1.0, **kw)

    def plus(x):
        ax.text(x, yc, "⊕", ha="center", va="center", fontsize=11, fontweight="bold", color=INK)

    blk(0.1, 1.4, "Input\nchars\n(B, T)", GREY_BG, GREY)
    blk(2.0, 2.3, "Token + position\nembedding\n→ 384, dropout", BLUE_BG, BLUE)
    link(1.5, 2.0)

    # transformer block (x6)
    ax.add_patch(FancyBboxPatch((4.8, 1.0), 9.0, 5.0, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc="#fafafa", ec=INK, lw=1.4, linestyle=(0, (5, 3))))
    ax.text(9.3, 5.6, "Transformer block × 6", ha="center", va="center", fontsize=7.5,
            fontweight="bold", color=INK)
    blk(5.35, 1.0, "Layer\nNorm", GREY_BG, GREY)
    blk(6.65, 2.6, "Causal multi-head\nself-attention\n6 heads × 64", GREEN_BG, GREEN)
    plus(9.6)
    blk(10.15, 1.0, "Layer\nNorm", GREY_BG, GREY)
    blk(11.45, 1.6, "MLP\n384→1536\n→384", PURPLE_BG, PURPLE)
    plus(13.4)
    link(4.3, 5.35)
    link(6.35, 6.65)
    link(9.25, 9.4)
    link(9.8, 10.15)
    link(11.15, 11.45)
    link(13.05, 13.2)

    # residual 1: block input -> first (+);  residual 2: first (+) -> second (+)
    ax.plot([5.1, 5.1, 9.6], [yc, 2.1, 2.1], color=ORANGE, lw=1.0, ls="--")
    arrow(ax, (9.6, 2.1), (9.6, yc - 0.28), color=ORANGE, ls="--", lw=1.0)
    ax.plot([9.95, 9.95, 13.4], [yc, 1.6, 1.6], color=ORANGE, lw=1.0, ls="--")
    arrow(ax, (13.4, 1.6), (13.4, yc - 0.28), color=ORANGE, ls="--", lw=1.0)
    ax.text(7.4, 1.3, "residual connections", fontsize=6, color=ORANGE, ha="center", va="center")

    blk(14.4, 1.6, "Final\nLayerNorm", GREY_BG, GREY)
    blk(16.4, 1.4, "Linear\n384→65\n(logits)", BLUE_BG, BLUE)
    blk(18.2, 2.3, "Sampling\n÷ temp, top-k,\nsoftmax, draw", ORANGE_BG, ORANGE)
    link(13.6, 14.4)
    link(16.0, 16.4)
    link(17.8, 18.2)

    # weight tying: output Linear <-> token embedding
    ax.plot([17.1, 17.1, 3.15], [top, 6.9, 6.9], color=BLUE, lw=1.0, ls=(0, (2, 2)))
    arrow(ax, (3.15, 6.9), (3.15, top), color=BLUE, ls=(0, (2, 2)), lw=1.0)
    ax.text(10.1, 7.2, "weight tying (shared 65×384 matrix)", fontsize=6.5, color=BLUE,
            ha="center", va="center")
    # feedback loop
    ax.plot([19.35, 19.35, 0.8], [bot, 0.45, 0.45], color=GREY, lw=1.0)
    arrow(ax, (0.8, 0.45), (0.8, bot), color=GREY, lw=1.0)
    ax.text(10.1, 0.2, "append sampled character to the context and repeat", fontsize=6.5,
            color=GREY, ha="center", va="center")

    fig.tight_layout(pad=0.2)
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"architecture_wide.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def loss_curve():
    rows = []
    for line in LOG.read_text().splitlines():
        m = re.match(r"iter\s+(\d+) \| train ([\d.]+) \| val ([\d.]+)", line)
        if m:
            rows.append((int(m[1]), float(m[2]), float(m[3])))
    if not rows:
        raise SystemExit(f"no 'iter ...' lines found in {LOG}")
    it, tr, va = zip(*rows)
    best_i = min(range(len(va)), key=va.__getitem__)

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.plot(it, tr, color=BLUE, lw=1.6, marker="o", ms=2.5, label="train")
    ax.plot(it, va, color=ORANGE, lw=1.6, marker="s", ms=2.5, label="validation")
    ax.axhline(4.17, color=GREY, lw=0.9, ls=":")
    ax.text(it[-1], 4.17 + 0.07, "uniform guess (ln 65)", ha="right", va="bottom", fontsize=6.5,
            color=GREY)
    ax.annotate(f"best val {va[best_i]:.2f}\n(iter {it[best_i]})", xy=(it[best_i], va[best_i]),
                xytext=(it[best_i] - 2200, va[best_i] + 0.85), fontsize=6.5, color=INK,
                arrowprops=dict(arrowstyle="->", lw=0.8, color=INK))
    ax.set_xlabel("training iteration", fontsize=8)
    ax.set_ylabel("cross-entropy (nats/char)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=7, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.82))
    ax.set_ylim(1.0, 4.6)
    fig.tight_layout(pad=0.4)
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"loss_curve.{ext}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    architecture()
    architecture_wide()
    loss_curve()
    print("wrote architecture, architecture_wide and loss_curve (.png/.pdf) in", HERE)
