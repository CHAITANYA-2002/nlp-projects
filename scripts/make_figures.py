"""
Regenerate every figure in docs/assets/.

    python scripts/make_figures.py

Idempotent and seeded: running it twice produces byte-identical output. Every
number plotted is measured from the code in this repository at run time --
parameter counts come from instantiated modules, IntNet widths from the tensors
IntNet actually returns, the transition matrix from a real CRF, and the Viterbi
comparison from brute-force enumeration of every possible labelling. Nothing
here is transcribed from the paper or from a remembered run.

Each figure is written twice, once for each GitHub theme, so the README can
serve the right one with <picture>.
"""
from __future__ import annotations

import io
import itertools
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.IntNet import IntNet          # noqa: E402
from model.crf import CRF, START_TAG, STOP_TAG   # noqa: E402
from model.seqlabel import SeqLabel      # noqa: E402
from utils.data import Data              # noqa: E402

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "docs", "assets")
SAMPLE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "sample_data", "eng.bioes.train")

THEMES = {
    # GitHub's two canvas colours, so a figure sits flush against the page.
    "light": dict(bg="#ffffff", fg="#1f2328", muted="#59636e", grid="#d1d9e0"),
    "dark": dict(bg="#0d1117", fg="#e6edf3", muted="#9198a1", grid="#3d444d"),
}

# One accent ramp, readable on both canvases.
ACCENT = ["#2f81f7", "#e3742f", "#3fb950", "#bc8cff", "#f778ba", "#d29922"]


def style(theme):
    c = THEMES[theme]
    plt.rcParams.update({
        "figure.facecolor": c["bg"],
        "axes.facecolor": c["bg"],
        "savefig.facecolor": c["bg"],
        "text.color": c["fg"],
        "axes.labelcolor": c["fg"],
        "axes.edgecolor": c["grid"],
        "xtick.color": c["muted"],
        "ytick.color": c["muted"],
        "grid.color": c["grid"],
        "axes.titlecolor": c["fg"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "figure.dpi": 160,
    })
    return c


def save(fig, name, theme):
    os.makedirs(ASSETS, exist_ok=True)
    path = os.path.join(ASSETS, "%s-%s.png" % (name, theme))
    fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", os.path.relpath(path, os.path.dirname(ASSETS)))


# --------------------------------------------------------------------------
# 1. IntNet's width under dense connections, measured against the old formula
# --------------------------------------------------------------------------

def intnet_width_measured(char_emb_dim, char_hidden_dim, intNet_layer, kernel_type=2):
    """Ask a real IntNet how wide its output is."""
    torch.manual_seed(0)
    net = IntNet(40, char_emb_dim, intNet_layer, kernel_type, 0.0, char_hidden_dim, False)
    net.eval()
    with torch.no_grad():
        produced = net.get_last_hiddens(torch.randint(0, 40, (2, 9)), [9, 9]).shape[1]
    assert produced == net.output_dim, "declared width disagrees with the tensor"
    return produced


def old_formula(char_emb_dim, char_hidden_dim, intNet_layer, kernel_type=2):
    """The width SentenceRep and WordSequence used to assume."""
    blocks = (intNet_layer - 1) // 2
    return blocks * char_hidden_dim * kernel_type + char_hidden_dim * 2 * kernel_type


def figure_intnet_width(theme):
    c = style(theme)
    configs = [
        ("32 / 16\n(shipped)", 32, 16),
        ("8 / 8", 8, 8),
        ("64 / 16", 64, 16),
        ("32 / 32", 32, 32),
    ]
    layers = [3, 5, 7, 9, 11]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    for i, (label, emb, hid) in enumerate(configs):
        widths = [intnet_width_measured(emb, hid, n) for n in layers]
        ax.plot(layers, widths, marker="o", color=ACCENT[i],
                label="char_emb %d, char_hidden %d%s"
                      % (emb, hid, " (shipped)" if (emb, hid) == (32, 16) else ""))
    ax.set_xlabel("intNet_layer")
    ax.set_ylabel("output width")
    ax.set_title("Width grows with depth and with char_emb_dim", fontsize=11)
    ax.set_xticks(layers)
    ax.grid(alpha=0.35, linewidth=0.7)
    ax.legend(fontsize=8, frameon=False, labelcolor=c["fg"])

    ax = axes[1]
    x = np.arange(len(configs))
    measured = [intnet_width_measured(emb, hid, 7) for _, emb, hid in configs]
    assumed = [old_formula(emb, hid, 7) for _, emb, hid in configs]
    ax.bar(x - 0.2, measured, 0.4, color=ACCENT[0], label="what IntNet returns")
    ax.bar(x + 0.2, assumed, 0.4, color=ACCENT[1], label="what the old formula assumed")
    for xi, (m, a) in enumerate(zip(measured, assumed)):
        if m != a:
            ax.annotate("mismatch", (xi, max(m, a)), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=8,
                        color=ACCENT[1], fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _, _ in configs], fontsize=9)
    ax.set_xlabel("char_emb_dim / char_hidden_dim")
    ax.set_ylabel("width at intNet_layer = 7")
    ax.set_title("Agreeing only when char_emb_dim = 2 x char_hidden_dim", fontsize=11)
    ax.grid(axis="y", alpha=0.35, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, frameon=False, labelcolor=c["fg"])

    fig.suptitle("IntNet output width: dense connections carry char_emb_dim to the output",
                 fontsize=12, fontweight="bold", color=c["fg"])
    fig.tight_layout()
    save(fig, "intnet-width", theme)


# --------------------------------------------------------------------------
# 2. Tensor width at every stage of the hierarchy, measured from a real model
# --------------------------------------------------------------------------

def shipped_data():
    """A Data matching demo.train.config, wired to the sample sentences."""
    torch.manual_seed(42)
    np.random.seed(42)
    data = Data()
    data.train_dir = data.dev_dir = data.test_dir = SAMPLE
    data.word_emb_dir = None
    data.label_emb_dir = None
    data.HP_gpu = False
    data.word_emb_dim = 100
    data.char_emb_dim = 32
    data.HP_char_hidden_dim = 16
    data.HP_intNet_layer = 7
    data.HP_hidden_dim = 256
    data.global_hidden_dim = 128
    data.initial_feature_alphabets()
    data.build_alphabet(data.train_dir)
    data.fix_alphabet()
    data.generate_instance("train")
    return data


def figure_dimension_flow(theme):
    c = style(theme)
    data = shipped_data()
    model = SeqLabel(data)
    sequence = model.word_hidden
    wordrep = sequence.wordrep

    char_w = wordrep.char_feature.output_dim
    word_w = data.word_emb_dim
    stages = [
        ("Character\nIntNet", char_w, ACCENT[0]),
        ("Word\nembedding", word_w, ACCENT[1]),
        ("WordRep\nconcatenated", wordrep.output_dim, ACCENT[2]),
        ("+ SentenceRep\nglobal feature", sequence.input_size, ACCENT[3]),
        ("Main BiLSTM\noutput", data.HP_hidden_dim, ACCENT[4]),
        ("Label scores\n(%d tags + START/STOP)" % data.label_alphabet.size(),
         data.label_alphabet_size, ACCENT[5]),
    ]

    fig, ax = plt.subplots(figsize=(10, 4))
    names = [s[0] for s in stages]
    widths = [s[1] for s in stages]
    colours = [s[2] for s in stages]
    bars = ax.bar(names, widths, color=colours, width=0.62)
    for bar, w in zip(bars, widths):
        ax.annotate(str(w), (bar.get_x() + bar.get_width() / 2, w),
                    textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=10, fontweight="bold", color=c["fg"])
    ax.set_ylabel("vector width")
    ax.set_title("Feature width through the hierarchy (demo.train.config dimensions)")
    ax.grid(axis="y", alpha=0.35, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=9)
    ax.margins(y=0.18)
    ax.text(0.5, -0.32,
            "%d (char) + %d (word) = %d, then + %d from SentenceRep = %d into the BiLSTM.\n"
            "Only the final bar depends on the corpus: it is the label alphabet of "
            "sample_data/, which a real corpus would widen."
            % (char_w, word_w, wordrep.output_dim, data.global_hidden_dim, sequence.input_size),
            transform=ax.transAxes, ha="center", fontsize=9, color=c["muted"])
    fig.tight_layout()
    save(fig, "dimension-flow", theme)


# --------------------------------------------------------------------------
# 3. Where the parameters actually live
# --------------------------------------------------------------------------

def figure_parameter_budget(theme):
    c = style(theme)
    data = shipped_data()
    model = SeqLabel(data)

    groups = {
        "Main BiLSTM": ["word_hidden.lstm"],
        "SentenceRep (own BiLSTM + WordRep)": ["word_hidden.sentrep"],
        "IntNet character CNN": ["word_hidden.wordrep.char_feature"],
        "Word embeddings": ["word_hidden.wordrep.word_embedding"],
        "MemoryBank": ["word_hidden.mem_bank"],
        "Output projections": ["word_hidden.hidden2tag", "word_hidden.mem2tag"],
        "CRF transitions": ["crf."],
        "Label embeddings": ["word_hidden.wordrep.label_embedding"],
        "Label-attention conv": ["word_hidden.label2cnn"],
    }
    counts, unassigned = {}, 0
    seen = set()
    for name, param in model.named_parameters():
        for group, prefixes in groups.items():
            if any(name.startswith(p) for p in prefixes):
                counts[group] = counts.get(group, 0) + param.numel()
                seen.add(name)
                break
        else:
            unassigned += param.numel()
    if unassigned:
        counts["Other"] = unassigned

    ordered = sorted(counts.items(), key=lambda kv: kv[1])
    labels = [k for k, _ in ordered]
    values = [v for _, v in ordered]
    total = sum(values)

    fig, ax = plt.subplots(figsize=(9, 4.4))
    bars = ax.barh(labels, values, color=[ACCENT[i % len(ACCENT)] for i in range(len(labels))])
    for bar, v in zip(bars, values):
        ax.annotate("%s  (%.1f%%)" % (format(v, ","), 100 * v / total),
                    (v, bar.get_y() + bar.get_height() / 2),
                    textcoords="offset points", xytext=(6, 0),
                    va="center", fontsize=9, color=c["fg"])
    ax.set_xlabel("trainable parameters (thousands)")
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: "%g" % (v / 1000)))
    ax.set_title("Two BiLSTMs hold %.0f%% of the %s parameters"
                 % (100 * (counts["Main BiLSTM"] + counts["SentenceRep (own BiLSTM + WordRep)"]) / total,
                    format(total, ",")))
    ax.grid(axis="x", alpha=0.35, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.margins(x=0.30)
    ax.text(0.5, -0.24,
            "Embedding tables are sized from sample_data/, so they are tiny here; "
            "a real vocabulary would dominate the total.",
            transform=ax.transAxes, ha="center", fontsize=8.5, color=c["muted"])
    fig.tight_layout()
    save(fig, "parameter-budget", theme)


# --------------------------------------------------------------------------
# 4. The CRF transition matrix's structural constraints
# --------------------------------------------------------------------------

def figure_crf_transitions(theme):
    c = style(theme)
    torch.manual_seed(0)
    tagset = 6
    crf = CRF(tagset, gpu=False)
    matrix = crf.transitions.detach().numpy().copy()

    size = matrix.shape[0]
    names = ["PAD"] + ["tag %d" % i for i in range(1, size - 2)] + ["START", "STOP"]
    blocked = matrix <= -10000.0

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.imshow(blocked, cmap=matplotlib.colors.ListedColormap([ACCENT[2], ACCENT[1]]),
              vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("to tag")
    ax.set_ylabel("from tag")
    ax.set_title("CRF transition matrix at initialisation\n%d of %d transitions are pinned unreachable"
                 % (int(blocked.sum()), blocked.size))
    ax.set_xticks(np.arange(-0.5, size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, size, 1), minor=True)
    ax.grid(which="minor", color=c["bg"], linewidth=1.5)
    ax.tick_params(which="minor", length=0)

    handles = [matplotlib.patches.Patch(facecolor=ACCENT[2], label="learnable (starts at 0)"),
               matplotlib.patches.Patch(facecolor=ACCENT[1], label="pinned at -10000")]
    ax.legend(handles=handles, fontsize=8, frameon=False, labelcolor=c["fg"],
              loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    fig.tight_layout()
    save(fig, "crf-transitions", theme)


# --------------------------------------------------------------------------
# 5. Viterbi checked against brute-force enumeration
# --------------------------------------------------------------------------

def brute_force(crf, feats):
    """Score every possible labelling of one sequence."""
    tag_size = feats.size(-1)
    start, stop = tag_size + START_TAG, tag_size + STOP_TAG
    usable = [t for t in range(tag_size) if t not in (0, start, stop)]

    def score(path):
        total, previous = 0.0, start
        for position, tag in enumerate(path):
            total += float(crf.transitions[previous, tag]) + float(feats[position, tag])
            previous = tag
        return total + float(crf.transitions[previous, stop])

    return [score(p) for p in itertools.product(usable, repeat=feats.size(0))]


def figure_viterbi_vs_brute_force(theme):
    c = style(theme)
    decoded_scores, best_scores, partition, brute_partition = [], [], [], []

    for trial in range(60):
        torch.manual_seed(trial)
        crf = CRF(3, gpu=False)
        with torch.no_grad():
            crf.transitions.add_(torch.randn_like(crf.transitions) * (crf.transitions > -1000))
        length = 1 + trial % 3
        feats = torch.randn(1, length, 5)
        mask = torch.ones(1, length, dtype=torch.bool)

        _, path = crf._viterbi_decode(feats, mask)
        tag_size = feats.size(-1)
        start, stop = tag_size + START_TAG, tag_size + STOP_TAG
        total, previous = 0.0, start
        for position, tag in enumerate(path[0].tolist()):
            total += float(crf.transitions[previous, tag]) + float(feats[0, position, tag])
            previous = tag
        total += float(crf.transitions[previous, stop])

        every = brute_force(crf, feats[0])
        decoded_scores.append(total)
        best_scores.append(max(every))
        partition.append(float(crf._calculate_PZ(feats, mask)[0]))
        brute_partition.append(math.log(sum(math.exp(s) for s in every)))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    ax = axes[0]
    lo = min(min(decoded_scores), min(best_scores)) - 0.5
    hi = max(max(decoded_scores), max(best_scores)) + 0.5
    ax.plot([lo, hi], [lo, hi], color=c["muted"], linewidth=1, linestyle="--", zorder=1)
    ax.scatter(best_scores, decoded_scores, s=26, color=ACCENT[0], zorder=2, alpha=0.85)
    ax.set_xlabel("best score over every possible labelling")
    ax.set_ylabel("score of the path Viterbi returned")
    worst = max(abs(a - b) for a, b in zip(decoded_scores, best_scores))
    ax.set_title("Viterbi finds the argmax\nlargest gap over %d problems: %.1e"
                 % (len(decoded_scores), worst))
    ax.grid(alpha=0.35, linewidth=0.7)

    ax = axes[1]
    lo = min(min(partition), min(brute_partition)) - 0.5
    hi = max(max(partition), max(brute_partition)) + 0.5
    ax.plot([lo, hi], [lo, hi], color=c["muted"], linewidth=1, linestyle="--", zorder=1)
    ax.scatter(brute_partition, partition, s=26, color=ACCENT[3], zorder=2, alpha=0.85)
    ax.set_xlabel("log-sum-exp over every labelling")
    ax.set_ylabel("forward algorithm's log Z")
    worst = max(abs(a - b) for a, b in zip(partition, brute_partition))
    ax.set_title("The partition function agrees\nlargest gap: %.1e" % worst)
    ax.grid(alpha=0.35, linewidth=0.7)

    fig.suptitle("Every point is one randomly generated CRF checked exhaustively",
                 fontsize=11, fontweight="bold", color=c["fg"])
    fig.tight_layout()
    save(fig, "crf-verification", theme)


# --------------------------------------------------------------------------
# 6. Tag schemes side by side
# --------------------------------------------------------------------------

def figure_tag_schemes(theme):
    """
    The argument for BIOES is adjacent entities of the same type. The tags below
    are produced by the repository's own converter, not typed by hand.
    """
    import tempfile
    from utils.tagSchemeConverter import BIO2BIOES

    c = style(theme)
    tokens = ["Peter", "Blackburn", "Kofi", "Annan", "visited", "Brussels"]
    bio = ["B-PER", "I-PER", "B-PER", "I-PER", "O", "B-LOC"]

    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "in.txt")
        dst = os.path.join(tmp, "out.txt")
        with io.open(src, "w", encoding="utf-8") as f:
            f.write("\n".join("%s %s" % pair for pair in zip(tokens, bio)) + "\n\n")
        BIO2BIOES(src, dst)
        bioes = [line.split()[-1] for line in io.open(dst, encoding="utf-8")
                 if line.strip()]
    assert len(bioes) == len(tokens)

    colour = {"PER": ACCENT[0], "LOC": ACCENT[2]}
    n = len(tokens)

    fig, ax = plt.subplots(figsize=(9.8, 3.4))
    ax.set_xlim(-1.35, n - 0.42)
    ax.set_ylim(-0.30, 2.30)
    ax.axis("off")

    # Shade the two adjacent PER entities so the boundary is visible at a glance.
    for span in [(0, 1), (2, 3)]:
        ax.add_patch(matplotlib.patches.Rectangle(
            (span[0] - 0.44, 0.62), (span[1] - span[0]) + 0.88, 1.52,
            facecolor=ACCENT[0], alpha=0.10, edgecolor="none", zorder=0))

    for row, (scheme, tags) in enumerate([("BIOES", bioes), ("BIO", bio)]):
        y = 1.82 - row * 0.72
        ax.text(-1.25, y, scheme, fontsize=11, fontweight="bold",
                color=c["fg"], ha="left", va="center")
        for i, tag in enumerate(tags):
            kind = tag.split("-")[-1] if "-" in tag else None
            face = colour.get(kind)
            ax.text(i, y, tag, ha="center", va="center", fontsize=9.5,
                    fontweight="bold" if face else "normal",
                    color="#ffffff" if face else c["muted"],
                    bbox=dict(boxstyle="round,pad=0.34",
                              facecolor=face if face else c["bg"],
                              edgecolor=face if face else c["grid"], linewidth=1.2),
                    zorder=2)

    # Ring the one cell that carries the whole argument.
    ax.add_patch(matplotlib.patches.FancyBboxPatch(
        (2 - 0.30, 1.10 - 0.13), 0.60, 0.26,
        boxstyle="round,pad=0.10", facecolor="none",
        edgecolor=ACCENT[1], linewidth=2.0, zorder=3))

    for i, token in enumerate(tokens):
        ax.text(i, 0.34, token, ha="center", va="center", fontsize=10, color=c["fg"])

    ax.text(-1.25, -0.12,
            "Two adjacent PER entities, then a LOC. BIOES marks every end (E-) and "
            "every one-token entity (S-) outright;\nin BIO the only thing separating "
            "the two people is the circled second B-.",
            ha="left", va="center", fontsize=8.5, color=c["muted"])

    ax.set_title("Why the shipped config uses BIOES", fontsize=12,
                 fontweight="bold", color=c["fg"], pad=12)
    fig.tight_layout()
    save(fig, "tag-schemes", theme)


FIGURES = [
    figure_intnet_width,
    figure_dimension_flow,
    figure_parameter_budget,
    figure_crf_transitions,
    figure_viterbi_vs_brute_force,
    figure_tag_schemes,
]


def main():
    for build in FIGURES:
        for theme in THEMES:
            build(theme)
    print("\n%d figures x %d themes -> %s" % (len(FIGURES), len(THEMES), ASSETS))


if __name__ == "__main__":
    main()
