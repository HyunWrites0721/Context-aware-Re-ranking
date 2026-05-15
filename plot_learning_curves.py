"""
Plot SASRec training curves from a RecBole log file.

Usage:
    python plot_learning_curves.py <log_path>

Output:
    figures/<log_stem>.png
"""

import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


FIGURES_DIR = "./figures"
PAPER_NDCG10 = 0.0517  # Table 2, SASRec IDRec, MicroLens-100K


def parse_log(log_path):
    train_epochs, train_losses = [], []
    eval_epochs, eval_scores = [], []
    best_valid, test_result = {}, {}

    train_pat = re.compile(r"epoch (\d+) training \[time:.+, train loss: ([\d.]+)\]")
    eval_pat  = re.compile(r"epoch (\d+) evaluating \[time:.+, valid_score: ([\d.]+)\]")
    kv_pat    = re.compile(r"'([\w@]+)',\s*([\d.]+)")

    with open(log_path) as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        m = train_pat.search(line)
        if m:
            train_epochs.append(int(m.group(1)))
            train_losses.append(float(m.group(2)))
            continue
        m = eval_pat.search(line)
        if m:
            eval_epochs.append(int(m.group(1)))
            eval_scores.append(float(m.group(2)))
            continue
        if "best valid" in line:
            best_valid = {k: float(v) for k, v in kv_pat.findall(line)}
        if "test result" in line:
            test_result = {k: float(v) for k, v in kv_pat.findall(line)}

    return train_epochs, train_losses, eval_epochs, eval_scores, best_valid, test_result


def plot(log_path, output_path):
    train_epochs, train_losses, eval_epochs, eval_scores, best_valid, test_result = parse_log(log_path)

    log_stem = os.path.splitext(os.path.basename(log_path))[0]

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"SASRec Training Curves\n(microLens-100k | {log_stem})",
                 fontsize=12, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # 1) Train Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(train_epochs, train_losses, color="#2196F3", linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss")
    ax1.set_title("Training Loss")
    ax1.grid(True, alpha=0.3)

    # 2) Valid NDCG@10
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(eval_epochs, eval_scores, color="#4CAF50", linewidth=1.5,
             marker="o", markersize=4, label="Valid NDCG@10")
    best_epoch = eval_epochs[int(np.argmax(eval_scores))]
    best_score = max(eval_scores)
    ax2.axvline(best_epoch, color="#FF5722", linestyle="--", linewidth=1, alpha=0.7,
                label=f"Best epoch={best_epoch}")
    ax2.axhline(PAPER_NDCG10, color="#9C27B0", linestyle=":", linewidth=1.5, alpha=0.8,
                label=f"Paper = {PAPER_NDCG10:.4f}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("NDCG@10")
    ax2.set_title("Validation NDCG@10")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)

    # 3) Train Loss (log scale)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.semilogy(train_epochs, train_losses, color="#FF9800", linewidth=1.5)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Train Loss (log scale)")
    ax3.set_title("Training Loss (Log Scale)")
    ax3.grid(True, alpha=0.3, which="both")

    # 4) Final metrics bar chart
    ax4 = fig.add_subplot(gs[1, 1])
    metrics    = ["Hit@10", "Hit@20", "NDCG@10", "NDCG@20"]
    valid_vals = [best_valid.get("hit@10", 0), best_valid.get("hit@20", 0),
                  best_valid.get("ndcg@10", 0), best_valid.get("ndcg@20", 0)]
    test_vals  = [test_result.get("hit@10", 0), test_result.get("hit@20", 0),
                  test_result.get("ndcg@10", 0), test_result.get("ndcg@20", 0)]

    x = np.arange(len(metrics))
    w = 0.35
    bars1 = ax4.bar(x - w/2, valid_vals, w, label="Best Valid", color="#42A5F5", alpha=0.85)
    bars2 = ax4.bar(x + w/2, test_vals,  w, label="Test",       color="#66BB6A", alpha=0.85)
    for bar in list(bars1) + list(bars2):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                 f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=7.5)
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics)
    ax4.set_ylabel("Score")
    ax4.set_title("Final Evaluation Metrics")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis="y")

    test_ndcg = test_result.get("ndcg@10", 0)
    fig.text(0.5, 0.01,
             f"Best valid NDCG@10={best_score:.4f}  |  Test NDCG@10={test_ndcg:.4f}"
             f"  |  Paper NDCG@10={PAPER_NDCG10:.4f}  ({test_ndcg/PAPER_NDCG10*100:.1f}% of paper)",
             ha="center", fontsize=9, color="#555555")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    log_path = sys.argv[1] if len(sys.argv) > 1 else \
        "./log/SASRec/SASRec-microlens100k-May-15-2026_06-52-54-c18cd4.log"
    stem = os.path.splitext(os.path.basename(log_path))[0]
    output_path = os.path.join(FIGURES_DIR, f"{stem}.png")
    plot(log_path, output_path)
