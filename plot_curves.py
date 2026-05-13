"""
Parse a RecBole log file and plot SASRec learning curves.

Usage:
    python plot_curves.py --log log/SASRec-<datetime>.log
    python plot_curves.py --log log/SASRec-<datetime>.log --out results/curves.png
"""

import argparse
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_log(log_path: str):
    """Extract per-epoch train loss and validation metrics from a RecBole log."""
    train_losses = {}
    valid_metrics = {}  # epoch -> {metric_name: value}
    current_epoch = None

    with open(log_path) as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        epoch_match = re.search(r'epoch\s+(\d+)\s+training', line, re.IGNORECASE)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))

        loss_match = re.search(r'train loss1\s*:\s*([\d.]+)', line, re.IGNORECASE)
        if loss_match and current_epoch is not None:
            train_losses[current_epoch] = float(loss_match.group(1))

        if 'valid result' in line.lower() and current_epoch is not None:
            for j in range(i + 1, min(i + 5, len(lines))):
                candidate = lines[j]
                if any(m in candidate.lower() for m in ['hit', 'ndcg']):
                    parsed = {
                        m.group(1).lower(): float(m.group(2))
                        for m in re.finditer(r'(\w+@\d+)\s*:\s*([\d.]+)', candidate, re.IGNORECASE)
                    }
                    if parsed:
                        valid_metrics[current_epoch] = parsed
                    break

    return train_losses, valid_metrics


def plot(log_path: str, out_path: str = 'learning_curve.png') -> None:
    train_losses, valid_metrics = parse_log(log_path)

    if not train_losses and not valid_metrics:
        print("[plot] No parseable metrics found. Check log path.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('SASRec · MicroLens-100k', fontsize=13)

    # --- Training loss ---
    if train_losses:
        epochs = sorted(train_losses)
        axes[0].plot(epochs, [train_losses[e] for e in epochs], color='steelblue')
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('CE Loss')
    axes[0].grid(True, alpha=0.3)

    # --- Hit Ratio ---
    if valid_metrics:
        epochs = sorted(valid_metrics)
        for key, color in [('hit@10', 'tab:blue'), ('hit@20', 'tab:cyan')]:
            values = [valid_metrics[e].get(key) for e in epochs]
            if any(v is not None for v in values):
                axes[1].plot(epochs, values, label=key.upper(), color=color)
    axes[1].set_title('Hit Ratio (Validation)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('HR')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # --- NDCG ---
    if valid_metrics:
        for key, color in [('ndcg@10', 'tab:orange'), ('ndcg@20', 'tab:red')]:
            values = [valid_metrics[e].get(key) for e in epochs]
            if any(v is not None for v in values):
                axes[2].plot(epochs, values, label=key.upper(), color=color)
    axes[2].set_title('NDCG (Validation)')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('NDCG')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"[plot] Saved → {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot RecBole SASRec learning curves')
    parser.add_argument('--log', required=True, help='RecBole log file (log/SASRec-*.log)')
    parser.add_argument('--out', default='learning_curve.png', help='Output image path')
    args = parser.parse_args()
    plot(args.log, args.out)
