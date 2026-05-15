import re
import sys
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

LOG_PATH = sys.argv[1] if len(sys.argv) > 1 else "./log/SASRec/SASRec-microlens100k-May-15-2026_06-52-54-c18cd4.log"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_PATH = f"./learning_curves_{timestamp}.png"

# ── parse ──────────────────────────────────────────────────────────────────
train_loss_pattern = re.compile(r"epoch (\d+) training \[time: .+, train loss: ([\d.]+)\]")
eval_pattern       = re.compile(r"epoch (\d+) evaluating \[time: .+, valid_score: ([\d.]+)\]")

train_epochs, train_losses = [], []
eval_epochs,  eval_scores  = [], []

with open(LOG_PATH) as f:
    for line in f:
        m = train_loss_pattern.search(line)
        if m:
            train_epochs.append(int(m.group(1)))
            train_losses.append(float(m.group(2)))
            continue
        m = eval_pattern.search(line)
        if m:
            eval_epochs.append(int(m.group(1)))
            eval_scores.append(float(m.group(2)))

# reported final metrics
best_valid = {"hit@10": 0.0750, "hit@20": 0.1021, "ndcg@10": 0.0423, "ndcg@20": 0.0491}
test_result = {"hit@10": 0.0544, "hit@20": 0.0779, "ndcg@10": 0.0301, "ndcg@20": 0.0361}

# paper reference (Table 2, SASRec IDRec)
paper_ndcg10 = 0.0517

# ── plot ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
fig.suptitle("SASRec Training Curves\n(microLens-100k)", fontsize=15, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

# 1) Train Loss
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(train_epochs, train_losses, color="#2196F3", linewidth=1.5, label="Train Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Train Loss")
ax1.set_title("Training Loss")
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9)

# 2) Valid NDCG@10 over eval steps
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(eval_epochs, eval_scores, color="#4CAF50", linewidth=1.5,
         marker="o", markersize=4, label="Valid NDCG@10")
best_epoch = eval_epochs[int(np.argmax(eval_scores))]
best_score = max(eval_scores)
ax2.axvline(best_epoch, color="#FF5722", linestyle="--", linewidth=1, alpha=0.7,
            label=f"Best epoch={best_epoch}")
ax2.axhline(paper_ndcg10, color="#9C27B0", linestyle=":", linewidth=1.5, alpha=0.8,
            label=f"Paper NDCG@10 = {paper_ndcg10:.4f}")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("NDCG@10")
ax2.set_title("Validation NDCG@10")
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=9)

# 3) Train Loss (log scale)
ax3 = fig.add_subplot(gs[1, 0])
ax3.semilogy(train_epochs, train_losses, color="#FF9800", linewidth=1.5, label="Train Loss (log)")
ax3.set_xlabel("Epoch")
ax3.set_ylabel("Train Loss (log scale)")
ax3.set_title("Training Loss (Log Scale)")
ax3.grid(True, alpha=0.3, which="both")
ax3.legend(fontsize=9)

# 4) Final metrics bar chart
ax4 = fig.add_subplot(gs[1, 1])
metrics   = ["Hit@10", "Hit@20", "NDCG@10", "NDCG@20"]
valid_vals = [best_valid["hit@10"],  best_valid["hit@20"],
              best_valid["ndcg@10"], best_valid["ndcg@20"]]
test_vals  = [test_result["hit@10"],  test_result["hit@20"],
              test_result["ndcg@10"], test_result["ndcg@20"]]

x = np.arange(len(metrics))
width = 0.35
bars1 = ax4.bar(x - width/2, valid_vals, width, label="Best Valid", color="#42A5F5", alpha=0.85)
bars2 = ax4.bar(x + width/2, test_vals,  width, label="Test",       color="#66BB6A", alpha=0.85)

for bar in bars1:
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
             f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=7.5)
for bar in bars2:
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
             f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=7.5)

ax4.set_xticks(x)
ax4.set_xticklabels(metrics)
ax4.set_ylabel("Score")
ax4.set_title("Final Evaluation Metrics")
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, axis="y")

# annotation
fig.text(0.5, 0.01,
         f"Best valid NDCG@10={best_score:.4f}  |  Test NDCG@10={test_result['ndcg@10']:.4f}"
         f"  |  Paper NDCG@10={paper_ndcg10:.4f}  (achieved {test_result['ndcg@10']/paper_ndcg10*100:.1f}%)",
         ha="center", fontsize=9, color="#555555")

plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved: {OUTPUT_PATH}")
