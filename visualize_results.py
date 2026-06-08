"""
Visualization for wav2vec2 depression detection results.
Reads from models_wav2vec2/test_results.json and training_history.json
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── Load data ──────────────────────────────────────────────────────────────────
results_dir = Path("models_wav2vec2")

with open(results_dir / "test_results.json") as f:
    test = json.load(f)

with open(results_dir / "training_history.json") as f:
    history = json.load(f)

train_loss   = history["train_loss"]
val_metrics  = history["val_metrics"]
epochs       = list(range(1, len(train_loss) + 1))

val_acc  = [m["accuracy"] for m in val_metrics]
val_f1   = [m["f1"]       for m in val_metrics]
val_auc  = [m["roc_auc"]  for m in val_metrics]
val_loss = [m["loss"]      for m in val_metrics]

cm = np.array(test["confusion_matrix"])   # [[TN, FP], [FN, TP]]

# ── Figure layout ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
fig.suptitle("Wav2Vec2 Depression Detection — Results", fontsize=16, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])   # train vs val loss
ax2 = fig.add_subplot(gs[0, 1])   # val accuracy & F1
ax3 = fig.add_subplot(gs[0, 2])   # val ROC-AUC
ax4 = fig.add_subplot(gs[1, 0])   # confusion matrix
ax5 = fig.add_subplot(gs[1, 1])   # per-class F1
ax6 = fig.add_subplot(gs[1, 2])   # test metrics bar chart

# ── 1. Train vs Val Loss ───────────────────────────────────────────────────────
ax1.plot(epochs, train_loss, "o-", color="#2196F3", label="Train Loss", linewidth=2)
ax1.plot(epochs, val_loss,   "s--", color="#FF5722", label="Val Loss",   linewidth=2)
ax1.set_title("Train vs Validation Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xticks(epochs)

# ── 2. Val Accuracy & F1 ──────────────────────────────────────────────────────
ax2.plot(epochs, val_acc, "o-", color="#4CAF50", label="Accuracy", linewidth=2)
ax2.plot(epochs, val_f1,  "s--", color="#9C27B0", label="F1 Score", linewidth=2)
ax2.axhline(y=test["accuracy"], color="#4CAF50", linestyle=":", alpha=0.6, label=f"Test Acc {test['accuracy']:.4f}")
ax2.axhline(y=test["f1"],       color="#9C27B0", linestyle=":", alpha=0.6, label=f"Test F1  {test['f1']:.4f}")
ax2.set_title("Validation Accuracy & F1")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Score")
ax2.set_ylim(0.6, 1.01)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(epochs)

# ── 3. Val ROC-AUC ────────────────────────────────────────────────────────────
ax3.plot(epochs, val_auc, "o-", color="#FF9800", linewidth=2)
ax3.axhline(y=test["roc_auc"], color="#FF9800", linestyle=":", alpha=0.6,
            label=f"Test AUC {test['roc_auc']:.4f}")
ax3.set_title("Validation ROC-AUC")
ax3.set_xlabel("Epoch")
ax3.set_ylabel("ROC-AUC")
ax3.set_ylim(0.85, 1.01)
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_xticks(epochs)

# ── 4. Confusion Matrix ───────────────────────────────────────────────────────
im = ax4.imshow(cm, interpolation="nearest", cmap="Blues")
fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
ax4.set_title("Test Confusion Matrix")
ax4.set_xlabel("Predicted Label")
ax4.set_ylabel("True Label")
ax4.set_xticks([0, 1]); ax4.set_xticklabels(["Non-Dep (0)", "Dep (1)"])
ax4.set_yticks([0, 1]); ax4.set_yticklabels(["Non-Dep (0)", "Dep (1)"])

thresh = cm.max() / 2.0
for i in range(2):
    for j in range(2):
        ax4.text(j, i, f"{cm[i, j]}",
                 ha="center", va="center", fontsize=14, fontweight="bold",
                 color="white" if cm[i, j] > thresh else "black")

tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
ax4.set_xlabel(f"Predicted  |  Sensitivity: {sensitivity:.3f}  Specificity: {specificity:.3f}")

# ── 5. Per-Class F1 over epochs ───────────────────────────────────────────────
class0_f1 = [m["class_0_f1"] for m in val_metrics]
class1_f1 = [m["class_1_f1"] for m in val_metrics]

ax5.plot(epochs, class0_f1, "o-", color="#00BCD4", label="Class 0 (Non-Dep)", linewidth=2)
ax5.plot(epochs, class1_f1, "s--", color="#E91E63", label="Class 1 (Dep)",     linewidth=2)
ax5.axhline(y=test["class_0_f1"], color="#00BCD4", linestyle=":", alpha=0.6,
            label=f"Test C0 F1 {test['class_0_f1']:.4f}")
ax5.axhline(y=test["class_1_f1"], color="#E91E63", linestyle=":", alpha=0.6,
            label=f"Test C1 F1 {test['class_1_f1']:.4f}")
ax5.set_title("Per-Class F1 Score")
ax5.set_xlabel("Epoch")
ax5.set_ylabel("F1 Score")
ax5.set_ylim(0.6, 1.01)
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)
ax5.set_xticks(epochs)

# ── 6. Test Metrics Bar Chart ─────────────────────────────────────────────────
metrics_names  = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
metrics_values = [test["accuracy"], test["precision"], test["recall"],
                  test["f1"], test["roc_auc"]]
colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#FF5722"]

bars = ax6.bar(metrics_names, metrics_values, color=colors, edgecolor="white", linewidth=1.2)
ax6.set_title("Test Set Metrics Summary")
ax6.set_ylabel("Score")
ax6.set_ylim(0.85, 1.02)
ax6.grid(True, axis="y", alpha=0.3)

for bar, val in zip(bars, metrics_values):
    ax6.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
             f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

# ── Save ───────────────────────────────────────────────────────────────────────
out_path = "wav2vec2_results_visualization.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved → {out_path}")
plt.show()
