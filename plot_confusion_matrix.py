"""
plot_confusion_matrix.py
-------------------------
Regenerates a confusion-matrix figure in the same visual style as the
existing MIT-BIH confusion matrix (Figure 2) so that the PTB confusion
matrix (Figure 4) and any future dataset use a consistent look.

Usage
-----
    from plot_confusion_matrix import plot_confusion_matrix
    plot_confusion_matrix(y_true, y_pred, labels=["Normal", "Abnormal (MI)"],
                           title="PTB Diagnostic Confusion Matrix",
                           out_path="figs/fig4_ptb_confmat.png")

If you only have the summary counts (TP/FP/FN/TN) rather than the raw
y_true / y_pred vectors -- e.g. because you are re-plotting from a paper
draft rather than from a live run -- use `plot_confusion_matrix_from_counts`
instead (see the __main__ block below for a worked PTB example that
reproduces the numbers reported in the paper).
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, labels, title, out_path, cmap="Blues", figsize=(4, 3.5)):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    _plot_matrix(cm, labels, title, out_path, cmap=cmap, figsize=figsize)


def plot_confusion_matrix_from_counts(matrix, labels, title, out_path, cmap="Blues", figsize=(4, 3.5)):
    """matrix: 2D array-like, rows = true label, cols = predicted label."""
    cm = np.array(matrix)
    _plot_matrix(cm, labels, title, out_path, cmap=cmap, figsize=figsize)


def _plot_matrix(cm, labels, title, out_path, cmap="Blues", figsize=(4, 3.5)):
    fig, ax = plt.subplots(figsize=figsize, dpi=200)
    im = ax.imshow(cm, cmap=cmap)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    if title:
        ax.set_title(title)

    vmax = cm.max()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > vmax * 0.6 else "black"
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color=color, fontsize=11)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    # Worked example reproducing the PTB test-set confusion matrix reported
    # in the paper (Precision=99.50%, Recall=98.39% on the Abnormal class,
    # Accuracy=99.42%). Replace this block with your real y_true / y_pred
    # (or your real held-out counts) once you have them.
    # rows = true label [Normal, Abnormal], cols = predicted [Normal, Abnormal]
    ptb_counts = [
        [2097, 4],    # True Normal
        [13, 796],    # True Abnormal (MI)
    ]
    plot_confusion_matrix_from_counts(
        ptb_counts,
        labels=["Normal", "Abnormal (MI)"],
        title="PTB Diagnostic Database — Confusion Matrix",
        out_path="figs/fig4_ptb_confmat_regenerated.png",
    )
