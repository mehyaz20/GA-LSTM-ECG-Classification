"""
plot_training_curves.py
------------------------
Regenerates the two-panel "Training Loss" / "Validation F1-Score" figure
in the same visual style used for the MIT-BIH dataset (Figure 3), so it can
be reproduced for PTB (Figure 5) or any future dataset directly from your
logged training history.

Usage
-----
    from plot_training_curves import plot_training_curves
    plot_training_curves(
        epochs=list(range(1, 101)),
        train_loss=train_loss_history,      # list/array, one value per epoch
        val_f1=val_f1_history,              # list/array, one value per epoch (0-100 scale)
        dataset_name="PTB",
        out_path="figs/fig5_ptb_training.png",
    )

`train_loss_history` and `val_f1_history` should come from your actual
PyTorch training loop (e.g. append `loss.item()` and `f1_score(...)` each
epoch and save them to a CSV/JSON so this script can be re-run at
publication quality/resolution at any time).
"""

import matplotlib.pyplot as plt


def plot_training_curves(epochs, train_loss, val_f1, dataset_name, out_path,
                          loss_color="blue", f1_color="green", figsize=(9, 3)):
    assert len(epochs) == len(train_loss) == len(val_f1), "epochs, train_loss, val_f1 must be the same length"

    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=200)

    axes[0].plot(epochs, train_loss, color=loss_color, marker="o", markersize=2, linewidth=1)
    axes[0].set_title(f"Training Loss — {dataset_name}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, val_f1, color=f1_color, marker="o", markersize=2, linewidth=1)
    axes[1].set_title(f"Validation F1 — {dataset_name}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1 (%)")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    # Placeholder synthetic example ONLY for demonstrating the plotting
    # code end-to-end. Replace with your real per-epoch training history
    # (e.g. loaded from a CSV you saved during the actual PTB training run)
    # before using this for the paper -- do not use synthetic curves in the
    # submitted manuscript.
    import numpy as np
    epochs = list(range(1, 101))
    train_loss = 0.45 * np.exp(-0.04 * np.array(epochs)) + 0.02
    val_f1 = 70 + 30 * (1 - np.exp(-0.05 * np.array(epochs)))

    plot_training_curves(
        epochs, train_loss, val_f1,
        dataset_name="PTB (SYNTHETIC EXAMPLE — replace with real log)",
        out_path="figs/fig5_ptb_training_regenerated_example.png",
    )
