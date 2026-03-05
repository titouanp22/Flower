from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch


def plot_snapshots(snapshots, true_samples, h_vec, y_value, out_file: Path) -> None:
    keys = sorted(snapshots.keys())
    n = len(keys)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    xmin, xmax = -1.5, 1.5
    xline = torch.linspace(xmin, xmax, 200)
    yline = (y_value - h_vec[0] * xline) / (h_vec[1] + 1e-12)

    for i, k in enumerate(keys):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        xt = snapshots[k]
        ax.scatter(true_samples[:, 0], true_samples[:, 1], s=2, alpha=0.2, label="true posterior")
        ax.scatter(xt[:, 0], xt[:, 1], s=2, alpha=0.35, label="flower")
        ax.plot(xline.numpy(), yline.numpy(), "--", lw=1.0, label="h^T x = y")
        ax.set_title(f"step {k}")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(xmin, xmax)
        ax.set_aspect("equal")
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)

    for i in range(n, rows * cols):
        r, c = divmod(i, cols)
        axes[r][c].axis("off")

    fig.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def plot_step_by_step(step_details, true_samples, h_vec, y_value, out_file: Path) -> None:
    keys = sorted(step_details.keys())
    if not keys:
        raise ValueError("step_details is empty")

    fig, axes = plt.subplots(len(keys), 4, figsize=(16, 4 * len(keys)), squeeze=False)
    xmin, xmax = -1.5, 1.5
    xline = torch.linspace(xmin, xmax, 200)
    yline = (y_value - h_vec[0] * xline) / (h_vec[1] + 1e-12)

    titles = ["x_t (start)", "Step 1: x_hat", "Step 2: x_tilde", "Step 3: x_{t+dt}"]
    for j, title in enumerate(titles):
        axes[0][j].set_title(title)

    for i, step in enumerate(keys):
        row = step_details[step]
        items = [row["x_t"], row["x_hat"], row["x_tilde"], row["x_next"]]
        t = row["t"]

        for j, pts in enumerate(items):
            ax = axes[i][j]
            ax.scatter(true_samples[:, 0], true_samples[:, 1], s=2, alpha=0.2, label="true posterior")
            ax.scatter(pts[:, 0], pts[:, 1], s=2, alpha=0.35, label="flower")
            ax.plot(xline.numpy(), yline.numpy(), "--", lw=1.0, label="h^T x = y")
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(xmin, xmax)
            ax.set_aspect("equal")
            if j == 0:
                ax.set_ylabel(f"step={step}, t={t:.3f}")
            if i == 0 and j == 0:
                ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=160)
    plt.close(fig)
