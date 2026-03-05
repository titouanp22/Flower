from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.create_gmm_data import generate_gmm_data
from scripts.gmm_flow_model import TimeConditionedMLP, sample_with_euler


def train_flow_matching_gmm(
    *,
    output_dir: str | Path = "results/models_gmm",
    seed: int = 7,
    n_prior: int = 4000,
    n_posterior: int = 12000,
    cov_scalar: float = 0.0625,
    sigma_n: float = 0.25,
    y_clean: float = 1.0,
    batch_size: int = 256,
    epochs: int = 400,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    depth: int = 3,
    device: str = "cpu",
    log_every: int = 50,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    device_obj = torch.device(device)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    means = np.asarray([[-0.25, -0.25], [-0.25, 0.25], [0.25, -0.25]], dtype=np.float64)
    h = np.asarray([1.5, 1.5], dtype=np.float64)
    data = generate_gmm_data(
        seed=seed,
        n_prior=n_prior,
        n_posterior=n_posterior,
        means=means,
        cov_scalar=cov_scalar,
        h=h,
        sigma_n=sigma_n,
        y_clean=y_clean,
    )

    target = torch.tensor(data["posterior_samples"], dtype=torch.float32)
    loader = DataLoader(TensorDataset(target), batch_size=batch_size, shuffle=True, drop_last=True)

    model = TimeConditionedMLP(input_dim=2, hidden_dim=hidden_dim, depth=depth).to(device_obj)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses: list[float] = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_items = 0

        for (x1,) in loader:
            x1 = x1.to(device_obj)
            bsz = x1.shape[0]

            x0 = torch.randn_like(x1)
            t = torch.rand(bsz, 1, device=device_obj)
            xt = (1.0 - t) * x0 + t * x1
            target_v = x1 - x0

            pred_v = model(xt, t)
            loss = torch.mean((pred_v - target_v) ** 2)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * bsz
            n_items += bsz

        mean_loss = epoch_loss / max(n_items, 1)
        losses.append(mean_loss)
        if epoch % log_every == 0 or epoch == epochs - 1:
            print(f"epoch={epoch:04d} loss={mean_loss:.6f}")

    model_path = output_path / "model_final.pt"
    torch.save(model.state_dict(), model_path)

    losses_path = output_path / "losses.npy"
    np.save(losses_path, np.asarray(losses, dtype=np.float32))

    with torch.no_grad():
        generated = sample_with_euler(model, n_samples=3000, n_steps=250, device=device_obj).cpu().numpy()

    summary = {
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "hidden_dim": hidden_dim,
        "depth": depth,
        "device": str(device_obj),
        "final_loss": float(losses[-1]),
        "model_path": str(model_path.resolve()),
        "losses_path": str(losses_path.resolve()),
        "y_observed": float(data["y_observed"]),
        "post_weights": [float(x) for x in data["post_weights"]],
    }

    samples_path = output_path / "generated_samples.npy"
    np.save(samples_path, generated.astype(np.float32))
    summary["generated_samples_path"] = str(samples_path.resolve())

    summary_path = output_path / "train_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "model": model,
        "data": data,
        "losses": losses,
        "generated_samples": generated,
        "summary": summary,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a 2D flow matching model on GMM posterior samples.")
    parser.add_argument("--output-dir", type=str, default="results/models_gmm")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-prior", type=int, default=4000)
    parser.add_argument("--n-posterior", type=int, default=12000)
    parser.add_argument("--cov-scalar", type=float, default=0.0625)
    parser.add_argument("--sigma-n", type=float, default=0.25)
    parser.add_argument("--y-clean", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log-every", type=int, default=50)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = train_flow_matching_gmm(
        output_dir=args.output_dir,
        seed=args.seed,
        n_prior=args.n_prior,
        n_posterior=args.n_posterior,
        cov_scalar=args.cov_scalar,
        sigma_n=args.sigma_n,
        y_clean=args.y_clean,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        device=args.device,
        log_every=args.log_every,
    )
    print("Training complete.")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
