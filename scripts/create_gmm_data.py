from pathlib import Path
import json
import numpy as np


def parse_list_of_floats(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",")]


def parse_means(s: str) -> np.ndarray:
    # Example: "-0.25,-0.25;-0.25,0.25;0.25,-0.25"
    rows = []
    for block in s.split(";"):
        rows.append(parse_list_of_floats(block))
    means = np.asarray(rows, dtype=np.float64)
    if means.ndim != 2 or means.shape[1] != 2:
        raise ValueError("means must define K points in 2D")
    return means


def sample_gmm_prior(n: int, means: np.ndarray, cov_scalar: float, rng: np.random.Generator) -> np.ndarray:
    k = means.shape[0]
    ids = rng.integers(0, k, size=n)
    eps = rng.normal(size=(n, 2))
    return means[ids] + np.sqrt(cov_scalar) * eps


def posterior_params(means: np.ndarray, cov_scalar: float, h: np.ndarray, sigma_n: float, y: float):
    sigma = cov_scalar * np.eye(2)
    sigma_inv = np.linalg.inv(sigma)
    h = h.reshape(2, 1)
    a = sigma_inv + (1.0 / (sigma_n ** 2)) * (h @ h.T)
    post_cov = np.linalg.inv(a)

    post_means = []
    for i in range(means.shape[0]):
        rhs = (1.0 / (sigma_n ** 2)) * h.flatten() * y + sigma_inv @ means[i]
        post_means.append(post_cov @ rhs)
    post_means = np.stack(post_means, axis=0)

    var_y = float((h.T @ sigma @ h).item()) + sigma_n ** 2
    m_y = (means @ h).squeeze(-1)
    logw = -0.5 * ((y - m_y) ** 2) / var_y
    logw = logw - np.max(logw)
    w = np.exp(logw)
    w = w / np.sum(w)
    return w, post_means, post_cov


def sample_posterior(n: int, weights: np.ndarray, post_means: np.ndarray, post_cov: np.ndarray, rng: np.random.Generator):
    k = weights.shape[0]
    ids = rng.choice(np.arange(k), size=n, replace=True, p=weights)
    eps = rng.normal(size=(n, 2))
    chol = np.linalg.cholesky(post_cov)
    return post_means[ids] + eps @ chol.T


def generate_gmm_data(
    *,
    seed: int = 0,
    n_prior: int = 4000,
    n_posterior: int = 4000,
    means: np.ndarray | None = None,
    cov_scalar: float = 0.0625,
    h: np.ndarray | None = None,
    sigma_n: float = 0.25,
    y_clean: float = 1.0,
):
    """
    Generate toy GMM prior/posterior data for notebook experiments.
    Returns a dict with arrays and metadata.
    """
    if means is None:
        means = np.asarray([[-0.25, -0.25], [-0.25, 0.25], [0.25, -0.25]], dtype=np.float64)
    if h is None:
        h = np.asarray([1.5, 1.5], dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    if means.ndim != 2 or means.shape[1] != 2:
        raise ValueError("means must have shape [K,2]")
    if h.shape != (2,):
        raise ValueError("h must have shape [2]")

    rng = np.random.default_rng(seed)
    prior_samples = sample_gmm_prior(n_prior, means, cov_scalar, rng)

    obs_noise = rng.normal(loc=0.0, scale=sigma_n)
    y_observed = float(y_clean + obs_noise)

    weights, post_means, post_cov = posterior_params(
        means=means,
        cov_scalar=cov_scalar,
        h=h,
        sigma_n=sigma_n,
        y=y_observed,
    )
    posterior_samples = sample_posterior(n_posterior, weights, post_means, post_cov, rng)

    return {
        "prior_samples": prior_samples,
        "posterior_samples": posterior_samples,
        "means": means,
        "post_means": post_means,
        "post_cov": post_cov,
        "post_weights": weights,
        "h": h,
        "y_clean": float(y_clean),
        "y_observed": y_observed,
        "sigma_n": float(sigma_n),
        "cov_scalar": float(cov_scalar),
        "seed": int(seed),
    }
