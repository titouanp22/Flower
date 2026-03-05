from __future__ import annotations

import torch


def _time_tensor(x: torch.Tensor, t: float, *, flat: bool) -> torch.Tensor:
    if flat:
        return torch.full((x.shape[0],), float(t), device=x.device, dtype=x.dtype)
    return torch.full((x.shape[0], 1), float(t), device=x.device, dtype=x.dtype)


@torch.no_grad()
def step1_destination_estimation(
    model,
    x: torch.Tensor,
    t: float,
    *,
    flat_time: bool = False,
) -> torch.Tensor:
    t_vec = _time_tensor(x, t, flat=flat_time)
    v = model(x, t_vec)
    return x + (1.0 - t) * v


@torch.no_grad()
def step2_destination_refinement_2d(
    x_hat: torch.Tensor,
    h_vec: torch.Tensor,
    y_value: float,
    sigma_n: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    h_vec = h_vec.view(1, 2)
    h_norm2 = float((h_vec * h_vec).sum())
    alpha = lam / (sigma_n**2 + lam * h_norm2 + 1e-12)
    resid = float(y_value) - (x_hat * h_vec).sum(dim=1, keepdim=True)
    prox_mean = x_hat + alpha * resid * h_vec

    eye = torch.eye(2, device=x_hat.device, dtype=x_hat.dtype)
    h_col = h_vec.view(2, 1)
    cov = torch.linalg.inv((1.0 / (lam + 1e-12)) * eye + (1.0 / (sigma_n**2)) * (h_col @ h_col.T))
    return prox_mean, cov


@torch.no_grad()
def step2_uncertainty_sampling_2d(prox_mean: torch.Tensor, cov: torch.Tensor, gamma: int) -> torch.Tensor:
    eps = torch.randn_like(prox_mean)
    if gamma == 0:
        sigma_iso2 = torch.trace(cov) / 2.0
        return prox_mean + torch.sqrt(torch.clamp(sigma_iso2, min=0.0)) * eps

    jitter = 1e-8 * torch.eye(2, device=cov.device, dtype=cov.dtype)
    chol = torch.linalg.cholesky(cov + jitter)
    return prox_mean + eps @ chol.T


@torch.no_grad()
def step3_time_progression(x_tilde: torch.Tensor, t: float, dt: float) -> torch.Tensor:
    t_next = t + dt
    sigma_prog = max(1.0 - t_next, 0.0)
    z = torch.randn_like(x_tilde)
    return t_next * x_tilde + sigma_prog * z


@torch.no_grad()
def run_flower_trajectories_2d(
    model,
    n_traj: int,
    N: int,
    h_vec: torch.Tensor,
    y_value: float,
    sigma_n: float,
    *,
    gamma: int = 0,
    save_every: int = 50,
) -> dict[int, torch.Tensor]:
    model.eval()
    dt = 1.0 / float(N)
    x = torch.randn(n_traj, 2, device=h_vec.device)
    snapshots = {0: x.detach().cpu()}

    for k in range(N):
        t = k * dt
        x_hat = step1_destination_estimation(model, x, t, flat_time=False)
        sigma_r = (1.0 - t) / ((t * t + (1.0 - t) * (1.0 - t)) ** 0.5 + 1e-12)
        lam = sigma_r**2
        prox_mean, cov = step2_destination_refinement_2d(x_hat, h_vec, y_value, sigma_n, lam)
        x_tilde = step2_uncertainty_sampling_2d(prox_mean, cov, gamma=gamma)
        x = step3_time_progression(x_tilde, t, dt)

        step_idx = k + 1
        if step_idx % save_every == 0 or step_idx == N:
            snapshots[step_idx] = x.detach().cpu()

    return snapshots


@torch.no_grad()
def run_flower_step_by_step_2d(
    model,
    n_traj: int,
    N: int,
    h_vec: torch.Tensor,
    y_value: float,
    sigma_n: float,
    *,
    gamma: int,
    observe_steps: list[int],
) -> dict[int, dict[str, torch.Tensor | float]]:
    model.eval()
    dt = 1.0 / float(N)
    observe = set(int(s) for s in observe_steps)
    x = torch.randn(n_traj, 2, device=h_vec.device)
    details: dict[int, dict[str, torch.Tensor | float]] = {}

    for k in range(N):
        t = k * dt
        x_t = x
        x_hat = step1_destination_estimation(model, x_t, t, flat_time=False)
        sigma_r = (1.0 - t) / ((t * t + (1.0 - t) * (1.0 - t)) ** 0.5 + 1e-12)
        lam = sigma_r**2
        prox_mean, cov = step2_destination_refinement_2d(x_hat, h_vec, y_value, sigma_n, lam)
        x_tilde = step2_uncertainty_sampling_2d(prox_mean, cov, gamma=gamma)
        x_next = step3_time_progression(x_tilde, t, dt)

        step_idx = k + 1
        if step_idx in observe:
            details[step_idx] = {
                "t": float(t),
                "x_t": x_t.detach().cpu(),
                "x_hat": x_hat.detach().cpu(),
                "x_tilde": x_tilde.detach().cpu(),
                "x_next": x_next.detach().cpu(),
            }
        x = x_next

    return details


@torch.no_grad()
def cg_solve(
    b: torch.Tensor,
    x0: torch.Tensor,
    lam: float,
    sigma_noise: float,
    H,
    Ht,
    *,
    max_iter: int = 50,
    eps: float = 1e-5,
) -> torch.Tensor:
    def A(x: torch.Tensor) -> torch.Tensor:
        return Ht(H(x)) / (sigma_noise**2) + x / lam

    x = x0.clone()
    r = b - A(x)
    p = r.clone()
    r_norm = (r * r).sum(dim=(1, 2, 3), keepdim=True)

    for _ in range(max_iter):
        Ap = A(p)
        alpha = r_norm / ((p * Ap).sum(dim=(1, 2, 3), keepdim=True) + 1e-12)
        x = x + alpha * p
        r_new = r - alpha * Ap
        r_new_norm = (r_new * r_new).sum(dim=(1, 2, 3), keepdim=True)
        if torch.sqrt(r_new_norm).mean() < eps:
            break
        beta = r_new_norm / (r_norm + 1e-12)
        p = r_new + beta * p
        r, r_norm = r_new, r_new_norm
    return x


@torch.no_grad()
def run_flower_inverse_problem_steps(
    model,
    y: torch.Tensor,
    H,
    Ht,
    sigma_noise: float,
    N: int,
    observe_steps: list[int],
) -> tuple[torch.Tensor, dict[int, dict[str, torch.Tensor | float]]]:
    x = torch.randn_like(Ht(y))
    dt = 1.0 / float(N)
    details: dict[int, dict[str, torch.Tensor | float]] = {}
    observe = set(int(s) for s in observe_steps)

    for k in range(N):
        t = k * dt
        x_hat = step1_destination_estimation(model, x, t, flat_time=True)

        sigma_r = (1.0 - t) / ((t * t + (1.0 - t) * (1.0 - t)) ** 0.5 + 1e-12)
        lam = sigma_r**2
        b = Ht(y) / (sigma_noise**2) + x_hat / lam
        x_star = cg_solve(b, x_hat, lam, sigma_noise, H, Ht)
        x_next = step3_time_progression(x_star, t, dt)

        step_id = k + 1
        if step_id in observe:
            details[step_id] = {
                "t": float(t),
                "x_t": x.detach().cpu(),
                "x_hat": x_hat.detach().cpu(),
                "x_star": x_star.detach().cpu(),
                "x_next": x_next.detach().cpu(),
            }
        x = x_next

    return x.detach().cpu(), details
