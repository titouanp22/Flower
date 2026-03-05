from __future__ import annotations

import random

import numpy as np
import torch


class LocalSuperresolution:
    def __init__(self, sf: int):
        self.sf = int(sf)

    def H(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, 0 :: self.sf, 0 :: self.sf]

    def H_adj(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        out = torch.zeros(b, c, h * self.sf, w * self.sf, device=x.device, dtype=x.dtype)
        out[:, :, 0 :: self.sf, 0 :: self.sf] = x
        return out


class SingleImageDataset:
    def __init__(self, image):
        self.image = image

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int):
        if idx != 0:
            raise IndexError(idx)
        return self.image, 0


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def psnr_torch(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    mse = torch.mean((pred - target) ** 2)
    return float(10.0 * torch.log10(1.0 / (mse + eps)))
