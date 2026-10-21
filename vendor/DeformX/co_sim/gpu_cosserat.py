"""Explicit Torch-CUDA Cosserat rod approximation for hand-only rollouts.

This backend intentionally does not mutate PyElastica objects.  It provides a
small, device-resident gravity/damping integrator with the same dynamic state
keys as :class:`FreeRodEngine`.  ``backend='torch_cuda'`` is opt-in and fails
fast when CUDA is unavailable; it never silently falls back to CPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GpuRodConfig:
    n_elem: int = 16
    base_length: float = 0.5
    base_radius: float = 0.008
    density: float = 1000.0
    dt: float = 1.0e-3
    damping: float = 0.05
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)


class TorchCudaCosseratRod:
    """Device-resident explicit rod state with a stable gravity step."""

    backend = "torch_cuda"
    approximation = "explicit_torch_cosserat"

    def __init__(self, config: GpuRodConfig | None = None, *, device: str = "cuda") -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("torch is required for the torch_cuda backend") from exc
        if not torch.cuda.is_available() or not str(device).startswith("cuda"):
            raise RuntimeError("torch_cuda backend requires an available CUDA device")
        self.torch = torch
        self.cfg = config or GpuRodConfig()
        if self.cfg.n_elem < 2 or self.cfg.dt <= 0 or self.cfg.damping < 0:
            raise ValueError("invalid GPU rod configuration")
        self.device = torch.device(device)
        dtype = torch.float32
        x = torch.linspace(0.0, self.cfg.base_length, self.cfg.n_elem + 1,
                           device=self.device, dtype=dtype)
        self.rod_position = torch.zeros((3, self.cfg.n_elem + 1), device=self.device, dtype=dtype)
        self.rod_position[0] = x
        self.rod_velocity = torch.zeros_like(self.rod_position)
        self.rod_director = torch.eye(3, device=self.device, dtype=dtype).unsqueeze(-1).repeat(1, 1, self.cfg.n_elem)
        self.rod_omega = torch.zeros((3, self.cfg.n_elem), device=self.device, dtype=dtype)
        self.time = torch.zeros((), device=self.device, dtype=dtype)
        self._gravity = torch.tensor(self.cfg.gravity, device=self.device, dtype=dtype).view(3, 1)

    def step(self, steps: int = 1) -> dict[str, Any]:
        if int(steps) < 1:
            raise ValueError("steps must be positive")
        dt = self.cfg.dt
        for _ in range(int(steps)):
            self.rod_velocity = self.rod_velocity + dt * self._gravity
            self.rod_velocity = self.rod_velocity * max(0.0, 1.0 - self.cfg.damping * dt)
            self.rod_position = self.rod_position + dt * self.rod_velocity
            self.time = self.time + dt
        return self.snapshot(device=True)

    advance = step

    def snapshot(self, *, device: bool = True) -> dict[str, Any]:
        values = {"mode": "FREE", "active_constraints": (), "sim_time": self.time,
                  "rod_position": self.rod_position, "rod_velocity": self.rod_velocity,
                  "rod_director": self.rod_director, "rod_omega": self.rod_omega,
                  "backend": self.backend, "approximation": self.approximation}
        if device:
            return {k: (v.clone() if hasattr(v, "clone") else v) for k, v in values.items()}
        return {k: (v.detach().cpu().numpy().copy() if hasattr(v, "detach") else v) for k, v in values.items()}


def create_rod_backend(backend: str, **kwargs: Any) -> Any:
    """Create an explicit backend; CPU creation remains owned by FreeRodEngine."""
    if backend != "torch_cuda":
        raise ValueError("GPU factory only accepts backend='torch_cuda'; use FreeRodEngine for CPU")
    return TorchCudaCosseratRod(**kwargs)


__all__ = ["GpuRodConfig", "TorchCudaCosseratRod", "create_rod_backend"]
