"""Explicit Cosserat backend probe used by the hand-only MPC collector.

The probe advances only the selected rod state.  No hand collider is attached,
so its report is deliberately non-qualifying until a full Isaac contact run
supplies measured loads, slip, and lift evidence.
"""
from __future__ import annotations

import numpy as np


class HandOnlyRodRuntime:
    def __init__(self, backend: str, *, device: str = "cpu") -> None:
        if backend not in {"cpu", "torch_cuda"}:
            raise ValueError("rod backend must be cpu or torch_cuda")
        self.backend = backend
        self.device = str(device)
        self.steps = 0
        self.engine = None
        self._initial_top_z = 0.0
        if backend == "torch_cuda":
            if not self.device.startswith("cuda"):
                raise RuntimeError("torch_cuda rod backend requires a CUDA device")
            try:
                from co_sim.gpu_cosserat import GpuRodConfig, TorchCudaCosseratRod
            except ImportError as exc:  # pragma: no cover - environment-specific
                raise RuntimeError("torch_cuda backend is unavailable") from exc
            self.engine = TorchCudaCosseratRod(
                GpuRodConfig(n_elem=40, base_length=.5, base_radius=.008,
                             density=1000., dt=1e-5, damping=1e-2),
                device=self.device,
            )
            self._initial_top_z = float(self.engine.rod_position[2].max().detach().cpu())

    def step(self) -> dict[str, object]:
        if self.engine is None:
            self.steps += 1
            return {"backend": "cpu", "state_device": "cpu", "steps": self.steps,
                    "wire_lift_m": 0.0, "physics_valid": False,
                    "contact_coupled": False}
        snapshot = self.engine.step(1, return_snapshot=True)
        self.steps += 1
        position = snapshot["rod_position"]
        lift = float(position[2].max().detach().cpu()) - self._initial_top_z
        return {"backend": "torch_cuda", "state_device": str(position.device),
                "steps": self.steps, "wire_lift_m": lift,
                "physics_valid": False, "contact_coupled": False}


__all__ = ["HandOnlyRodRuntime"]
