"""Direct close/hold/lift stage contract for the native dual-Panda scene."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


SIDES = ("left", "right")
STAGES = ("close", "hold", "lift")


@dataclass(frozen=True)
class NativeObservation:
    contact_established: bool
    contact_duration_s: float
    hold_duration_s: float
    slip_ok: bool
    lift_displacement_m: float
    lift_duration_s: float
    whole_wire_clear: bool
    hand_active_targets: dict[str, np.ndarray]


@dataclass(frozen=True)
class NativeCommand:
    stage: str
    panda_delta_z: dict[str, float]
    hand_active_targets: dict[str, np.ndarray]
    stop: bool = False
    failure_reason: str | None = None


class NativeStageController:
    """Advance only through close, hold, and lift with explicit gates."""

    def __init__(self, start_stage: str = "close", lift_step_m: float = 0.002):
        if start_stage not in STAGES:
            raise ValueError("start stage must be close, hold, or lift")
        if not np.isfinite(lift_step_m) or lift_step_m <= 0.0:
            raise ValueError("lift step must be a positive finite value")
        self.stage = start_stage
        self.lift_step_m = float(lift_step_m)
        self.failed = False
        self.failure_reason: str | None = None
        self.completed = False
        self._history: list[dict[str, object]] = []

    def _targets(self, observation: NativeObservation) -> dict[str, np.ndarray]:
        if set(observation.hand_active_targets) != set(SIDES):
            raise ValueError("hand targets must contain distinct left and right sides")
        return {
            side: np.asarray(observation.hand_active_targets[side], dtype=np.float64).copy()
            for side in SIDES
        }

    def _command(self, observation: NativeObservation, *, stop=False, reason=None):
        delta = {side: 0.0 for side in SIDES}
        if self.stage == "lift" and not stop and not self.completed:
            delta = {side: self.lift_step_m for side in SIDES}
        command = NativeCommand(
            stage=self.stage,
            panda_delta_z=delta,
            hand_active_targets=self._targets(observation),
            stop=stop,
            failure_reason=reason,
        )
        self._history.append(
            {"stage": command.stage, "panda_delta_z": dict(command.panda_delta_z),
             "stop": command.stop, "failure_reason": command.failure_reason}
        )
        return command

    def step(self, observation: NativeObservation) -> NativeCommand:
        if self.failed or self.completed:
            return self._command(observation, stop=True, reason=self.failure_reason)
        if not observation.slip_ok:
            self.failed = True
            self.failure_reason = f"slip gate failed during {self.stage}"
            return self._command(observation, stop=True, reason=self.failure_reason)
        if self.stage == "close":
            if observation.contact_established and observation.contact_duration_s >= 0.1:
                self.stage = "hold"
        elif self.stage == "hold":
            if not observation.contact_established:
                self.failed = True
                self.failure_reason = "contact lost during hold"
                return self._command(observation, stop=True, reason=self.failure_reason)
            if observation.hold_duration_s >= 1.0:
                self.stage = "lift"
        elif self.stage == "lift":
            if (
                observation.lift_displacement_m >= 0.03
                and observation.lift_duration_s >= 1.0
                and observation.whole_wire_clear
            ):
                self.completed = True
        return self._command(observation)

    def stage_evidence(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "failed": self.failed,
            "completed": self.completed,
            "failure_reason": self.failure_reason,
            "history": list(self._history),
        }

