"""Low-overhead, phase-aware wall-clock instrumentation for DH116 runs."""
from __future__ import annotations

import json
import math
import os
import statistics
import time
from contextlib import contextmanager
from pathlib import Path

from .evidence_json import dumps


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


class PerfMeter:
    """Record step sections without changing simulation synchronization."""

    def __init__(self, path: Path, physics_dt: float, flush_every_steps: int = 1000):
        if not math.isfinite(physics_dt) or physics_dt <= 0:
            raise ValueError("physics_dt must be positive and finite")
        if isinstance(flush_every_steps, bool) or flush_every_steps < 1:
            raise ValueError("flush_every_steps must be positive")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("x", encoding="utf-8")
        self.physics_dt = float(physics_dt)
        self.flush_every_steps = int(flush_every_steps)
        self._step = None
        self._started_ns = None
        self._section_name = None
        self._section_started_ns = None
        self._sections: dict[str, float] = {}
        self._rows = 0
        self._closed = False
        self._stage_durations: dict[str, list[float]] = {}
        self._stage_steps: dict[str, int] = {}
        self._stage_completed: dict[str, int] = {}

    def begin_step(self, step: int, stage: str, sim_time: float) -> None:
        if self._closed:
            raise ValueError("cannot begin a step after close")
        if self._step is not None:
            raise RuntimeError("previous step is still active")
        if not isinstance(step, int) or step < 0:
            raise ValueError("step must be a nonnegative integer")
        if not isinstance(stage, str) or not stage:
            raise ValueError("stage must be a nonempty string")
        if not math.isfinite(sim_time):
            raise ValueError("sim_time must be finite")
        self._step = step
        self._stage = stage
        self._sim_time = float(sim_time)
        self._started_ns = time.perf_counter_ns()
        self._sections = {}

    @contextmanager
    def section(self, name: str):
        if self._step is None:
            raise RuntimeError("section requires an active step")
        if not isinstance(name, str) or not name:
            raise ValueError("section name must be nonempty")
        if self._section_name is not None:
            raise RuntimeError("nested performance sections are not supported")
        if name in self._sections:
            raise RuntimeError(f"duplicate performance section: {name}")
        self._section_name = name
        self._section_started_ns = time.perf_counter_ns()
        try:
            yield
        finally:
            elapsed = (time.perf_counter_ns() - self._section_started_ns) / 1e9
            self._sections[name] = elapsed
            self._section_name = None
            self._section_started_ns = None

    def record_section(self, name: str, elapsed_s: float) -> None:
        """Record a section measured by an existing runtime boundary."""
        if self._step is None:
            raise RuntimeError("section requires an active step")
        if name in self._sections:
            raise RuntimeError(f"duplicate performance section: {name}")
        if not math.isfinite(elapsed_s) or elapsed_s < 0:
            raise ValueError("section duration must be finite and nonnegative")
        self._sections[name] = float(elapsed_s)

    def end_step(self, completed: bool) -> None:
        if self._step is None:
            raise RuntimeError("no active step")
        if self._section_name is not None:
            raise RuntimeError("cannot end a step with an active section")
        elapsed = (time.perf_counter_ns() - self._started_ns) / 1e9
        row = {
            "step": self._step,
            "stage": self._stage,
            "sim_time_s": self._sim_time,
            "completed": bool(completed),
            "wall_s": elapsed,
            "sections": dict(self._sections),
        }
        self._file.write(dumps(row) + "\n")
        self._rows += 1
        self._stage_steps[self._stage] = self._stage_steps.get(self._stage, 0) + 1
        self._stage_completed[self._stage] = (
            self._stage_completed.get(self._stage, 0) + int(bool(completed))
        )
        self._stage_durations.setdefault(self._stage, []).append(elapsed)
        self._step = None
        self._started_ns = None
        if self._rows % self.flush_every_steps == 0:
            self._file.flush()

    def summary(self) -> dict[str, object]:
        stages: dict[str, object] = {}
        for stage, durations in self._stage_durations.items():
            total = sum(durations)
            steps = len(durations)
            stages[stage] = {
                "steps": steps,
                "completed_steps": self._stage_completed[stage],
                "wall_s": total,
                "mean_step_s": statistics.fmean(durations),
                "p50_step_s": _percentile(durations, 0.50),
                "p95_step_s": _percentile(durations, 0.95),
                "sps": steps / total if total > 0 else 0.0,
                "rtf": steps * self.physics_dt / total if total > 0 else 0.0,
            }
        return {
            "physics_dt_s": self.physics_dt,
            "steps": self._rows,
            "stages": stages,
        }

    def close(self) -> dict[str, object]:
        if self._closed:
            return self.summary()
        if self._step is not None:
            self.end_step(completed=False)
        summary = self.summary()
        self._file.flush()
        self._file.close()
        temporary = self.path.with_suffix(self.path.suffix + ".summary.tmp")
        temporary.write_text(dumps(summary) + "\n", encoding="utf-8")
        os.replace(temporary, self.path.with_name("perf_summary.json"))
        self._closed = True
        return summary

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
