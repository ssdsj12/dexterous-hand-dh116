"""Long-horizon (>=10s) 480Hz SDK clock reader using float64 accumulation envelope."""
from copy import deepcopy
import math


def _clock_long(sim, rod, *, max_step=100000):
    try:
        step, stamp, dt = sim.current_time_step_index, float(sim.current_time), float(sim.get_physics_dt())
        if (type(step) is not int or not 0 <= step <= max_step or not math.isfinite(stamp)
                or not math.isclose(dt, 1 / 480, rel_tol=0, abs_tol=1e-12)):
            raise ValueError(f"actual 480Hz SDK clock outside [0, {max_step}] scope")
        tolerance = max(1e-6, step * 1e-11)
        if abs(stamp - step * dt) > tolerance:
            raise ValueError("actual SDK clock exceeds float64 long-horizon accumulation envelope")
        row = deepcopy(rod.clock())
        if not math.isfinite(row["solver_time_s"]):
            raise ValueError("finite actual solver clock required")
        return dict(sdk_physics_step=step, sdk_timestamp_s=stamp, sdk_dt_s=dt, rod=row)
    finally:
        sim = rod = None
