"""Teacher-only solver status for the independent supported17 runtime.

The original health module and its frozen historical source SHA are retained.
This adapter reads status only; it does not qualify contact, motion or reward.
"""
import hashlib
import math
from pathlib import Path

from . import tabletop_r0_health_separation as health
from .tabletop_bimanual_table_contacts import _clock


def capture_teacher_solver_health(runtime, *, backend, sim, expected_clock, grasp):
    """Read one owned, completed boundary without stepping either solver."""
    base = rope = boundary = None
    try:
        if grasp == "pinch":
            return health.capture_solver_health(
                runtime, backend=backend, sim=sim, expected_clock=expected_clock)
        from .tabletop_deformable_grasp_runtime import FreeRodRuntime
        from .tabletop_power_shared_rod import PowerSharedRodRuntime, PROXIMAL_POWER_MESH_NAMES

        expected = _clock(lambda: expected_clock)
        health._require(
            grasp in ("tripod", "power") and expected == expected_clock
            and type(runtime) is PowerSharedRodRuntime
            and type(runtime._base) is FreeRodRuntime,
            "actual runtime and exact completed supported rod clock required")
        base, rope = runtime._base, runtime.rope
        health._require(
            tuple(runtime.mesh_names) == PROXIMAL_POWER_MESH_NAMES
            and len(runtime.bindings) == 14 and rope is base.rope,
            "complete supported17 runtime and same base rod required")
        health._require(
            Path(backend.__file__).resolve() == health.BACKEND.resolve()
            and hashlib.sha256(health.BACKEND.read_bytes()).hexdigest() == health.BACKEND_SHA
            and type(rope) is backend.RopeBatch and rope.num_envs == 1 and rope.verts_per_rope == 41,
            "locked actual single free rod getter backend required")

        def boundary():
            health._require(
                runtime._base is base and runtime.rope is base.rope is rope
                and type(rope) is backend.RopeBatch
                and Path(backend.__file__).resolve() == health.BACKEND.resolve()
                and hashlib.sha256(health.BACKEND.read_bytes()).hexdigest() == health.BACKEND_SHA,
                "backend or actual rod owner changed during health read")
            actual = _clock(runtime.clock)
            health._require(
                actual == expected and type(sim.current_time_step_index) is int
                and sim.current_time_step_index == expected["physics_step"]
                and math.isclose(float(sim.current_time), expected["timestamp_s"], rel_tol=0, abs_tol=2e-7)
                and float(rope.time) == expected["solver_time_s"],
                "actual SDK/solver clock changed during health read")
            return actual, float(sim.current_time), float(rope.time)

        before = boundary()
        value = error = None
        try:
            value = rope.last_error
            if type(value) is not int:
                raise ValueError("actual native last_error is not an integer")
        except Exception as caught:
            value, error = None, type(caught).__name__ + ": " + str(caught)
        health._require(boundary() == before, "health read advanced or changed clock")
        return health._solver_status(
            expected, value, source_scope="native_getter", source_instance_id=id(rope),
            backend_sha256=health.BACKEND_SHA, capture_error=error)
    finally:
        runtime = backend = sim = base = rope = boundary = None
