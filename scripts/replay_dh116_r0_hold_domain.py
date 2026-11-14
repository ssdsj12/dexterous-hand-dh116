"""Reuse the frozen R0 replay with an explicit opt-in necessary-hold domain."""
from copy import deepcopy
from pathlib import Path
import hashlib
import numpy as np

from scripts import replay_dh116_effective_pad_r0 as base
from go2_pvcnn.control.dh116.tabletop_r0_hold_necessary_domain import (
    NecessaryHoldR0ControlStep, contract, VERSION,
)

class RecordedHoldDomainR0(NecessaryHoldR0ControlStep):
    def __init__(self, *args, reset_approach, **kwargs):
        super().__init__(*args, **kwargs)
        self._recorded_reset_approach = deepcopy(reset_approach)

    def reset(self, reset_hooks, read_after_reset):
        return super().reset(reset_hooks, read_after_reset,
                             reset_approach=deepcopy(self._recorded_reset_approach))

def motion_measurements(exchanges):
    """Actual solver velocity, not a new stable=True predicate or qualification."""
    rows = []
    for x in exchanges:
        rod = x["rod"]
        v = np.asarray(rod["velocities"], dtype=float)
        p = np.asarray(rod["node_positions"], dtype=float)
        if v.shape != (1,41,3) or p.shape != v.shape or not np.isfinite(v).all() or not np.isfinite(p).all():
            raise ValueError("finite actual solver node position/velocity arrays required")
        speed = np.linalg.norm(v[0], axis=1)
        i = int(np.argmax(speed))
        rows.append(dict(identity=deepcopy(x["action_readback"]["post"]["identity"][0]),
            node_index=i, max_node_speed_m_s=float(speed[i]),
            mean_node_speed_m_s=float(speed.mean()),
            guard_failure=x["guard_failure"], stable=None,
            source="same-boundary solver node velocities; not rigid root velocity",
            stability_qualified=False))
    values = [r["max_node_speed_m_s"] for r in rows]
    peak = max(rows, key=lambda r:r["max_node_speed_m_s"])
    return dict(samples=rows, maximum_speed_sample=peak,
        speed_p50_p90_p95_m_s=np.percentile(values,[50,90,95]).tolist(),
        rigid_legacy_0_03_m_s_comparison_only=sum(v>=.03 for v in values),
        adopted_stability_threshold=None, stable_measurement=None,
        new_physics_steps=0, teacher_admission=False)

def run(result, emit):
    source = base.run(result, emit, negative_reducer_cls=RecordedHoldDomainR0)
    motion = motion_measurements(source.exchanges)
    for row in motion["samples"]:
        emit(dict(operation="actual_solver_motion", payload=row))
    result.update(hold_domain_contract=contract(), motion_diagnostics=motion,
        hold_domain_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        unchanged_stability_source=None, native_stability_qualified=False,
        teacher_admission=False)

if __name__ == "__main__":
    raise SystemExit(base.main(run_fn=run, version=VERSION))
