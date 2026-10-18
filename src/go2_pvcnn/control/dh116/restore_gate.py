"""Fresh measured-state qualification after a stage reset, not grasp success.

The caller must validate provenance and inject physical states before using this
gate. It neither restores a solver nor grants qualifications to imported stages.
"""
import numpy as np


class RestoreGate:
    def __init__(self, entry_stage, *, minimum_loads_n, saved_anchors_local):
        if entry_stage not in ('approach', 'close', 'hold', 'lift'):
            raise ValueError('unsupported restore entry stage')
        loads = np.asarray(minimum_loads_n, dtype=float)
        anchors = np.asarray(saved_anchors_local, dtype=float)
        if (loads.shape != (4,) or anchors.shape != (4, 3)
                or not np.isfinite(loads).all() or not np.isfinite(anchors).all()
                or np.any(loads < .005) or np.any(loads > 3.)):
            raise ValueError('invalid saved restore thresholds or anchors')
        self.entry_stage = entry_stage
        self.minimum_loads = loads.copy()
        self.anchors = anchors.copy()
        self.last_time = None
        self.valid_since = None
        self.reason = None

    def update(self, time_s, *, safety_ok, active_constraints, settled,
               approach_ok, loads_n, valid_contacts, normals, anchors_local):
        loads = np.asarray(loads_n, dtype=float)
        valid = np.asarray(valid_contacts)
        normal = np.asarray(normals, dtype=float)
        anchors = np.asarray(anchors_local, dtype=float)
        if (type(time_s) not in (int, float) or not np.isfinite(time_s)
                or time_s < 0 or any(type(x) is not bool for x in (safety_ok, settled, approach_ok))
                or not isinstance(active_constraints, (list, tuple))
                or loads.shape != (4,) or valid.shape != (4,) or valid.dtype.kind != 'b'
                or normal.shape != (4, 3) or anchors.shape != (4, 3)
                or not all(np.isfinite(x).all() for x in (loads, normal, anchors))
                or np.any(loads < 0)
                or not np.allclose(np.linalg.norm(normal, axis=1), 1., atol=1e-6, rtol=0)):
            raise ValueError('invalid measured restore evidence')
        if ((self.last_time is None and time_s > .01+1e-12)
                or (self.last_time is not None and not 0 < time_s-self.last_time <= .01+1e-12)):
            raise ValueError('restore evidence requires fresh contiguous time')
        self.last_time = float(time_s)
        contact_stage = self.entry_stage in ('hold', 'lift')
        slip = float(np.linalg.norm(anchors-self.anchors, axis=1).max())
        if self.reason is None:
            if not safety_ok or active_constraints:
                self.reason = 'safety failure or forbidden wire constraint'
            elif contact_stage and slip > .002:
                self.reason = 'restored material slip exceeds 2 mm'
        if self.entry_stage == 'approach':
            qualified = settled
        elif self.entry_stage == 'close':
            qualified = settled and approach_ok
        else:
            qualified = bool(valid.all() and np.all(loads >= self.minimum_loads)
                             and normal[0] @ normal[1] <= -.5
                             and normal[2] @ normal[3] <= -.5)
        if self.reason is not None or not qualified:
            self.valid_since = None
        elif self.valid_since is None:
            self.valid_since = float(time_s)
        duration = 1. if contact_stage else .1
        ready = self.valid_since is not None and time_s-self.valid_since >= duration-1e-12
        return dict(entry_stage=self.entry_stage, ready=bool(ready),
                    aborted=self.reason is not None, reason=self.reason,
                    max_material_slip_m=slip)
