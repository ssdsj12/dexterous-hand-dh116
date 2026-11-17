"""Owned-array admission for one new Tripod quiet interval, not grasp stability.

Only current-instance480Hz measurements are accepted. The original joint/force/
table guards still execute separately. No SDK handles, stepping or reset here.
"""
from copy import deepcopy
import math
import numpy as np
from scripts.audit_dh116_rod_quiet_window import IDENTITY,finite,summarize_window,extract
from scripts.prepare_dh116_tripod_owned_approach import QUIET_LIMITS

VERSION="dh116-tripod-quiet-eight-current-instance-v1"

class TripodQuietGate:
    def __init__(self,*,reset_nodes_m):
        self._reset=finite(reset_nodes_m,(41,3)).copy()
        self._rows=[];self._status="pending";self._receipt=None

    def require_ready(self):
        if self._status!="passed":raise RuntimeError("current-instance quiet interval has not passed")
        return deepcopy(self._receipt)

    def observe_exchange(self,exchange):
        return self.observe_measurement(extract({"exchange":exchange}))

    def observe_measurement(self,row):
        if self._status!="pending":raise RuntimeError("quiet budget consumed; cannot retry or extend")
        try:
            index=len(self._rows)
            if index==0 and (row["physics_step"]!=3 or row["rod_tick"]!=12 or row["policy_step"]!=0 or row["substep"]!=0):
                raise ValueError("first measured quiet sample must be SDK3/rod12/policy0/substep0")
            if index:
                first=self._rows[0]
                if any(row[k]!=first[k] for k in IDENTITY):raise ValueError("quiet identity changed")
            else:first=row
            if (row["physics_step"]!=3+index or row["rod_tick"]!=12*(index+1)
                    or row["policy_step"]!=0 or row["substep"]!=index
                    or not math.isclose(row["timestamp_s"],(3+index)/480,rel_tol=0,abs_tol=2e-7)):
                raise ValueError("quiet clock mismatch")
            if np.any(finite(row["action24"],(24,))) or row["guard_failure"] is not None:
                raise ValueError("original guard or nonzero action in quiet interval")
            finite(row["positions"],(41,3));finite(row["velocities"],(41,3))
            self._rows.append(deepcopy(row))
            receipt=dict(version=VERSION,status="pending",samples=len(self._rows),required_samples=8,
                identity={k:first[k] for k in IDENTITY},limits=deepcopy(QUIET_LIMITS),
                terminal_reason=None,teacher_admission=False,grasp_qualified=False)
            if len(self._rows)==8:
                metrics=summarize_window(self._rows,8)
                metrics["source_scope"]="current_instance_left_tripod_quiet_admission"
                drift=max(float(np.linalg.norm(r["positions"]-self._reset,axis=-1).max()) for r in self._rows)
                reasons=[]
                if metrics["max_node_speed_m_s"]>QUIET_LIMITS["max_node_speed_m_s"]:reasons.append("rod_speed")
                if drift>QUIET_LIMITS["max_node_drift_m"]:reasons.append("rod_reset_drift")
                for side,joints in metrics["hand_joint_speed"].items():
                    for name,value in joints.items():
                        limit=QUIET_LIMITS["max_joint_angular_speed_rad_s"] if value["unit"]=="rad/s" else QUIET_LIMITS["max_mount_linear_speed_m_s"]
                        if value["max_abs"]>limit:reasons.append(side+":"+name+":speed")
                if metrics["table_loaded_solver_substeps"]!=96:reasons.append("table_support_missing")
                if metrics["hand_loaded_solver_substeps"]!=0:reasons.append("unexpected_hand_contact")
                self._status="failed" if reasons else "passed"
                receipt.update(status=self._status,metrics=metrics,max_node_displacement_from_reset_m=drift,
                    failure_reasons=reasons,terminal_reason="tripod_quiet_admission_failed" if reasons else None,
                    sample_scope="8current480Hz endpoints;12substep contact capture each;not5secondHold")
            self._receipt=receipt
            return deepcopy(receipt)
        except Exception:
            self._status="failed"
            raise
