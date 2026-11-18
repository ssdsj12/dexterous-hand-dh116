"""Owned-data observer attached after a live physical Reset transaction.

No SDK references, commands or physics advances are retained here. Native
execution is established by the caller and independent outer-process receipt;
CPU doubles and historical replay never acquire native qualification.
"""
from copy import deepcopy
import numpy as np
from .tabletop_deformable_r0_replay import PhysicalR0Diagnostic, _clock, _copy, _digest, SIDES
from .tabletop_r0_after_physical_reset import PostPhysicalResetR0ControlStep, reset_clock_matches

VERSION="dh116-live-post-reset-r0-observer-v1"

class LiveR0Observer(PhysicalR0Diagnostic):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        adapted=deepcopy(self._inputs["task"])
        adapted["backend"]["extension_sha256"]=self._inputs["backend_sha256"]
        self._reducer=PostPhysicalResetR0ControlStep(adapted,self._grasp,env_id=self._env,
            budget_config=self._inputs["budget_config"],
            diagnostic_coefficients=self._inputs["diagnostic_coefficients"])
        self.initialization=None

    def _reset(self,event):
        raise RuntimeError("use explicitly completed transaction, not replay callbacks")

    def complete_physical_reset(self,event,*,reset_approach):
        if self._state!="new":
            raise RuntimeError("one completed transaction per observer")
        try:
            receipt=event["reset"]
            clock=_clock(dict(receipt["after_clock"],policy_step=0,substep=0),self._env)
            if (event["kind"]!="reset" or receipt["status"]!="complete"
                    or clock["rod_tick"]!=0 or receipt["r0_reset_complete"] is not False):
                raise ValueError("completed actual Reset before R0 required")
            if receipt["new_sdk_steps"]!=0 or receipt["new_solver_steps"]!=0:
                raise ValueError("coupled Reset must not advance physics")
            reset_clock_matches(receipt["after_clock"],receipt["after_actual_clocks"]["rod"])
            if set(receipt["hands"])!=set(SIDES):
                raise ValueError("both actual joint reset readbacks required")
            for h in receipt["hands"].values():
                if any(np.asarray(h[k]).shape!=(1,17) or not np.isfinite(h[k]).all() for k in ("q","qd")):
                    raise ValueError("complete actual q/qd required")
            owners=receipt["owner_resets"]
            if set(owners)!={"controller","filter","contact","camera_history"}:
                raise ValueError("all existing owner reset outcomes required")
            for name,row in owners.items():
                if (row["owner"]!=name or row["episode_uid"]!=clock["episode_uid"]
                    or row["status"] not in ("cleared","not_present","unavailable")
                    or not isinstance(row["source"],str) or not row["source"]):
                    raise ValueError("same episode owner source/status required")
            boundary=self._boundary(event,clock,receipt["rod_reset"])
            init=self._reducer.initialize_completed_boundary(boundary,self._inputs["radius_sample"],
                                                            reset_approach=reset_approach)
            self.initialization=dict(version=VERSION,completed_transaction_observed=True,
                physical_transaction_sha256=_digest(receipt),initializer=init,
                owner_reset_outcomes=_copy(owners),original_six_hook_order_claimed=False,
                native_qualification=False,teacher_admission=False)
            self._reset_audit=_copy(self.initialization)
            self._last_clock,self._last_boundary=clock,boundary
            self._state="active"
            return _copy(self.initialization)
        except Exception:
            self._state="failed"
            raise

    def end(self,result):
        out=super().end(result)
        out["live_post_reset_initialization"]=_copy(self.initialization)
        # Original legacy hook-order qualification remains false. The new
        # post-coupled-reset contract has its own independently checked receipt.
        return out
