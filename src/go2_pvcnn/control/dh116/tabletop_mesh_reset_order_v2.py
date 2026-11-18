"""Independent v2 zero-step eleven-mesh rebuild with instance numeric reader."""
from copy import deepcopy
import math
import numpy as np
from .tabletop_mesh_reset_order import (MESH_NAMES, PLAN_VERSION, _digest, _encoded, verify_plan_sources)
from .tabletop_mesh_observation_source import verify_runtime as verify_runtime_source
from .tabletop_dual_hand_reset_v2 import (DualHandResetTransactionV2, _no_step,
    failure_clock_start, failure_clock_progress)
from .tabletop_native_training_clock import read_clock_v2

VERSION = "dh116-zero-step-reset-mesh-rebuild-v2"

def _plan(plan,rod):
    try:
        p=deepcopy(plan);claimed=p.pop("sha256",None)
        if claimed!=_digest(p) or p["version"]!=PLAN_VERSION:
            raise ValueError("owned mesh plan changed")
        if tuple(x["name"] for x in p["entries"])!=MESH_NAMES or tuple(rod.mesh_names)!=MESH_NAMES:
            raise ValueError("all eleven original mesh slots required")
        if _digest(p["bindings"])!=_digest(rod.bindings) or p["cfg"]!=rod._base.cfg:
            raise ValueError("runtime source identity/material changed")
        for i,row in enumerate(p["entries"]):
            a=row["values"];o=row["origin"]
            if (a.dtype!=np.float32 or a.ndim!=3 or min(a.shape)<2 or not np.isfinite(a).all()
                    or o.dtype!=np.float32 or o.shape!=(3,) or not np.isfinite(o).all()
                    or type(row["spacing"]) not in (int,float)
                    or not math.isfinite(row["spacing"]) or row["spacing"]<=0
                    or any(type(row[k]) not in (int,float) or not math.isfinite(row[k])
                           or row[k]<0 for k in ("mu_s","mu_k"))
                    or (i==5)!=(row["table_pose"] is not None)):
                raise ValueError("finite unchanged grid/material/pose arguments required")
            if row["table_pose"] is not None:
                pos,quat=map(np.asarray,row["table_pose"])
                if (pos.shape!=(3,) or quat.shape!=(4,) or not np.isfinite(pos).all()
                        or not np.isfinite(quat).all() or abs(float(np.linalg.norm(quat))-1)>1e-6):
                    raise ValueError("finite unit table pose required before any mutation")
        verify_plan_sources(plan);verify_runtime_source(rod)
        return p
    finally:rod=None

class RebuiltMeshResetV2(DualHandResetTransactionV2):
    def __init__(self,seed,*,episode_uid,mesh_plan,clock_policy,state_reader=None):
        super().__init__(seed,episode_uid=episode_uid,clock_policy=clock_policy)
        from .tabletop_mesh_reset_numeric_state_v2 import capture_numeric_state_v2
        self.mesh_plan=deepcopy(mesh_plan)
        self.state_reader=capture_numeric_state_v2 if state_reader is None else state_reader
        if not callable(self.state_reader):
            raise ValueError("explicit callable state_reader required")
        self._state_reader_binding=self.state_reader
        self._capture_reader_binding=capture_numeric_state_v2
        self.numeric_reset_audit=None

    def _verify_reader(self):
        from .tabletop_mesh_reset_numeric_state_v2 import capture_numeric_state_v2
        if (self.state_reader is not self._state_reader_binding
                or capture_numeric_state_v2 is not self._capture_reader_binding):
            raise ValueError("instance state reader binding changed")
        return ("existing_CAI_tensor_clone_blocking_CPU_copy"
            if self.state_reader is capture_numeric_state_v2
            else "explicit_instance_state_reader_unqualified")

    def _verify(self):
        super()._verify()
        self._verify_reader()

    def _state(self,rope,sim,rod):
        from .tabletop_mesh_reset_numeric_state_v2 import validate_numeric_sample_v2
        try:
            self._verify_reader()
            if len(self.numeric_reset_audit["samples"])>=2:
                raise ValueError("at most two numeric reset samples")
            sample=self.state_reader(rope,sim,rod,clock_policy=self.clock_policy)
            self.numeric_reset_audit["samples"].append(deepcopy(sample))
            return deepcopy(validate_numeric_sample_v2(sample,clock_policy=self.clock_policy))
        finally:
            rope=sim=rod=None

    def run(self,sim,rod,bindings,*,owner_resets):
        receipt=rope=before_state=after_state=None
        progress_start=None
        try:
            if self.status!="new":raise RuntimeError("reset transaction cannot be reused")
            self._verify()
            progress_start=failure_clock_start(sim,rod)
            plan=_plan(self.mesh_plan,rod)  # Fail before touching either hand.
            from .tabletop_mesh_reset_numeric_state_v2 import new_numeric_audit_v2
            self.numeric_reset_audit=new_numeric_audit_v2(clock_policy=self.clock_policy)
            self.numeric_reset_audit["state_reader"]=self._verify_reader()
            receipt=super().run(sim,rod,bindings,owner_resets=owner_resets)
            self.numeric_reset_audit["reset_epoch"]=receipt["reset_epoch"]
            before=read_clock_v2(sim,rod,policy=self.clock_policy)
            if before["rod"]["rod_tick"]!=0:raise ValueError("mesh rebuild is reset-only")
            rope=rod.rope
            before_state=self._state(rope,sim,rod)
            _no_step(before,read_clock_v2(sim,rod,policy=self.clock_policy))
            # Pair buffers/capacity survive clear_meshes in the pinned source;
            # any borrowed mesh/grid view must be considered invalid after this point.
            rod._needs_reset=True
            rope.clear_meshes()
            for index,row in enumerate(plan["entries"]):
                pos,quat=(row["table_pose"] if row["name"]=="table"
                          else receipt["pad_poses_xyzw"][row["name"]])
                slot=rope.add_mesh(row["values"],row["origin"],row["spacing"],
                    pos=np.asarray(pos,np.float32),quat=np.asarray(quat,np.float32),
                    mu_s=row["mu_s"],mu_k=row["mu_k"])
                if type(slot) is not int or slot!=index:raise ValueError("reconstructed collider slot identity differs")
            after_state=self._state(rope,sim,rod)
            mutation_after=read_clock_v2(sim,rod,policy=self.clock_policy)
            if rod._needs_reset is not True or mutation_after["rod"]["clock_valid"] is not False:
                raise ValueError("mutation region lost its poisoned clock state")
            # Only this expected local validity transition is normalized for comparison.
            # Keep the actual false state in the returned receipt.
            after=deepcopy(mutation_after)
            after["rod"]["clock_valid"]=before["rod"]["clock_valid"]
            _no_step(before,after)
            if after!=before:raise ValueError("full reset identity changed during mesh reconstruction")
            if _digest(before_state)!=_digest(after_state):
                raise ValueError("mesh-only reset changed rod nodes/velocity/directors/parameters")
            if np.asarray(rope.mesh_wrenches()).shape!=(1,11,6):
                raise ValueError("reconstructed actual eleven-mesh layout differs")
            # Scratch wrench values are not valid reset observations. Original reset
            # already zeroed both hand composers and clears exchange reactions.
            if _digest(plan["bindings"])!=_digest(rod.bindings) or plan["cfg"]!=rod._base.cfg:
                raise ValueError("reset source binding changed during reconstruction")
            rod._needs_reset=False
            receipt["mesh_reset"]=dict(version=VERSION,mesh_plan_sha256=self.mesh_plan["sha256"],
                before_clock=before,mutation_after_clock=mutation_after,
                after_clock=read_clock_v2(sim,rod,policy=self.clock_policy),mesh_names=list(MESH_NAMES),
                native_state_before=before_state,native_state_after=after_state,
                source_tree_sha256=plan["source_basis"]["source_tree_sha256"],
                sdk_clock_scope="explicit_clock_policy_v2",clock_policy_sha256=self.clock_policy_sha256,
                reset_epoch=receipt["reset_epoch"],
                native_current_pose_readback=None,source_derived_commit=True,
                pair_capacity_readback=None,pair_buffer_policy="preserved_by_locked_clear_add_source",
                contact_readback_valid=False,borrowed_mesh_views_invalidated=True,
                new_sdk_steps=0,new_solver_steps=0,native_qualification=False,teacher_admission=False)
            from .tabletop_mesh_reset_numeric_state_v2 import _finalize
            _finalize(self.numeric_reset_audit)
            if (self.numeric_reset_audit["complete_samples"]!=2
                    or self.numeric_reset_audit["byte_equal"] is not True):
                raise ValueError("two unchanged typed numeric samples required")
            self.numeric_reset_audit["new_sdk_steps"]=0
            self.numeric_reset_audit["new_solver_steps"]=0
            self.numeric_reset_audit["operation_completed"]=True
            receipt["numeric_reset_audit"]=deepcopy(self.numeric_reset_audit)
            return receipt
        except BaseException as error:
            if self.numeric_reset_audit is not None:
                progress=failure_clock_progress(progress_start,sim,rod)
                self.numeric_reset_audit["failure_clock_progress"]=progress
                self.numeric_reset_audit["new_sdk_steps"]=progress["observed_sdk_steps"]
                self.numeric_reset_audit["new_solver_steps"]=None
                self.numeric_reset_audit["operation_completed"]=False
                from .tabletop_mesh_reset_numeric_state_v2 import _finalize
                self.numeric_reset_audit["operation_error"]=type(error).__name__+": "+str(error)
                try:
                    _finalize(self.numeric_reset_audit)
                except BaseException as cleanup:
                    self.cleanup_failures.append(dict(stage="numeric_failure_finalize",
                        error_type=type(cleanup).__name__,error=str(cleanup)))
            if self.status!="new":
                self.status="failed";rod._needs_reset=True;rod._base.last=None
            raise
        finally:sim=rod=bindings=owner_resets=receipt=rope=before_state=after_state=None
