"""Live owned-data negative contact evidence with retained atomic counters.

Original pad-region geometry is unchanged. Even an accepted region candidate
remains unknown positive contact until depth/effective-contact sources qualify.
This module never queries/advances an SDK and never qualifies training.
"""
from copy import deepcopy
import hashlib,math
from pathlib import Path
import numpy as np
from .tabletop_complete_pair_capture import validate_complete_capture
from .tabletop_r0_effective_pad_source import negative_side_result
from .tabletop_r0_live_observer import LiveR0Observer
from .tabletop_deformable_r0_replay import _digest,_measurement
from .tabletop_deformable_pad_region import bind_geometry,classify_pair
from .tabletop_pair_substep_load_audit import SIDES,PADS
from .tabletop_dual_hand_rod_runtime import MESH_NAMES
from .tabletop_effective_pad import _patch

VERSION="dh116-live-complete-effective-pad-negative-v1"
IDENTITY=("run_uuid","env_id","episode_uid","reset_id","rod_uid")

class LiveEffectivePadSource:
    def __init__(self,project,*,reset_clock):
        from scripts.audit_dh116_pinch_initial_pose import load_geometry
        from scripts.audit_dh116_bilateral_effective_pad_regions import validate_binding
        self.project=Path(project).resolve()
        c=deepcopy(reset_clock)
        if (c.get("clock_valid") is not True or c.get("rod_tick")!=0
            or type(c.get("physics_step")) is not int or c["physics_step"]<0
            or any(k not in c for k in IDENTITY)
            or type(c.get("timestamp_s")) not in (float,int) or not math.isfinite(c["timestamp_s"])):
            raise ValueError("explicit actual post-reset identity and zero rod tick required")
        self.reset_clock=c
        self._validate_binding=validate_binding
        self._geometry={s:load_geometry(self.project,side=s) for s in SIDES}
        self._atlas={s+"."+n:_patch(self._geometry[s]["source_colliders"][n]["vertices"],s,n)
            for s in SIDES for n in PADS}
        self.source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def capture(self,exchange):
        clock=exchange["action_readback"]["post"]["identity"][0]
        rod=exchange["rod"]
        capture=rod["rod_mesh_contact_pairs"]
        # No inference of completeness from the legacy derived pair count.
        complete=validate_complete_capture(capture)
        delta=clock["physics_step"]-self.reset_clock["physics_step"]
        if (any(clock.get(k)!=self.reset_clock[k] for k in IDENTITY)
            or clock.get("clock_valid") is not True or type(delta) is not int or delta<1
            or clock["rod_tick"]!=delta*12
            or not math.isclose(clock["timestamp_s"]-self.reset_clock["timestamp_s"],delta/480,abs_tol=1e-8,rel_tol=0)
            or any(rod.get(k)!=clock.get(k) for k in (*IDENTITY,"physics_step","rod_tick","timestamp_s"))
            or capture["physics_step"]!=clock["physics_step"] or capture["rod_tick"]!=clock["rod_tick"]):
            raise ValueError("same live reset epoch and physical/solver boundary required")
        if (rod.get("contact_valid") is not True or rod.get("pair_capture_status")!="validated_solver_samples"
            or tuple(complete["mesh_names"])!=MESH_NAMES or complete["substeps"]!=12):
            raise ValueError("complete original 11-mesh/12-substep source required")
        for observed,expected in (
            (complete["node_forces_on_rod_world_n"],np.asarray(rod["distributed_contact"])[0]),
            (complete["mesh_wrenches_on_mesh_world"],np.asarray([rod["named_wrenches"][k] for k in MESH_NAMES]))):
            if np.asarray(observed).shape!=np.asarray(expected).shape or not np.allclose(observed,expected,atol=3e-4,rtol=5e-5):
                raise ValueError("atomic packet and independent live solver aggregates differ")
        # The retained atomic validator already proves unique indices, unit
        # normals, force/impulse decomposition and aggregate reconciliation.
        # Keep the stricter original zero lower bound without recomputing
        # unused surface-sharing loads (no such load grants positive contact).
        if any(p["normal_force_on_rod_n"]<0 for p in capture["pairs"]):
            raise ValueError("nonnegative measured normal force required")
        geometry=exchange["closure_geometry"];raw=geometry["raw"]
        if _digest(raw)!=geometry["raw_sha256"]:
            raise ValueError("actual geometry raw bytes differ")
        transforms={}
        for s in SIDES:
            for pad in PADS:
                key=s+"."+pad;b=rod["mesh_bindings"][key]
                self._validate_binding(b,self._geometry[s],pad)
                transforms[key]=bind_geometry(raw,clock,key,b)
        classified={s:[] for s in SIDES};records=[]
        for pair in capture["pairs"]:
            key=pair["mesh_name"]
            if key=="table":continue
            region=classify_pair(pair,transforms[key],self._atlas[key])
            classified[key.split(".")[0]].append(region["accepted_by_original_patch"])
            records.append(dict(mesh_name=key,solver_substep=pair["substep"],node_index=pair["node_index"],region=region))
        source=_digest(dict(identity=clock,reset=self.reset_clock,raw_sha256=complete["raw_sha256"],
            raw_geometry_sha256=geometry["raw_sha256"],mesh_bindings=rod["mesh_bindings"],
            producer_source_sha256=complete["producer_source_sha256"],implementation_sha256=self.source_sha256))
        return dict(version=VERSION,identity=deepcopy(clock),source_sha256=source,
            hands={s:negative_side_result(classified[s],complete=True,rod_uid=clock["rod_uid"]) for s in SIDES},
            pairs=records,raw_capacity_overflow_retained=True,
            capture_counts={k:complete[k] for k in ("attempted","capacity","overflow","substeps")},
            producer_source_sha256=complete["producer_source_sha256"],
            raw_capture_sha256=complete["raw_sha256"],
            timing_scope="post_SDK_link_pose_held_during_12_solver_substeps",
            positive_effective_contact_qualified=False,contact_depth_qualified=False,
            material_slip_qualified=False,native_qualification=False,teacher_admission=False,
            new_physics_steps=0,ppo_updates=0)

class CompletePairLiveR0Observer(LiveR0Observer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.effective_pad_source=None

    def complete_physical_reset(self,event,*,reset_approach):
        out=super().complete_physical_reset(event,reset_approach=reset_approach)
        try:
            self.effective_pad_source=LiveEffectivePadSource(Path(__file__).resolve().parents[4],
                reset_clock=event["reset"]["after_clock"])
        except Exception:
            self._state="failed"
            raise
        return out

    def _measurements(self,event,clock,auxiliary):
        measurements,diagnostics,unavailable=super()._measurements(event,clock,auxiliary)
        if self.effective_pad_source is None:raise RuntimeError("actual Reset source not initialized")
        evidence=self.effective_pad_source.capture(event["exchange"])
        if evidence["identity"]!=clock:raise ValueError("same R0 physical identity required")
        for side in SIDES:
            hand=evidence["hands"][side]
            source=None if hand["value"] is None else VERSION+":"+evidence["source_sha256"]+":"+side
            measurements[side]=_measurement(hand["value"],source)
            unavailable.pop(side,None)
            if hand["reason"]:unavailable[side]=hand["reason"]
        diagnostics["complete_effective_pad_source"]=evidence
        return measurements,diagnostics,unavailable
