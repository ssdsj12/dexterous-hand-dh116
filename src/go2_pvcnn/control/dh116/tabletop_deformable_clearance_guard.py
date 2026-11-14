"""Opt-in post-SDK source-envelope clearance evidence and terminal guard.

This subclass leaves the frozen exchange and zero-action operation unchanged.
It measures after each SDK step, completes the matching rod/readback boundary,
then delivers a terminal physical row before any subsequent action. It does
not prove native cooked geometry, attachment, table contact or Teacher fitness.
"""
from copy import deepcopy
import hashlib
import math
from pathlib import Path

import numpy as np

from .tabletop_deformable_bimanual_step import DT,_owned
from .tabletop_deformable_physical_exchange import FlexiblePhysicalExchange
from .tabletop_deformable_wrist_projection import CLEARANCE_M

VERSION="dh116-post-step-source-clearance-guard-v1"
GEOMETRY_SOURCE="same_clock_physx_link_transforms_local_com_and_original_collision_envelopes"
SIDES=("left","right")


def _pins():
    root=Path(__file__).parent
    names=(Path(__file__).name,"tabletop_deformable_physical_exchange.py",
           "tabletop_deformable_live_geometry.py","tabletop_batch_geometry.py",
           "tabletop_deformable_wrist_projection.py")
    return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}


def _measure(values):
    values=_owned(values)
    if values.shape!=(1,2) or values.dtype.kind!="f" or not np.isfinite(values).all():
        raise ValueError("finite measured single-environment two-hand source clearance required")
    failed=[side for side,value in zip(SIDES,values[0]) if value<CLEARANCE_M]
    return values,failed


def validate_post_step_clearance(row):
    """Recheck one decoded durable exchange row; never steps or owns SDK objects."""
    packet=row["post_step_clearance"]
    post=row["action_readback"]["post"]["identity"]
    if (len(post)!=1 or packet["version"]!=VERSION or packet["floor_m"]!=.0021
            or CLEARANCE_M!=.0021 or packet["source"]!=GEOMETRY_SOURCE
            or packet["source_sha256"]!=_pins() or packet["identity"]!=post[0]
            or packet["identity"]["clock_valid"] is not True
            or packet["geometry_phase"]!="sdk_complete_rod_pending"
            or packet["physics_step"]!=row["physics_step"]
            or packet["physics_step"]!=post[0]["physics_step"]
            or packet["rod_source_physics_step"]!=row["action_readback"]["pre"]["identity"][0]["physics_step"]
            or packet["rod_source_physics_step"]+1!=packet["physics_step"]):
        raise ValueError("post-step source, implementation or same-boundary identity differs")
    stamp=packet["sdk_time_s"]
    if (type(stamp) not in (int,float) or not math.isfinite(stamp)
            or not math.isclose(stamp,packet["physics_step"]*DT,rel_tol=0,abs_tol=2e-7)):
        raise ValueError("actual post-step SDK time differs")
    values,failed=_measure(packet["clearance_m"])
    indices=np.asarray(packet["lowest_shape_index"])
    if indices.shape!=(1,) or indices.dtype.kind not in "iu" or np.any((indices<0)|(indices>=34)):
        raise ValueError("actual global lowest source shape index required")
    if (packet["failed_sides"]!=failed or packet["passed"] is not (not failed)
            or packet["minimum_clearance_m"]!=float(values.min())
            or packet["native_clearance_qualified"] is not False
            or packet["hand_table_contacts"] is not None or packet["teacher_admission"] is not False):
        raise ValueError("post-step clearance result or qualification scope differs")
    original=FlexiblePhysicalExchange._guard_failure(None,row["action_readback"]["post"]["hands"],
        row["rod"]["hand_wrenches_com_world"])
    reason=original or ("post_step_source_clearance:"+",".join(failed) if failed else None)
    if row["guard_failure"]!=reason:
        raise ValueError("terminal reason differs from original guards and measured source clearance")
    return dict(passed=not failed,failed_sides=failed,minimum_clearance_m=float(values.min()),
        floor_m=CLEARANCE_M,terminal_reason=reason,native_clearance_qualified=False,
        hand_table_contacts=None,teacher_admission=False)


class ClearanceGuardedExchange(FlexiblePhysicalExchange):
    """Explicit successor choice at episode construction; never auto-installed."""

    def __init__(self,*args,**kwargs):
        try:
            if CLEARANCE_M!=.0021 or not callable(kwargs.get("read_geometry_after_sdk")):
                raise ValueError("unchanged 2.1mm floor and explicit after-SDK geometry callback required")
            super().__init__(*args,**kwargs)
            self._pending_clearance=None
        finally:
            self=args=kwargs=None

    def _geometry(self,expected,*,previous_rod_clock=None):
        row=None
        try:
            row=super()._geometry(expected,previous_rod_clock=previous_rod_clock)
            if previous_rod_clock is not None:
                if row.get("source")!=GEOMETRY_SOURCE:
                    raise ValueError("actual source-envelope geometry reader required")
                values,failed=_measure(row["clearance_m"])
                indices=_owned(row["lowest_shape_index"])
                if indices.shape!=(1,) or indices.dtype.kind not in "iu" or np.any((indices<0)|(indices>=34)):
                    raise ValueError("actual lowest source shape index required")
                self._pending_clearance=dict(version=VERSION,source=GEOMETRY_SOURCE,
                    source_sha256=_pins(),geometry_phase=row["geometry_phase"],
                    physics_step=row["physics_step"],rod_source_physics_step=row["rod_source_physics_step"],
                    sdk_time_s=float(self._sim.current_time),clearance_m=values.tolist(),
                    lowest_shape_index=indices.tolist(),floor_m=CLEARANCE_M,
                    minimum_clearance_m=float(values.min()),failed_sides=failed,passed=not failed,
                    native_clearance_qualified=False,hand_table_contacts=None,teacher_admission=False)
            return row
        finally:
            self=row=previous_rod_clock=None

    def _step(self,action,substep):
        row=packet=None
        self._pending_clearance=None
        try:
            row=super()._step(action,substep)
            if self._pending_clearance is None:
                raise ValueError("completed SDK step has no measured source clearance")
            packet=deepcopy(self._pending_clearance)
            packet["identity"]=deepcopy(row["action_readback"]["post"]["identity"][0])
            row["post_step_clearance"]=packet
            if packet["failed_sides"] and row["guard_failure"] is None:
                row["guard_failure"]="post_step_source_clearance:"+",".join(packet["failed_sides"])
            validate_post_step_clearance(row)
            return row
        finally:
            self._pending_clearance=None
            self=action=row=packet=None
