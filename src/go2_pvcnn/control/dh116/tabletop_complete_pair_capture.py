"""Opt-in owned raw solver contact packet retained at the original read boundary.

No solver advance, extra SDK read, geometry inference or source qualification.
The legacy projected pair fields remain byte-equivalent. Raw capacity/attempt/
overflow and aggregates are retained, rather than reconstructed from pair count.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace
from . import tabletop_deformx_pair_capture as legacy
from .tabletop_deformable_r0_replay import _digest

VERSION="dh116-complete-atomic-pair-capture-v1"
RAW_KEYS=frozenset(("substeps","mesh_count","capacity","attempted","overflow",
    "indices","vectors","node_forces_on_rod_world_n","mesh_wrenches_on_mesh_world"))

def sources():
    return {Path(p).name:hashlib.sha256(Path(p).read_bytes()).hexdigest()
        for p in (__file__,legacy.__file__)}

def project(raw,*,dt_s,physics_step,rod_tick,mesh_names):
    if not isinstance(raw,dict) or set(raw)!=RAW_KEYS:
        raise ValueError("exact complete atomic solver packet required")
    return legacy.read_pair_capture(SimpleNamespace(contact_pair_samples=lambda:raw),
        dt_s=dt_s,physics_step=physics_step,rod_tick=rod_tick,mesh_names=mesh_names)

def read_complete_pair_capture(solver,*,dt_s,physics_step,rod_tick,mesh_names):
    # One atomic snapshot only. Own it before projection or later solver reuse.
    raw=deepcopy(solver.contact_pair_samples())
    names=tuple(mesh_names)
    projected=project(raw,dt_s=dt_s,physics_step=physics_step,rod_tick=rod_tick,mesh_names=names)
    retained=dict(version=VERSION,dt_s=dt_s,physics_step=physics_step,rod_tick=rod_tick,
        mesh_names=list(names),raw=raw,raw_sha256=_digest(raw),producer_source_sha256=sources(),
        boundary_identity_source="actual_runtime_supplied_physics_and_rod_tick",
        hardware_measured=False,native_qualification=False)
    return dict(projected,complete_capture=retained)

def validate_complete_capture(capture):
    packet=capture.get("complete_capture")
    if not isinstance(packet,dict) or packet.get("version")!=VERSION:
        raise ValueError("retained complete atomic capture required; no historical reconstruction")
    expected_keys={"version","dt_s","physics_step","rod_tick","mesh_names","raw",
        "raw_sha256","producer_source_sha256","boundary_identity_source","hardware_measured","native_qualification"}
    if (set(packet)!=expected_keys or packet["producer_source_sha256"]!=sources()
        or packet["boundary_identity_source"]!="actual_runtime_supplied_physics_and_rod_tick"
        or packet["hardware_measured"] is not False or packet["native_qualification"] is not False
        or packet["physics_step"]!=capture.get("physics_step") or packet["rod_tick"]!=capture.get("rod_tick")):
        raise ValueError("complete capture source or owned endpoint differs")
    raw=packet["raw"]
    if _digest(raw)!=packet["raw_sha256"]:
        raise ValueError("retained raw capture bytes differ")
    projected=project(raw,dt_s=packet["dt_s"],physics_step=packet["physics_step"],
        rod_tick=packet["rod_tick"],mesh_names=packet["mesh_names"])
    if _digest(projected)!=_digest({k:v for k,v in capture.items() if k!="complete_capture"}):
        raise ValueError("retained raw packet and legacy pair projection differ")
    return dict(version=VERSION,complete=True,attempted=raw["attempted"],capacity=raw["capacity"],
        overflow=raw["overflow"],substeps=raw["substeps"],mesh_names=list(packet["mesh_names"]),
        raw_sha256=packet["raw_sha256"],producer_source_sha256=sources(),
        node_forces_on_rod_world_n=deepcopy(raw["node_forces_on_rod_world_n"]),
        mesh_wrenches_on_mesh_world=deepcopy(raw["mesh_wrenches_on_mesh_world"]),
        native_qualification=False,teacher_admission=False)
