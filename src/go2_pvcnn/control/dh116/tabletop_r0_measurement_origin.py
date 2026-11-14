"""Opt-in R0 measurement origins, separated from retained per-frame evidence.

No reward/control/physics change. Legacy observer stays default. Unknown values
stay unknown; every normalized source is reversible in the owned audit record.
"""
from copy import deepcopy
import hashlib
import math
from pathlib import Path
from .tabletop_deformable_r0_replay import _digest
from .tabletop_live_effective_pad_source import (
    CompletePairLiveR0Observer, VERSION as PAD_VERSION, IDENTITY)
VERSION="dh116-r0-measurement-origin-v1"
FIELDS=("left","right","table_contact","joint_limit")
FILES=("tabletop_r0_measurement_origin.py","tabletop_live_effective_pad_source.py",
       "tabletop_complete_pair_capture.py","tabletop_deformable_r0_replay.py",
       "tabletop_deformable_pad_region.py","tabletop_effective_pad.py",
       "tabletop_task.py","linear_coupling.py")

def _implementation():
    root=Path(__file__).parent
    return {n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in FILES}

class MeasurementOrigin:
    """One actual reset epoch and one immutable measurement origin per field."""
    def __init__(self,reset_clock):
        c=deepcopy(reset_clock)
        if (any(k not in c for k in IDENTITY) or c.get("clock_valid") is not True
            or type(c.get("physics_step")) is not int or c["physics_step"]<0
            or c.get("rod_tick")!=0 or type(c.get("timestamp_s")) not in (int,float)
            or not math.isfinite(c["timestamp_s"])):
            raise ValueError("actual completed reset clock required")
        self.reset_clock=c
        self.implementation=_implementation()
        self.origins={}
        self._binding=None

    def adapt(self,exchange,measurements,diagnostics):
        c=exchange["action_readback"]["post"]["identity"][0]
        e=diagnostics["complete_effective_pad_source"]
        delta=c["physics_step"]-self.reset_clock["physics_step"]
        if (any(c.get(k)!=self.reset_clock[k] for k in IDENTITY)
            or c.get("clock_valid") is not True or type(delta) is not int or delta<1
            or c["rod_tick"]!=delta*12
            or not math.isclose(c["timestamp_s"]-self.reset_clock["timestamp_s"],delta/480,abs_tol=1e-8,rel_tol=0)
            or c!=e["identity"] or e["version"]!=PAD_VERSION):
            raise ValueError("measurement evidence belongs to another reset/clock/version")
        hands=exchange["action_readback"]["post"]["hands"]
        identity_fields=("joint_names","prim_paths","row_indices","instance_object_id","view_object_id",
                         "dof_types","measured_source","active_indices","units_by_name")
        binding=dict(reset=self.reset_clock,implementation=self.implementation,
            mesh_bindings=exchange["rod"]["mesh_bindings"],rod_source=exchange["rod"]["source"],
            producer_source_sha256=e["producer_source_sha256"],
            hand_sources={side:{k:deepcopy(hands[side][k]) for k in identity_fields} for side in ("left","right")})
        binding_sha=_digest(binding)
        if self._binding is not None and self._binding!=binding_sha:
            raise ValueError("measurement origin changed within reset epoch")
        expected={side:PAD_VERSION+":"+e["source_sha256"]+":"+side for side in ("left","right")}
        expected["table_contact"]="validated_solver_pair_capture:"+_digest(exchange["rod"])+":any_table_SDF_pair_not_PhysX_hand_table"
        expected["joint_limit"]="actual_PhysX_q_original_joint_safety_failure_0.02rad_guard:"+_digest(exchange["action_readback"]["post"])
        result=deepcopy(measurements);samples={}
        for name in FIELDS:
            value=measurements[name]
            if set(value)!={"value","source"}:raise ValueError("explicit value/source required")
            if value["value"] is None:
                if value["source"] is not None:raise ValueError("unknown cannot acquire valid origin")
                origin=None
            else:
                if value["source"]!=expected[name]:
                    raise ValueError("sample source family/hash differs from owned original evidence")
                if name in ("left","right") and value["value"]!=e["hands"][name]["value"]:
                    raise ValueError("contact value differs from original evidence")
                origin=VERSION+":"+_digest(dict(binding=binding_sha,measurement=name))
            result[name]["source"]=origin
            samples[name]=dict(original_source=value["source"],origin=origin)
        self._binding=binding_sha
        for name,sample in samples.items():
            if sample["origin"] is not None:self.origins[name]=sample["origin"]
        provenance=dict(version=VERSION,identity=deepcopy(c),origin_binding_sha256=binding_sha,
            raw_capture_sha256=e["raw_capture_sha256"],effective_pad_sample_sha256=e["source_sha256"],
            samples=samples,teacher_admission=False,native_qualification=False)
        provenance["provenance_sha256"]=_digest(provenance)
        return result,provenance

def restore_measurements(measurements,provenance):
    p=deepcopy(provenance);saved=p.pop("provenance_sha256",None)
    if saved!=_digest(p) or p["version"]!=VERSION or set(p["samples"])!=set(FIELDS):
        raise ValueError("measurement provenance digest/contract differs")
    out=deepcopy(measurements)
    for name,s in p["samples"].items():
        if out[name]["source"]!=s["origin"]:
            raise ValueError("measurement origin differs from retained provenance")
        out[name]["source"]=s["original_source"]
    return out

class OriginLiveR0Observer(CompletePairLiveR0Observer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.measurement_origin=None

    def complete_physical_reset(self,event,*,reset_approach):
        super().complete_physical_reset(event,reset_approach=reset_approach)
        try:
            self.measurement_origin=MeasurementOrigin(event["reset"]["after_clock"])
            contract=dict(version=VERSION,implementation=self.measurement_origin.implementation,
                fields=list(FIELDS),sample_evidence_retained=True,training_admission=False)
            self.initialization["measurement_origin_contract"]=contract
            self._reset_audit=deepcopy(self.initialization)
            return deepcopy(self.initialization)
        except Exception:
            self._state="failed"
            raise

    def _measurements(self,event,clock,auxiliary):
        m,d,na=super()._measurements(event,clock,auxiliary)
        if self.measurement_origin is None:raise RuntimeError("actual Reset required before measurement origin")
        m,p=self.measurement_origin.adapt(event["exchange"],m,d)
        d["measurement_origin_provenance"]=p
        return m,d,na
