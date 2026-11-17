"""Same-solver-substep raw load audit using the existing surface-sharing rule.

Raw solver contact normals are not qualified effective pad surfaces. No R0,
control, threshold, training reward or success flag is changed by this module.
"""
import math
import numpy as np
import torch
from .tabletop_surface_force_sharing import surface_targets, CONTRACT

VERSION="dh116-pair-substep-load-audit-v1"
PADS=("finger14_link","finger23_link","finger33_link","finger43_link","finger53_link")
SIDES=("left","right")

def vector(value):
    v=np.asarray(value,dtype=float)
    if v.shape!=(3,) or not np.isfinite(v).all():raise ValueError("finite measured world vector required")
    return v

def audit_capture(capture,*,solver_substeps,complete_capture):
    if complete_capture is not True:raise ValueError("unknown/incomplete capture cannot become measured zeros")
    if type(solver_substeps) is not int or solver_substeps!=12:
        raise ValueError("original twelve solver substeps required")
    dt=capture["substep_dt_s"]
    if type(dt) not in (int,float) or not math.isfinite(dt) or not math.isclose(dt,1/5760,rel_tol=1e-8):
        raise ValueError("original480Hz/12solver clock required")
    loads=np.zeros((12,2,5));normals=np.zeros((12,2,5,3));forces=np.zeros_like(normals)
    counts=np.zeros((12,2,5),int);table=np.zeros(12,bool);seen=set()
    for pair in capture["pairs"]:
        k=pair["substep"];node=pair["node_index"];name=pair["mesh_name"]
        if type(k) is not int or not 0<=k<12 or type(node) is not int or node<0:
            raise ValueError("valid integer solver step and material node required")
        key=(k,node,name)
        if key in seen:raise ValueError("duplicate captured pair would duplicate measured force")
        seen.add(key)
        n=vector(pair["outward_normal_world"]);f=vector(pair["force_on_rod_world_n"])
        t=vector(pair["tangential_force_on_rod_world_n"]);imp=vector(pair["impulse_on_rod_ns"])
        fn=pair["normal_force_on_rod_n"]
        if (type(fn) not in (int,float) or not math.isfinite(fn) or fn<0
                or abs(np.linalg.norm(n)-1)>2e-5):
            raise ValueError("nonnegative measured normal force and unit world normal")
        if (not np.allclose(f,n*fn+t,rtol=1e-5,atol=1e-6)
                or not np.allclose(imp,f*dt,rtol=1e-5,atol=1e-9)):
            raise ValueError("recorded normal/tangent/impulse accounting differs")
        if name=="table":
            table[k] |= fn>0
            continue
        parts=name.split(".")
        if len(parts)!=2 or parts[0] not in SIDES or parts[1] not in PADS:
            raise ValueError("unknown collider partner cannot be silently discarded")
        h=SIDES.index(parts[0]);d=PADS.index(parts[1])
        loads[k,h,d]+=fn;normals[k,h,d]+=fn*n;forces[k,h,d]+=f;counts[k,h,d]+=1
    # Reuse the original thresholds and pairing logic, not a new criterion.
    sharing=surface_targets(torch.as_tensor(loads),torch.as_tensor(normals))
    rows=[]
    for k in range(12):
        hands={}
        for h,name in enumerate(SIDES):
            bearing=loads[k,h]>.05
            hands[name]=dict(normal_load_n=loads[k,h].tolist(),
                normal_vectors_world=normals[k,h].tolist(),force_on_rod_world_n=forces[k,h].tolist(),
                pair_counts=counts[k,h].tolist(),bearing_digit_count=int(bearing.sum()),
                above_original_load_threshold=bearing.tolist(),
                raw_surface_paired=bool(sharing["paired"][k,h]),
                raw_surface_balanced=bool(sharing["balanced"][k,h]),
                raw_surface_groups=sharing["groups"][k,h].tolist(),
                normal_coherence=np.divide(np.linalg.norm(normals[k,h],axis=-1),loads[k,h],
                    out=np.zeros(5),where=loads[k,h]>0).tolist(),
                attributed_force_z_n=float(forces[k,h,:,2].sum()),
                effective_pad_validity=None)
        rows.append(dict(substep=k,hands=hands,table_loaded=bool(table[k])))
    return dict(version=VERSION,physics_step=capture["physics_step"],rod_tick=capture["rod_tick"],
        substep_dt_s=dt,substeps=rows,original_surface_contract=dict(CONTRACT),
        normal_source="raw_solver_obstacle_outward_normal_weighted_by_normal_load",
        effective_surface_qualification_missing=True,positive_effective_contact_qualified=False,
        r0_stability=None,material_slip=None,native_qualification=False,teacher_admission=False)

def bind_clock(capture,rod,clock,*,expected_physics_step):
    """Check native endpoint identity; do not invent per-substep clock readback."""
    if (type(expected_physics_step) is not int or expected_physics_step<3
            or clock.get("clock_valid") is not True):
        raise ValueError("qualified original endpoint clock required")
    for k in ("run_uuid","env_id","episode_uid","reset_id","rod_uid","physics_step","rod_tick"):
        if rod[k]!=clock[k]:raise ValueError("rod and physical readback clock/episode differ")
    if (capture["physics_step"]!=expected_physics_step or clock["physics_step"]!=expected_physics_step
            or capture["rod_tick"]!=clock["rod_tick"]
            or clock["rod_tick"]!=(expected_physics_step-2)*12):
        raise ValueError("same native endpoint and original2setup/12substep sequence required")
    t=clock["timestamp_s"]
    if type(t) not in (float,int) or not math.isfinite(t) or abs(t-expected_physics_step/480)>1e-8:
        raise ValueError("original480Hz physical endpoint timestamp required")
