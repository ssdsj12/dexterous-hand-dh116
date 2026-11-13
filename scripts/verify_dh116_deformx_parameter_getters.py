"""Gated zero-step CUDA readback check; no Isaac, dynamics, reward or PPO."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

from go2_pvcnn.control.dh116.tabletop_deformx_parameter_snapshot import validate_snapshot
from go2_pvcnn.control.dh116.tabletop_stereo_native import native_instance_lock
from scripts import verify_dh116_deformx_pair_capture_v2 as worker_base
from scripts.verify_dh116_deformx_pair_capture import write_new

WORKER_ENV = "DH116_DEFORMX_PARAMETER_WORKER_V1"
ENTRY = "scripts/verify_dh116_deformx_parameter_getters.py"
SCOPE = "zero_step_parameter_getter_only"
ARRAYS = sorted(["gravity_world_m_s2", "rest_lengths_m", "stored_k_stretch_n_m",
                 "rest_quaternion_norm_n_m", "stretch_ea_n", "bend_ei_n_m2", "inverse_masses_kg_inv"])

def verify_worker(output, extension, expected_sha):
    saved = worker_base.ENTRY, worker_base.WORKER_ENV
    try:
        worker_base.ENTRY, worker_base.WORKER_ENV = ENTRY, WORKER_ENV
        worker_base.verify_worker(output, extension, expected_sha)
    finally:
        worker_base.ENTRY, worker_base.WORKER_ENV = saved

def plain(packet):
    return {key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in packet.items()}

def validate_raw(raw):
    if (len(raw["snapshots"]) != 3 or raw["time_before"] != 0. or raw["time_after"] != 0.
            or type(raw["solver_last_error"]) is not int or raw["solver_last_error"] != 0
            or raw["mutated_owned_host_arrays"] != ARRAYS):
        raise ValueError("wrong snapshot count, clock, mutation or solver error")
    expected = {"contact_radius_m": .005, "contact_mu_static": .5, "contact_mu_kinetic": .4,
        "contact_v_slip_m_s": .001, "contact_kappa": 1.,
        "gravity_world_m_s2": [0,0,-9.81], "rest_lengths_m": [[.01,.02]],
        "stretch_ea_n": [[800,1200]], "bend_ei_n_m2": [[.018,.027]],
        "inverse_masses_kg_inv": [[1000,500,1/.003]]}
    for p in raw["snapshots"]:
        validate_snapshot(p, expected_num_envs=1, expected_nodes=3, expected_time_s=0.)
        for key, value in expected.items():
            if not np.allclose(p[key], value, rtol=2e-6, atol=1e-8):
                raise ValueError("native parameter differs from probe input: " + key)
    for key, shape in (("positions",(1,3,3)),("velocities",(1,3,3)),("quaternions",(1,3,4))):
        before, after = (np.asarray(raw[s][key], dtype=np.float64) for s in ("state_before","state_after"))
        if (before.shape != shape or after.shape != shape or not np.isfinite(before).all()
                or not np.array_equal(before,after)):
            raise ValueError("readback changed or lost solver state: " + key)
    if not np.allclose(raw["state_before"]["positions"], [[[0,0,1],[0,.01,1],[0,.03,1]]],atol=1e-8,rtol=0):
        raise ValueError("probe initial nodes differ")
    return {"native_qualification_scope": SCOPE, "snapshot_count": 3,
            "cuda_solver_steps": 0, "rod_substeps": 0, "isaac_physics_steps": 0, "ppo_updates": 0,
            "parameters_reconciled": True, "owned_copy_verified": True,
            "state_unchanged": True, "teacher_qualified": False, "hardware_measured": False}

def validate_artifacts(output):
    raw = json.loads((output/"raw_parameters.json").read_text())
    result = json.loads((output/"result.json").read_text())
    required = validate_raw(raw)
    if any(result.get(key) != value for key,value in required.items()):
        raise ValueError("stored result does not reconcile with raw getter evidence")

def run_probe(extension, expected_sha):
    sys.path.insert(0,str(extension.parent.parent))
    from ropesim import RopeBatch, _ropesim
    import torch
    loaded=Path(_ropesim.__file__).resolve(strict=True)
    if loaded != extension or hashlib.sha256(loaded.read_bytes()).hexdigest()!=expected_sha:
        raise ValueError("loaded solver binary differs from frozen getter candidate")
    nodes=np.array([[[0,0,1],[0,.01,1],[0,.03,1]]],np.float32)
    rod=RopeBatch.from_positions(nodes,density=.065,k_stretch=800.,k_bend=.018,
        max_h=1/5760,num_itr=8,fix_root_orientation=False,
        pinned=np.zeros(3,dtype=bool),straight_rest=True,deterministic=True)
    rod.set_gravity(np.array([0,0,-9.81],np.float32))
    rod.set_contact(radius=.005,mu_s=.5,mu_k=.4,v_slip=.001,kappa=1.)
    rod.set_masses(np.array([.001,.002,.003],np.float32))
    rod.set_stretch_stiffnesses(np.array([800,1200],np.float32))
    rod.set_bend_stiffnesses(np.array([.018,.027],np.float32))
    def state():
        rod.gather()
        torch.cuda.synchronize()
        copied={name:torch.as_tensor(getattr(rod,name)(),device="cuda").cpu().numpy().copy().tolist()
                for name in ("positions","velocities","quaternions")}
        torch.cuda.synchronize()
        return copied
    raw={"time_before":float(rod.time),"state_before":state(),"snapshots":[]}
    first=rod.parameter_snapshot()
    raw["snapshots"].append(plain(first))
    for key in ARRAYS:
        if not isinstance(first[key],np.ndarray) or not first[key].flags.owndata:
            raise ValueError("getter returned an aliased array: "+key)
        first[key][...]=0
    raw["mutated_owned_host_arrays"]=ARRAYS
    raw["snapshots"].append(plain(rod.parameter_snapshot()))
    # A third call also verifies repeated readback does not advance the solver.
    raw["snapshots"].append(plain(rod.parameter_snapshot()))
    raw.update(state_after=state(),time_after=float(rod.time),solver_last_error=int(rod.last_error))
    del first
    del rod
    torch.cuda.synchronize()
    return raw

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--extension",required=True,type=Path)
    p.add_argument("--extension-sha256",required=True)
    p.add_argument("--execute-native",action="store_true")
    a=p.parse_args()
    ext=a.extension.resolve(strict=True)
    if (ext.parent.name!="ropesim" or not ext.name.startswith("_ropesim.cpython-310-")
            or hashlib.sha256(ext.read_bytes()).hexdigest()!=a.extension_sha256):
        raise ValueError("frozen cp310 extension mismatch")
    plan={"scope":SCOPE,"extension_sha256":a.extension_sha256,
          "cuda_solver_steps":0,"rod_substeps":0,"isaac_physics_steps":0,"ppo_updates":0}
    if not a.execute_native:
        print(json.dumps({**plan,"native_execution_performed":False}));return
    verify_worker(a.output,ext,a.extension_sha256)
    with native_instance_lock():
        a.output.mkdir(parents=True,exist_ok=False)
        write_new(a.output/"plan.json",plan)
        raw=run_probe(ext,a.extension_sha256)
        # Preserve the native data before validating, including failed probes.
        write_new(a.output/"raw_parameters.json",raw)
        result=validate_raw(raw)
        write_new(a.output/"result.json",{**plan,**result,"native_execution_performed":True})

if __name__=="__main__":main()
