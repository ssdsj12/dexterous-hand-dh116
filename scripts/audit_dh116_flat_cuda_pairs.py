from pathlib import Path
import os,sys,json,hashlib
import numpy as np
import torch
import argparse
parser=argparse.ArgumentParser(description="Verify new native flat CUDA ABI against unchanged batched ABI")
parser.add_argument("--candidate",type=Path,required=True)
parser.add_argument("--fixture-attempt",type=Path,required=True)
parser.add_argument("--gpu",type=int,required=True)
parser.add_argument("--output",type=Path,required=True)
args=parser.parse_args()
if os.environ.get("CUDA_VISIBLE_DEVICES")!=str(args.gpu):raise ValueError("explicit physical GPU required")
ev=args.output
ev.mkdir(parents=True,exist_ok=False)
info=json.loads(args.candidate.read_text())
for name,h in info["source_inventory"].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h
info["binary"]=info["binary_path"]
binary=Path(info["binary"]);assert hashlib.sha256(binary.read_bytes()).hexdigest()==info["binary_sha256"]
sys.path.insert(0,str(binary.parents[1]))
import importlib.util
spec=importlib.util.spec_from_file_location("_ropesim",binary)
backend=importlib.util.module_from_spec(spec);spec.loader.exec_module(backend)
assert Path(backend.__file__).resolve()==binary.resolve()
from scripts.audit_dh116_deformx_shared_pairs import load_inputs,make_solver
from go2_pvcnn.control.dh116.tabletop_batched_pair_capture import compact_pair_capture
from go2_pvcnn.control.dh116.tabletop_tensor_pair_capture import flat_cuda_pair_capture
a=json.loads(args.fixture_attempt.read_text())["argv"]
arg=lambda k:a[a.index("--"+k)+1]
plan,reset=load_inputs("pinch",Path(arg("receipt")),arg("receipt-sha256"),Path(arg("smoke")),arg("smoke-sha256"),arg("smoke-ledger-sha256"))
capacity=41*len(plan["entries"])*12
solver=make_solver(backend,plan,reset,np.array([[0,0,0],[3,0,0]],dtype=np.float32),style="pinch",capacity=capacity)
saved=[];packets=[];errors=[]
for step in range(2):
    solver.step_free(1/480,12)
    old=solver.contact_pair_samples_batched_v1()
    ref=compact_pair_capture(old,dt_s=1/480,device="cuda:0")
    if step==0:
        with torch.cuda.stream(torch.cuda.Stream()):
            try:solver.contact_pair_samples_flat_cuda_v1()
            except ValueError:pass
            else:raise AssertionError("unsafe nondefault stream accepted")
    packet=solver.contact_pair_samples_flat_cuda_v1()
    actual=flat_cuda_pair_capture(packet,n=2,vertices=41,meshes=len(plan["entries"]),substeps=12,device="cuda:0",expected_owner_time_s=float(solver.time))
    for key in ("counts","env_ids","indices","vectors","normal_force"):
        torch.testing.assert_close(actual[key],ref[key],rtol=0,atol=0)
    for key in ("node_forces","mesh_wrenches"):
        delta=float((actual[key]-ref[key]).abs().max())
        errors.append(delta)
        torch.testing.assert_close(actual[key],ref[key],rtol=0,atol=1e-12)
    for prior,expected in zip(packets,saved):
        for key in prior:torch.testing.assert_close(prior[key],expected[key],rtol=0,atol=0)
    packets.append({k:v for k,v in packet.items() if isinstance(v,torch.Tensor)})
    assert actual["counts"][0]==0 and actual["counts"][1]>0
    try:solver.contact_pair_samples_flat_cuda_v1()
    except ValueError:pass
    else:raise AssertionError("flat read-once not enforced")
    saved.append({k:v.clone() for k,v in packet.items() if isinstance(v,torch.Tensor)})
    print("STEP",step,actual["counts"].tolist(),flush=True)
# Native-owned output tensors must remain unchanged after a later solver mutation.
for key in saved[0]:
    if key in packet:pass
assert not torch.equal(saved[0]["node_forces_gpu"],saved[1]["node_forces_gpu"])
solver.reset_envs(np.array([0],np.int32))
try:solver.contact_pair_samples_flat_cuda_v1()
except ValueError:pass
else:raise AssertionError("stale post-reset flat contact")
# All-empty batch and per-lane overflow keep original acceptance semantics.
from copy import deepcopy
empty_reset=deepcopy(reset)
empty_reset["rod_reset"]["node_positions"]=np.asarray(empty_reset["rod_reset"]["node_positions"])+np.array([2.,0.,0.])
empty=make_solver(backend,plan,empty_reset,np.array([[0,0,0],[3,0,0]],dtype=np.float32),style="pinch",capacity=capacity,empty_lane=False)
empty.step_free(1/480,12);empty_raw=empty.contact_pair_samples_flat_cuda_v1()
empty_capture=flat_cuda_pair_capture(empty_raw,n=2,vertices=41,meshes=len(plan["entries"]),substeps=12,device="cuda:0",expected_owner_time_s=float(empty.time))
assert empty_capture["counts"].sum()==0 and len(empty_capture["indices"])==0
overflow=make_solver(backend,plan,reset,np.array([[0,0,0],[3,0,0]],dtype=np.float32),style="pinch",capacity=1)
overflow.step_free(1/480,12)
try:flat_cuda_pair_capture(overflow.contact_pair_samples_flat_cuda_v1(),n=2,vertices=41,meshes=len(plan["entries"]),substeps=12,device="cuda:0",expected_owner_time_s=float(overflow.time))
except ValueError:pass
else:raise AssertionError("overflow accepted")
from go2_pvcnn.control.dh116.tabletop_tensor_rod_owner import TensorRodOwner
owned_solver=make_solver(backend,plan,reset,np.array([[0,0,0],[3,0,0]],dtype=np.float32),style="pinch",capacity=capacity)
owner=TensorRodOwner(owned_solver,run_uuid="flat-rod-probe",physics_step=2,episodes=("a","b"),
    mesh_names=tuple(x["name"] for x in plan["entries"]),device="cuda:0")
owner.advance(physics_step=3);owner.consume(physics_step=4)
owner.advance(physics_step=4);peer=torch.as_tensor(owned_solver.positions())[1].clone()
owner.reset_lanes((0,),episode_uids=("a2",))
assert torch.equal(torch.as_tensor(owned_solver.positions())[1],peer)
owner.consume(physics_step=5);owner.advance(physics_step=5);owner.consume(physics_step=6)
assert owner.ticks==(12,36)
from go2_pvcnn.control.dh116.tabletop_tensor_mesh_bridge import TensorMeshBridge
snapshot=owned_solver.mesh_state_snapshot()
M=len(plan["entries"]);pos=torch.tensor(snapshot["current_positions_world_m"][:,:M],device="cuda:0")
quat=torch.tensor(snapshot["current_quaternions_xyzw"][:,:M],device="cuda:0")
bindings=plan["bindings"]
bridge=TensorMeshBridge(tuple(x["name"] for x in plan["entries"]),bindings,pos,quat)
links=torch.zeros(2,2,23,7,device="cuda:0");links[...,6]=1
for key,row in bindings.items():
    j=bridge.names.index(key);s=0 if row["side"]=="left" else 1;b=row["body_index"]
    links[:,s,b,:3]=pos[:,j];links[:,s,b,3:]=quat[:,j]
bridge.submit(owned_solver,links)
actual=owned_solver.mesh_state_snapshot()
np.testing.assert_array_equal(actual["target_positions_world_m"][:,:M],pos.cpu().numpy())
result=dict(binary_sha256=info["binary_sha256"],steps=7,environments=2,tensor_owner_reset_continuation=True,gpu_all_mesh_submission=True,nondefault_stream_rejected=True,all_empty_batch=True,overflow_rejected=True,owned_outputs_stable=True,
    raw_indices_vectors_exact=True,reductions_max_abs_error=max(errors),reductions_atol=1e-12,empty_lane=True,contacting_lane=True,read_once=True,reset_rejected=True,
    isaac_steps=0,ppo_updates=0,device=os.environ["CUDA_VISIBLE_DEVICES"])
(ev/"flat-cuda-native-parity.json").write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
