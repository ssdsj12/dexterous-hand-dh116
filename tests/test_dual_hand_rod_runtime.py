"""CPU fake solver around the unchanged real FreeRodRuntime; no native physics."""
import copy
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
PADS=("finger14_link","finger23_link","finger33_link","finger43_link","finger53_link")

def module():
    path=ROOT/"src/go2_pvcnn/control/dh116/tabletop_dual_hand_rod_runtime.py"
    assert path.exists(),"isolated dual-hand runtime missing"
    return importlib.import_module("go2_pvcnn.control.dh116.tabletop_dual_hand_rod_runtime")

class Grid:
    def as_args(self):
        return np.ones((2,2,2)),np.zeros(3),.01

class FakeRope:
    @classmethod
    def from_positions(cls,nodes,**kwargs):
        obj=cls();obj.nodes=nodes.copy();obj.num_envs=1
        obj.fixed_root_orientation=kwargs["fix_root_orientation"]
        obj.pinned=kwargs["pinned"].copy();obj.time=0.;obj.last_error=""
        obj.slots=[];obj.poses=[];obj.steps=0;obj.resets=0
        obj.num_itr=kwargs["num_itr"];obj.density=kwargs["density"]
        obj.force=np.zeros_like(nodes);obj.wrenches=np.zeros((1,0,6))
        return obj
    def set_gravity(self,*a):pass
    def set_contact(self,**k):pass
    def set_rope_contact(self,*a):pass
    def add_mesh(self,*args,**kwargs):
        slot=len(self.slots);self.slots.append((args,kwargs))
        self.wrenches=np.pad(self.wrenches,((0,0),(0,1),(0,0)))
        return slot
    def enable_contact_forces(self,*a):pass
    def inv_masses(self):
        from go2_pvcnn.control.dh116.tabletop_deformable_grasp_runtime import lumped_masses
        return 1/lumped_masses(self.nodes[0],self.density)
    def reset_envs(self,ids):
        self.resets+=1;self.force[:]=0;self.wrenches[:]=0
    def set_mesh_poses(self,slot,p,q,*,teleport):
        self.poses.append((slot,np.array(p,copy=True),np.array(q,copy=True),teleport))
    def positions(self):return self.nodes
    def velocities(self):return np.zeros_like(self.nodes)
    def contact_forces(self):return self.force
    def mesh_wrenches(self):return self.wrenches
    def step_free(self,dt,substeps):
        self.steps+=1;self.time+=dt
        self.force[:]=0;self.wrenches[:]=0
        self.force[0,0,2]=3.
        self.wrenches[0,0,2]=-1.
        self.wrenches[0,6,2]=-2.

def setup(monkeypatch,*,pairs=False):
    from go2_pvcnn.control.dh116.tabletop_deformable_grasp_runtime import FreeRodRuntime
    monkeypatch.setitem(sys.modules,"ropesim",SimpleNamespace(RopeBatch=FakeRope,
        sdf=SimpleNamespace(from_function=lambda *a:Grid())))
    cfg=json.loads((ROOT/"configs/dh116_tasks/flexible_pinch_candidate_v1.json").read_text())
    initial={n:(np.zeros(3),np.array([0,0,0,1.])) for n in PADS}
    base=FreeRodRuntime(cfg,[[0,0,.2],[.1,0,.2],[.2,0,.2]],{n:Grid() for n in PADS},
        initial,[0,0,0],[1,1,.1],episode_uid="construction")
    bindings={f"{side}.{name}":dict(side=side,body_name=name,body_index=i,
        prim_path=f"/World/{side}Hand/{name}",source_usd_sha256="a"*64,
        cooked_geometry_sha256="b"*64,sdf_sha256="c"*64)
        for side in ("left","right") for i,name in enumerate(PADS)}
    poses={f"{side}.{name}":(np.array([float(side=="right"),0,0]),np.array([0,0,0,1.]))
        for side in ("left","right") for name in PADS}
    centers={name:pose[0]+[0,1,0] for name,pose in poses.items()}
    if pairs:
        def capture():
            indices=np.array([[s,0,slot] for s in range(12) for slot in (0,6)],dtype=np.int64)
            vectors=np.zeros((24,12));vectors[:,5]=1;vectors[:,8]=np.tile([1.,2.],12)
            return dict(substeps=12,mesh_count=11,capacity=64,attempted=24,overflow=False,
                indices=indices,vectors=vectors,
                node_forces_on_rod_world_n=base.rope.force[0].copy(),
                mesh_wrenches_on_mesh_world=base.rope.wrenches[0].copy())
        base.rope.contact_pair_samples=capture
    return base,bindings,poses,centers

def runtime(monkeypatch,*,pairs=False):
    m=module();base,bindings,poses,centers=setup(monkeypatch,pairs=pairs)
    obj=m.DualHandRodRuntime(base,bindings=bindings,right_grids={n:Grid() for n in PADS},
        right_poses={n:poses["right."+n] for n in PADS},run_uuid="run-cpu",env_id=7)
    return obj,base,bindings,poses,centers

def reset(obj,poses,*,uid="episode-1",reset_id=1,step=2):
    return obj.reset(physics_step=step,episode_uid=uid,reset_id=reset_id,pad_poses=poses)

def test_composes_one_free_rod_and_keeps_original_mesh_slots_and_source_false(monkeypatch):
    obj,base,bindings,poses,centers=runtime(monkeypatch)
    assert base.names==list(PADS)+["table"]
    assert obj.mesh_names==tuple(["left."+n for n in PADS]+["table"]+["right."+n for n in PADS])
    assert len(base.rope.slots)==11 and not base.rope.pinned.any()
    state=reset(obj,poses)
    assert state["rod_tick"]==0 and state["contact_valid"] is False
    assert state["rod_mesh_contact_pairs"] is None
    assert state["native_qualification"] is state["source_qualification"] is state["teacher_admission"] is False
    assert {p[0] for p in base.rope.poses}==set(range(5))|set(range(6,11))
    assert all(p[3] for p in base.rope.poses)
    assert state["env_id"]==7 and state["run_uuid"]=="run-cpu"
    # Copy returned data; never expose mutable solver scratch.
    state["node_positions"][:]=100
    assert not (base.rope.nodes==100).any()

def test_one_step_routes_world_com_wrench_to_both_hands_without_table_or_sign_confusion(monkeypatch):
    obj,base,bindings,poses,centers=runtime(monkeypatch)
    reset(obj,poses);row=obj.step(poses,centers,physics_step=3)
    assert base.rope.steps==1 and row["rod_tick"]==12 and row["physics_step"]==3
    assert row["timestamp_s"]==pytest.approx(3/480)
    assert row["solver_time_s"]-row["solver_time_origin_s"]==pytest.approx(1/480)
    left,right=row["hand_wrenches_com_world"]["left"],row["hand_wrenches_com_world"]["right"]
    assert left["body_ids"]==right["body_ids"]==[0,1,2,3,4]
    np.testing.assert_allclose(left["forces"][0,0],[0,0,-1])
    np.testing.assert_allclose(right["forces"][0,0],[0,0,-2])
    np.testing.assert_allclose(left["torques"][0,0],[1,0,0])
    np.testing.assert_allclose(right["torques"][0,0],[2,0,0])
    assert left["is_global"] is right["is_global"] is True
    assert row["table_wrench_world"].shape==(6,)
    assert row["rod_mesh_contact_pairs"] is None and row["pair_capture_status"]=="unavailable"
    assert row["source_qualification"] is row["full_state371_bound"] is False
    packet=obj.read_rod_snapshot()
    assert packet["env_ids"]==[7] and packet["physics_step"]==3

def test_valid_atomic_pairs_bind_real_current_totals_and_distinct_sides(monkeypatch):
    obj,base,bindings,poses,centers=runtime(monkeypatch,pairs=True)
    reset(obj,poses);row=obj.step(poses,centers,physics_step=3)
    names={p["mesh_name"] for p in row["rod_mesh_contact_pairs"]["pairs"]}
    assert names=={"left.finger14_link","right.finger14_link"}
    assert row["pair_capture_status"]=="validated_solver_samples"
    assert row["contact_pair_qualification"] is False

@pytest.mark.parametrize("edit",["duplicate","overflow","wrong_substeps","stale_totals","wrong_mesh_count"])
def test_invalid_pairs_invalidate_step_without_inventing_rollback_or_allowing_next_step(monkeypatch,edit):
    obj,base,bindings,poses,centers=runtime(monkeypatch,pairs=True);reset(obj,poses)
    original=base.rope.contact_pair_samples
    def bad():
        p=original()
        if edit=="duplicate":p["indices"][1]=p["indices"][0]
        if edit=="overflow":p["overflow"]=True
        if edit=="wrong_substeps":p["substeps"]=13
        if edit=="stale_totals":p["node_forces_on_rod_world_n"]*=0
        if edit=="wrong_mesh_count":p["mesh_count"]=10
        return p
    base.rope.contact_pair_samples=bad
    with pytest.raises(ValueError):obj.step(poses,centers,physics_step=3)
    assert base.rope.steps==1 and obj.clock()["physics_step"]==3
    with pytest.raises(RuntimeError,match="reset"):obj.step(poses,centers,physics_step=4)
    with pytest.raises(RuntimeError,match="measured|valid"):obj.read_rod_snapshot()
    base.rope.contact_pair_samples=original
    reset(obj,poses,uid="episode-2",reset_id=2,step=3)
    assert obj.step(poses,centers,physics_step=4)["rod_tick"]==12

@pytest.mark.parametrize("step",[2,4,-1,True])
def test_clock_mismatch_never_advances_solver(monkeypatch,step):
    obj,base,bindings,poses,centers=runtime(monkeypatch);reset(obj,poses)
    with pytest.raises(ValueError):obj.step(poses,centers,physics_step=step)
    assert base.rope.steps==0

@pytest.mark.parametrize("edit",["missing","duplicate_body","crossed_side","wrong_link","bad_hash"])
def test_invalid_binding_fails_before_appending_any_mesh(monkeypatch,edit):
    m=module();base,bindings,poses,centers=setup(monkeypatch)
    if edit=="missing":bindings.pop("right.finger14_link")
    if edit=="duplicate_body":bindings["left.finger23_link"]["body_index"]=0
    if edit=="crossed_side":bindings["left.finger14_link"]["side"]="right"
    if edit=="wrong_link":bindings["right.finger14_link"]["prim_path"]="/World/rightHand/finger23_link"
    if edit=="bad_hash":bindings["right.finger14_link"]["sdf_sha256"]="0"*64
    with pytest.raises(ValueError):
        m.DualHandRodRuntime(base,bindings=bindings,right_grids={n:Grid() for n in PADS},
            right_poses={n:poses["right."+n] for n in PADS},run_uuid="run",env_id=0)
    assert len(base.rope.slots)==6

def test_reset_rejects_reused_episode_and_clears_previous_reactions(monkeypatch):
    obj,base,bindings,poses,centers=runtime(monkeypatch)
    reset(obj,poses);obj.step(poses,centers,physics_step=3)
    with pytest.raises(ValueError):reset(obj,poses,uid="episode-1",reset_id=2,step=3)
    state=reset(obj,poses,uid="episode-2",reset_id=2,step=3)
    assert state["rod_tick"]==0 and state["named_wrenches"] is None
    with pytest.raises(RuntimeError):obj.read_rod_snapshot()
    with pytest.raises(ValueError):reset(obj,poses,uid="episode-1",reset_id=3,step=3)

def test_bad_pose_or_com_never_advances_solver(monkeypatch):
    obj,base,bindings,poses,centers=runtime(monkeypatch);reset(obj,poses)
    bad=copy.deepcopy(poses);bad["right.finger14_link"][1][:]=0
    with pytest.raises(ValueError):obj.step(bad,centers,physics_step=3)
    bad_com=copy.deepcopy(centers);bad_com["left.finger14_link"][:]=np.nan
    with pytest.raises(ValueError):obj.step(poses,bad_com,physics_step=3)
    assert base.rope.steps==0

def test_composer_submission_is_one_exclusive_world_frame_write_per_hand(monkeypatch):
    m=module();obj,base,bindings,poses,centers=runtime(monkeypatch);reset(obj,poses)
    row=obj.step(poses,centers,physics_step=3)
    events=[]
    class Composer:
        def __init__(self,side):self.side=side
        def reset(self):events.append((self.side,"reset"))
        def set_forces_and_torques(self,**kwargs):events.append((self.side,kwargs))
    m.write_composer_reactions({s:Composer(s) for s in ("left","right")},
        row["hand_wrenches_com_world"],to_tensor=lambda a:np.array(a,copy=True))
    assert [e[0] for e in events]==["left","left","right","right"]
    assert events[1][1]["is_global"] is events[3][1]["is_global"] is True
    assert events[1][1]["body_ids"]==[0,1,2,3,4]


def test_exclusive_composition_cannot_append_second_right_hand(monkeypatch):
    m=module();obj,base,bindings,poses,centers=runtime(monkeypatch)
    with pytest.raises(ValueError,match="fresh"):
        m.DualHandRodRuntime(base,bindings=bindings,right_grids={n:Grid() for n in PADS},
            right_poses={n:poses["right."+n] for n in PADS},run_uuid="other",env_id=0)
    assert len(base.rope.slots)==11

def test_required_pair_capture_refuses_missing_api_before_solver_step(monkeypatch):
    m=module();base,bindings,poses,centers=setup(monkeypatch)
    obj=m.DualHandRodRuntime(base,bindings=bindings,right_grids={n:Grid() for n in PADS},
        right_poses={n:poses["right."+n] for n in PADS},run_uuid="run",env_id=0,require_pairs=True)
    reset(obj,poses)
    with pytest.raises(ValueError,match="capture unavailable"):obj.step(poses,centers,physics_step=3)
    assert base.rope.steps==0

def test_solver_fault_keeps_supplied_sdk_step_but_marks_rod_tick_unknown_until_reset(monkeypatch):
    obj,base,bindings,poses,centers=runtime(monkeypatch);reset(obj,poses)
    def fail(dt,substeps):
        base.rope.time+=dt/2
        raise RuntimeError("CPU injected partial solver call")
    base.rope.step_free=fail
    with pytest.raises(RuntimeError,match="partial solver"):obj.step(poses,centers,physics_step=3)
    state=obj.clock()
    assert state["physics_step"]==3 and state["rod_tick"] is None
    assert state["clock_valid"] is False and state["solver_time_s"]==pytest.approx(1/960)

@pytest.mark.parametrize("fault",["clock","balance","nonfinite"])
def test_solver_output_guards_require_reset_after_failed_actual_step(monkeypatch,fault):
    obj,base,bindings,poses,centers=runtime(monkeypatch);reset(obj,poses)
    original=base.rope.step_free
    def bad(dt,substeps):
        original(dt,substeps)
        if fault=="clock":base.rope.time+=.1
        if fault=="balance":base.rope.wrenches[0,5,0]=5
        if fault=="nonfinite":base.rope.wrenches[0,6,3]=np.nan
    base.rope.step_free=bad
    with pytest.raises(ValueError):obj.step(poses,centers,physics_step=3)
    assert base.rope.steps==1 and obj.clock()["clock_valid"] is False

def test_composer_conversion_failure_cannot_partially_write_one_hand(monkeypatch):
    m=module();obj,base,bindings,poses,centers=runtime(monkeypatch);reset(obj,poses)
    row=obj.step(poses,centers,physics_step=3);calls=[]
    class Composer:
        def reset(self):calls.append("reset")
        def set_forces_and_torques(self,**k):calls.append("write")
    count=[0]
    def convert(a):
        count[0]+=1
        if count[0]==3:raise RuntimeError("CPU right conversion failed")
        return a
    with pytest.raises(RuntimeError,match="conversion"):
        m.write_composer_reactions({"left":Composer(),"right":Composer()},
            row["hand_wrenches_com_world"],to_tensor=convert)
    assert calls==[]


def test_provider_packet_remains_explicitly_unqualified_and_owns_its_arrays(monkeypatch):
    obj,base,bindings,poses,centers=runtime(monkeypatch);reset(obj,poses)
    obj.step(poses,centers,physics_step=3)
    packet=obj.read_rod_snapshot()
    assert packet["source_scope"]=="solver_readback_unqualified"
    assert packet["source_qualification"] is packet["teacher_admission"] is False
    packet["node_positions"][:]=100
    assert not (base.rope.nodes==100).any()


def test_pose_write_fault_records_already_advanced_sdk_without_advancing_rod(monkeypatch):
    obj,base,bindings,poses,centers=runtime(monkeypatch);reset(obj,poses)
    def fail(*args,**kwargs):raise RuntimeError("CPU mesh pose write failed")
    base.rope.set_mesh_poses=fail
    with pytest.raises(RuntimeError,match="pose write"):obj.step(poses,centers,physics_step=3)
    state=obj.clock()
    assert state["physics_step"]==3 and state["rod_tick"]==0
    assert state["clock_valid"] is False and base.rope.steps==0
