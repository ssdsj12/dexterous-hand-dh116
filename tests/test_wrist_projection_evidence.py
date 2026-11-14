"""Projection source provenance; CPU fixtures never count as native qualification."""
import importlib
from copy import deepcopy
import numpy as np
import pytest
import torch
from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import project_policy_wrist_targets as original
from go2_pvcnn.control.dh116.tabletop_batch_geometry import collision_clearance

def api():
    return importlib.import_module("go2_pvcnn.control.dh116.tabletop_wrist_projection_evidence")

def sample():
    clock=dict(run_uuid="test",env_id=0,episode_uid="test:1",reset_id=1,rod_uid="rod0",
        physics_step=175,rod_tick=2076,policy_step=21,substep=5,timestamp_s=175/480,
        solver_time_s=2076/5760,clock_valid=True)
    poses=np.zeros((1,2,23,7));poses[...,3]=1;poses[...,2]=.4
    names=["base_link"]+["body"+str(i) for i in range(22)]
    shapes=[(h,i,np.array([[0.,0.,-.09785],[.001,0.,-.0978]])) for h in range(2) for i in range(17)]
    clear,lowest=collision_clearance(poses,shapes)
    seed=poses[:,:,0].copy();seed[:,:,2]+=.001129
    palms=np.zeros((1,2,3));palms[:,:,2]=-.00117
    requested=palms.copy();requested[:,:,2]-=.000009375
    geom=dict(physics_step=175,rod_source_physics_step=175,geometry_phase="synchronized",
        root_poses_w=torch.tensor(poses[:,:,0]),clearance_m=torch.tensor(clear),
        lowest_shape_index=lowest,source="same_clock_physx_link_transforms_local_com_and_original_collision_envelopes",
        projection_raw=dict(identity=clock,link_poses_w=poses,shapes=shapes,
            body_names=[names,names],base_indices=[0,0],
            prim_paths=[["/World/leftHand/wrist_fixed"],["/World/rightHand/wrist_fixed"]],
            link_paths=[[["/World/"+side+"Hand/"+n for n in names]] for side in ("left","right")]))
    args=dict(root_seed_w=torch.tensor(seed),measured_base_w=geom["root_poses_w"],
        common_lift_offset_m=torch.zeros(1,dtype=torch.float64),measured_clearance_m=geom["clearance_m"],
        palm_offsets_m=torch.tensor(palms),requested_palm_offsets_m=torch.tensor(requested),
        angles_rad=torch.zeros((1,2,3),dtype=torch.float64),
        requested_angles_rad=torch.zeros((1,2,3),dtype=torch.float64),
        allowed=torch.ones(1,dtype=torch.bool),live=torch.ones(1,dtype=torch.bool))
    return args,geom,clock

def test_exact_original_targets_and_owned_complete_inputs():
    m=api();args,g,c=sample();out=m.record_projection(g,c,**args);expected=original(**args)
    for k,v in expected.items():
        if isinstance(v,torch.Tensor):assert torch.equal(out[k],v)
        else:assert out[k]==v
    checked=m.validate_record(out,c)
    assert checked["independent_source_clearance_recomputed"]
    assert checked["safe_z_active"]==[True,True]
    args["requested_palm_offsets_m"].fill_(9);g["projection_raw"]["link_poses_w"].fill(9)
    assert m.validate_record(out,c)==checked
    assert not checked["native_qualified"]

@pytest.mark.parametrize("bad",["clock","geometry_clock","pose","shape","base","clearance","safe_z","flag","source","requested","seed","missing"])
def test_input_geometry_or_result_tampering_fails(bad):
    m=api();a,g,c=sample();out=m.record_projection(g,c,**a);out=deepcopy(out)
    p=out["input_evidence"];geometry=p["geometry"]
    if bad=="clock":p["identity"]["episode_uid"]="other"
    if bad=="geometry_clock":geometry["projection_raw"]["identity"]["physics_step"]-=1
    if bad=="pose":geometry["projection_raw"]["link_poses_w"][0,0,0,2]+=.001
    if bad=="shape":geometry["projection_raw"]["shapes"][0][2][0,2]-=.001
    if bad=="base":geometry["projection_raw"]["base_indices"][0]=1
    if bad=="clearance":p["inputs"]["measured_clearance_m"][0,0]+=.001
    if bad=="safe_z":out["safe_z_offset_m"][0,0]-=.001
    if bad=="flag":out["safe_z_clipped"][0,0]=False
    if bad=="source":out["projection_source"]="invented"
    if bad=="requested":p["inputs"]["requested_palm_offsets_m"][0,0,2]-=.001
    if bad=="seed":p["inputs"]["root_seed_w"][0,0,2]+=.001
    if bad=="missing":del geometry["projection_raw"]["shapes"]
    with pytest.raises((ValueError,KeyError)):m.validate_record(out,c)

def test_bad_original_call_is_not_swallowed():
    m=api();a,g,c=sample();a["requested_angles_rad"][0,0,0]=9
    with pytest.raises(ValueError):m.record_projection(g,c,**a)

def test_geometry_diagnostics_keep_raw_when_exact_path_rejects():
    m=api()
    raw={"identity":{"physics_step":176},"owned_pose":np.arange(3.)}
    def failure(r,x):raise ValueError("exact path differs")
    packet=m.retain_failed_geometry(raw,{},failure)
    assert packet["raw"]["identity"]==raw["identity"]
    assert packet["terminal_reason"]=="closure_geometry_source_invalid"
    assert not packet["passed"] and not packet["teacher_admission"]
    raw["owned_pose"][:]=9
    assert np.array_equal(packet["raw"]["owned_pose"],np.arange(3.))

def test_successful_geometry_unchanged():
    m=api();packet={"passed":True,"terminal_reason":None}
    assert m.retain_failed_geometry({}, {},lambda r,x:packet) is packet

def test_retained_geometry_mixin_recomputes_failure_and_rejects_tampering():
    m=api()
    class Original:
        sources={"locked":"source"}
        def _evaluate(self,raw,x):
            if x["bad"]:raise ValueError("strict path mismatch")
            return {"version":"original","passed":True,"raw":deepcopy(raw)}
        def validate(self,packet,x):return self._evaluate(packet["raw"],x)
    class Kept(m.RetainedGeometryMixin,Original):pass
    g=Kept();raw={"identity":{"physics_step":176},"xyz":[1.,2.,3.]}
    packet=g._evaluate(raw,{"bad":True})
    assert packet["terminal_reason"]=="closure_geometry_source_invalid"
    assert g.validate(packet,{"bad":True})==packet
    packet["failure_reasons"]=["fabricated"]
    with pytest.raises(ValueError):g.validate(packet,{"bad":True})
    good=g._evaluate(raw,{"bad":False})
    assert good["version"]=="original"
    assert g.validate(good,{"bad":False})==good
