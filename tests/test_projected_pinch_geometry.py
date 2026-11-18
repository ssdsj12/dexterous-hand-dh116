"""Original safe-z is accepted only after complete own-source reconstruction."""
from copy import deepcopy
import importlib,importlib.util
import numpy as np
import pytest
import torch
from test_wrist_projection_evidence import sample,api as evidence_api

def api():
    name="go2_pvcnn.control.dh116.tabletop_projected_pinch_geometry"
    assert importlib.util.find_spec(name) is not None,"projected target contract missing"
    return importlib.import_module(name)

def data():
    args,g,pre=sample()
    projection=evidence_api().record_projection(g,pre,**args)
    action=np.zeros((1,24))
    action[:,12:18]=(args["requested_palm_offsets_m"].numpy()-args["palm_offsets_m"].numpy()).reshape(1,6)*480/.01
    post={**pre,"physics_step":176,"rod_tick":2088,"timestamp_s":176/480}
    x=dict(action_readback=dict(pre=dict(identity=[pre]),post=dict(identity=[post]),
        request=dict(action24=action,requested_palm_offsets_m=args["requested_palm_offsets_m"].numpy(),
            requested_angles_rad=args["requested_angles_rad"].numpy()),
        submission=dict(projected_root_poses=projection["projected_root_poses"],
            projected_angles=projection["projected_angles"],projection_source=projection["projection_source"])),
        wrist_projection=projection)
    pair={"reset_seed":{"reset_root_pose_w":args["root_seed_w"].numpy()[0]}}
    actions=np.tile(action,(29,1))
    return pair,x,actions

def test_real_protection_not_silenced_or_mislabeled_as_source_error():
    m=api();p,x,a=data();r=m.validate_projected_target(p,x,a)
    assert r["safe_z_active"]==[True,True]
    assert r["original_projection_recomputed"]
    assert not r["fixed_unprojected_path_qualified"]
    assert not r["native_qualified"]

@pytest.mark.parametrize("bad",["action","seed","request","root","angles","source","clock","missing_inputs"])
def test_foreign_actions_inputs_or_sdk_targets_fail(bad):
    m=api();p,x,a=data();x=deepcopy(x);p=deepcopy(p)
    if bad=="action":x["action_readback"]["request"]["action24"][0,14]+=.1
    if bad=="seed":p["reset_seed"]["reset_root_pose_w"][0,2]+=.001
    if bad=="request":x["action_readback"]["request"]["requested_palm_offsets_m"][0,0,2]+=.001
    if bad=="root":x["action_readback"]["submission"]["projected_root_poses"][0,0,2]+=.001
    if bad=="angles":x["action_readback"]["submission"]["projected_angles"][0,0,0]=.1
    if bad=="source":x["action_readback"]["submission"]["projection_source"]="foreign"
    if bad=="clock":x["action_readback"]["post"]["identity"][0]["physics_step"]+=1
    if bad=="missing_inputs":del x["wrist_projection"]["input_evidence"]
    with pytest.raises((ValueError,KeyError)):m.validate_projected_target(p,x,a)

def test_cross_step_applied_offset_continuity_is_required():
    m=api();p,x,a=data()
    previous=deepcopy(x)
    previous["wrist_projection"]["palm_offsets_m"]=x["wrist_projection"]["input_evidence"]["inputs"]["palm_offsets_m"].copy()
    previous["wrist_projection"]["projected_angles"]=x["wrist_projection"]["input_evidence"]["inputs"]["angles_rad"].copy()
    previous["action_readback"]["post"]["identity"]=[deepcopy(x["action_readback"]["pre"]["identity"][0])]
    assert m.validate_continuity(previous,x)
    previous["wrist_projection"]["palm_offsets_m"][0,0,2]+=.001
    with pytest.raises(ValueError):m.validate_continuity(previous,x)
