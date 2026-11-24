"""CPU checks of offline original-target replay and locked F4/SDF acquisition."""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from test_dh116_tensor_control import prepare
from go2_pvcnn.control.dh116 import tabletop_tensor_collector_replay as replay


def test_captured_full_command_and_readback_compared_independently():
    _,_,_,state,roots,ctrl,stamp=prepare()
    clear=torch.full((2,2),.003,dtype=torch.float64)
    action=torch.linspace(-.7,.7,48,dtype=torch.float64).reshape(2,24)
    state["base_link_pose_w"]=roots
    tx=ctrl.prepare(action,state,clear,roots,stamp=stamp)
    post=deepcopy(state)
    for i in (0,1):
        post["backend_targets"][:,i,ctrl.write_ids[i]]=tx["expected"][i].to(post["backend_targets"])
    meta=dict(roots=roots,write_ids=ctrl.write_ids,active_ids=ctrl.active,
        joint_names=tuple(ctrl.batch.sides[s].joint_names for s in replay.SIDES))
    row=dict(state_pre=state,action=action,palms=ctrl.palms,angles=ctrl.angles,
        pre_clearance=clear,projection=tx["projection"],command=tx["expected"],state_post=post)
    stats={}
    replay.original_targets(meta,row,stats)
    assert stats["issued_24d"]==0 and stats["actual_targets"]==0
    post["backend_targets"][1,0,ctrl.write_ids[0][0]]+=1.1e-7
    with pytest.raises(AssertionError,match="actual_targets"):
        replay.original_targets(meta,row,{})


def test_load_all_locked_hulls_and_pinch_grids_without_isaac():
    from scripts.solve_dh116_tripod_sdf_constrained_direction import SDF_DIR,SDF_SHA,checked_json
    summary=checked_json(SDF_DIR/"result.json",SDF_SHA)
    distal=sorted(k.split(".")[1] for k in summary["pads"] if k.startswith("left."))
    names=distal+["link"+str(i) for i in range(23-len(distal))]
    bindings={k:dict(capture_sha256=v["gpu_row_sha256"],sdf_sha256=v["sdf_file_sha256"])
        for k,v in summary["pads"].items()}
    o=replay.PinchOracle(dict(shapes=[(s,i,np.zeros((3,3))) for s in (0,1) for i in range(17)],
        body_names=(names,names),bindings=bindings,radius=.005))
    assert [len(s) for s in o.shapes]==[22,22]
    assert all(set(replay.REQUIRED)<=v.keys() for v in o.grids)

