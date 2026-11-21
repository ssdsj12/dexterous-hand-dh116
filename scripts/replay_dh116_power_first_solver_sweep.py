"""Replay one archived SDK interval in the installed GPU solver only."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import argparse

import numpy as np
from ropesim import sdf
from go2_pvcnn.control.dh116.tabletop_deformable_grasp_runtime import FreeRodRuntime, load_candidate
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
from go2_pvcnn.control.dh116.tabletop_deformx_end_supported_scene import make_support_sdf_grids
from go2_pvcnn.control.dh116.tabletop_dual_hand_rod_runtime import PADS
from go2_pvcnn.control.dh116.tabletop_dual_hand_scene_preparation import prepare_scene
from go2_pvcnn.control.dh116.tabletop_power_shared_rod import SUPPORTED_POWER_MESH_NAMES
from go2_pvcnn.control.dh116.tabletop_material import hand_material_contract

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("case", type=Path, help="bound failed near-open acquisition")
parser.add_argument("--receipt-sha256", required=True)
args = parser.parse_args()
root = Path.cwd()
case = args.case.resolve(strict=True)
native = case / "native"
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(case / "outer.json") == args.receipt_sha256
outer=json.loads((case/"outer.json").read_bytes())
def archived(name):
    path=native/name
    assert sha(path)==outer["artifact_sha256"][name]
    return decode_owned(json.loads(path.read_bytes()))
reset=archived("power_reset_source_1.json")
post=archived("power_exchange_3.json")
read=reset["reset"]["before_hands"]
initial={}
for name in post["hand_link_poses"]:
    side,link=name.split(".")
    row=read[side]["body_poses_xyzw"][0,read[side]["body_names"].index(link)]
    initial[name]=(row[:3],row[3:])
cfg=load_candidate(root/"configs/dh116_tasks/flexible_pinch_candidate_v1.json", root)
prepared=prepare_scene(root)
scene=json.loads((root/"configs/dh116_tasks/flexible_power_supported_contact_v1.json").read_bytes())["support_scene"]
mounts=reset["reset"]["pad_poses_xyzw"]
nodes=reset["reset"]["rod_reset"]["node_positions"][0]
grids={}
for name,binding in post["rod"]["mesh_bindings"].items():
    if not name.endswith(".base_link"):
        path=Path(binding["sdf_path"])
        assert sha(path)==binding["sdf_sha256"]
        with np.load(path,allow_pickle=False) as a:
            grids[name]=sdf.SdfGrid(a["values"],a["origin"],float(a["spacing"]))
# Palms were registered after the SDK reset, use the actual locked bake operands.
argv=json.loads((case/"argv.json").read_bytes())
for side in ("left","right"):
    bake=Path(argv[argv.index("--"+side+"-bake")+1])
    name=side+".base_link"; path=bake/(name+".npz")
    assert sha(path)==post["rod"]["mesh_bindings"][name]["sdf_sha256"]
    with np.load(path,allow_pickle=False) as a:
        grids[name]=sdf.SdfGrid(a["values"],a["origin"],float(a["spacing"]))
for label,grid,box in zip(("support_0","support_1"),make_support_sdf_grids(scene),scene["supports"],strict=True):
    grids[label]=grid
    initial[label]=(box["center_m"],[0,0,0,1])
hand=hand_material_contract(cfg["hand"]["material_profile"])["hand_contact_material"]
result={}
for mode in ("overwrite_queued_reset_teleport","retain_first_teleport"):
    r=FreeRodRuntime(cfg,nodes,{name:grids["left."+name] for name in PADS},
        {name:initial["left."+name] for name in PADS},prepared["table"]["center_m"],prepared["table"]["size_m"],
        episode_uid="root-cause-replay")
    for i,name in enumerate(SUPPORTED_POWER_MESH_NAMES[6:11],6):
        assert r.rope.add_mesh(*grids[name].as_args(),pos=initial[name][0],quat=initial[name][1],
            mu_s=max(cfg["rod"]["mu_s"],hand["static_friction"]),mu_k=max(cfg["rod"]["mu_k"],hand["dynamic_friction"])) == i
    r.reset(physics_step=2,episode_uid="root-cause-replay:1",reset_id=1,
        pad_poses={name:mounts["left."+name] for name in PADS})
    for i,name in enumerate(SUPPORTED_POWER_MESH_NAMES[6:11],6):
        r.rope.set_mesh_poses(i,*mounts[name],teleport=True)
    for i,name in enumerate(SUPPORTED_POWER_MESH_NAMES[11:],11):
        pos,q=post["hand_link_poses"][name] if name in post["hand_link_poses"] else initial[name]
        assert r.rope.add_mesh(*grids[name].as_args(),pos=pos,quat=q,mu_s=cfg["table"]["mu_s"] if name.startswith("support") else max(cfg["rod"]["mu_s"],hand["static_friction"]),
            mu_k=cfg["table"]["mu_k"] if name.startswith("support") else max(cfg["rod"]["mu_k"],hand["dynamic_friction"])) == i
    r.rope.enable_contact_pair_capture(41*15*12)
    for i,name in enumerate(SUPPORTED_POWER_MESH_NAMES):
        if name in post["hand_link_poses"]:
            r.rope.set_mesh_poses(i,*post["hand_link_poses"][name],teleport=(mode=="retain_first_teleport"))
    r.rope.step_free(1/480,12)
    packet=r.rope.contact_pair_samples()
    labels=[SUPPORTED_POWER_MESH_NAMES[int(idx[2])] for idx in packet["indices"]]
    triples={tuple(int(n) for n in idx):v for idx,v in zip(packet["indices"],packet["vectors"],strict=True)}
    expected={(pair["substep"],pair["node_index"],pair["mesh_index"]):np.r_[pair["contact_point_world_m"],
        pair["outward_normal_world"],pair["force_on_rod_world_n"],pair["torque_on_mesh_about_origin_world_nm"]]
        for pair in post["rod"]["rod_mesh_contact_pairs"]["pairs"]}
    result[mode]=dict(substeps=packet["substeps"],count=packet["attempted"],histogram=dict(Counter(labels)),
        same_pair_identities=(triples.keys()==expected.keys()),
        max_abs_pair_vector_difference=max(float(np.max(np.abs(triples[key]-expected[key]))) for key in triples.keys()&expected.keys()),
        measured_hand_pair_ids=sorted(key for key in triples if SUPPORTED_POWER_MESH_NAMES[key[2]].startswith(("left.", "right."))))
    if mode=="overwrite_queued_reset_teleport":
        assert triples.keys()==expected.keys()
        assert result[mode]["max_abs_pair_vector_difference"]<1.e-5
    else:
        assert result[mode]["histogram"]=={"support_0":12,"support_1":12}
print(json.dumps(result,sort_keys=True))
