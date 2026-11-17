"""Actual USD authoring CPU checks; no Kit/PhysX is started."""
import hashlib
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]

def mod():
    from go2_pvcnn.control.dh116 import tabletop_teacher_seed_initialization as m
    return m

def seed():
    from go2_pvcnn.control.dh116.linear_coupling import active_limits,expand
    lo,hi=active_limits();q=(lo+hi)/2
    nodes=np.c_[np.full(41,.43),np.linspace(-.25,.25,41),np.full(41,.35)]
    return dict(nodes=nodes.tolist(),joints=[expand(q).tolist(),expand(q).tolist()],
        roots=[[.5,.2,.45,1.,0.,0.,0.],[.5,-.2,.45,1.,0.,0.,0.]])

def setup_stage():
    from go2_pvcnn.control.dh116.tabletop_dual_hand_scene_preparation import prepare_scene,_load_offline_usd_schema
    _load_offline_usd_schema()
    from pxr import Usd,UsdGeom
    prepared=prepare_scene(ROOT)
    stage=Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage,1.)
    sources={}
    for side in ("left","right"):
        sources[side]=ROOT/prepared["hands"][side]["usd_path"]
        root=UsdGeom.Xform.Define(stage,"/World/envs/env_0/"+side+"Hand")
        root.GetPrim().GetReferences().AddReference(str(sources[side]))
    return stage,sources

def test_bad_coupling_is_rejected_before_stage_changes():
    m=mod();stage,sources=setup_stage();s=seed();s["joints"][0][2]+=.1
    before=stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError,match="coupling"):m.author_seed_before_reset(stage,sources,s)
    assert stage.GetRootLayer().ExportToString()==before

def test_actual_usd_world_fk_and_degree_state_match_both_hands_before_reset():
    m=mod();stage,sources=setup_stage();s=seed()
    from go2_pvcnn.control.dh116.linear_coupling import JOINTS,ACTIVE
    from go2_pvcnn.control.dh116.hand_point_kinematics import HandPointFK
    from pxr import Usd,UsdGeom,PhysxSchema,UsdPhysics
    sha={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in sources.items()}
    before={}
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.RevoluteJoint):
            j=UsdPhysics.RevoluteJoint(prim)
            before[str(prim.GetPath())]=(j.GetLowerLimitAttr().Get(),j.GetUpperLimitAttr().Get())
    result=m.author_seed_before_reset(stage,sources,s)
    assert result["physics_steps"]==0 and result["native_ready"] is False
    for hi,side in enumerate(("left","right")):
        q=np.array(s["joints"][hi])[[JOINTS.index(n) for n in ACTIVE]]
        fk=HandPointFK(sources[side],side,s["roots"][hi])
        expected={k.rsplit("/",1)[-1]:v for k,v in fk.poses(q).items()}
        for name,t in expected.items():
            prim=stage.GetPrimAtPath("/World/envs/env_0/"+side+"Hand/"+name)
            actual=np.array(UsdGeom.XformCache().GetLocalToWorldTransform(prim)).T
            assert np.allclose(actual,t,atol=2e-9,rtol=0),name
        joints={p.GetName():p for p in Usd.PrimRange(stage.GetPrimAtPath("/World/envs/env_0/"+side+"Hand"))
            if p.IsA(UsdPhysics.RevoluteJoint)}
        for name,qrad in zip(JOINTS,s["joints"][hi]):
            api=PhysxSchema.JointStateAPI(joints[name],"angular")
            assert api.GetPositionAttr().Get()==pytest.approx(np.degrees(qrad),abs=1e-5)
            assert api.GetVelocityAttr().Get()==0
        assert result["hands"][side]["body_count"]==17
    for path,bounds in before.items():
        j=UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(path))
        assert (j.GetLowerLimitAttr().Get(),j.GetUpperLimitAttr().Get())==bounds
    assert {k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in sources.items()}==sha

def test_prepared_layout_compensates_legacy_builder_offset_without_mutating_input():
    import copy
    m=mod();s=seed();layout=dict(hand_roots={},active_joints={},wire_center=np.zeros(3))
    original=copy.deepcopy(layout)
    adjusted=m.prepare_seed_layout(layout,s)
    for hi,side in enumerate(("left","right")):
        assert adjusted["hand_roots"][side][2]+.04==pytest.approx(s["roots"][hi][2])
    assert not layout["hand_roots"] and not layout["active_joints"]
    assert np.array_equal(layout["wire_center"],original["wire_center"])

def test_source_wiring_uses_one_pair_of_seeded_assets_before_first_reset():
    import ast
    from go2_pvcnn.control.dh116.tabletop_teacher_force_runtime import compile_force_runtime
    src=(ROOT/"scripts/train_dh116_gpu_teacher_ppo.py").read_text()
    tree=compile_force_runtime(src,hashlib.sha256(src.encode()).hexdigest())
    body=tree.body[0].body
    text=ast.unparse(tree)
    assert "spawn_force_scene(" in text
    assert "left_hands = selected_scene['hands'][0]" in text
    assert "right_hands = selected_scene['hands'][1]" in text
    # Unused old function remains byte-source-bound; no duplicate call sites.
    for n in ast.walk(tree):
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name):
            assert n.func.id!="_make_art"


def test_mount_joint_state_is_explicit_zero_before_first_step():
    m=mod();stage,sources=setup_stage();s=seed()
    from go2_pvcnn.control.dh116.tabletop_palm_mount import author_mount,JOINT_NAMES,ANGULAR_JOINT_NAMES
    from pxr import PhysxSchema
    for side in ("left","right"):
        prefix="/World/envs/env_0/"+side+"Hand"
        author_mount(stage,prefix,rotation=True)
        for name in JOINT_NAMES+ANGULAR_JOINT_NAMES:
            axis="linear" if name in JOINT_NAMES else "angular"
            api=PhysxSchema.JointStateAPI.Apply(stage.GetPrimAtPath(prefix+"/"+name),axis)
            api.CreatePositionAttr(.05);api.CreateVelocityAttr(.02)
    m.author_seed_before_reset(stage,sources,s)
    for side in ("left","right"):
        for name in JOINT_NAMES+ANGULAR_JOINT_NAMES:
            axis="linear" if name in JOINT_NAMES else "angular"
            api=PhysxSchema.JointStateAPI(stage.GetPrimAtPath(
                "/World/envs/env_0/"+side+"Hand/"+name),axis)
            assert api.GetPositionAttr().Get()==0
            assert api.GetVelocityAttr().Get()==0

def test_actual_setup_readback_precedes_any_post_reset_seed_writes():
    import ast
    from go2_pvcnn.control.dh116.tabletop_teacher_force_runtime import compile_force_runtime
    src=(ROOT/"scripts/train_dh116_gpu_teacher_ppo.py").read_text()
    tree=compile_force_runtime(src,hashlib.sha256(src.encode()).hexdigest())
    text=ast.unparse(tree)
    assert text.index("audit_seed_after_setup(")<text.index("cfg_rod =")
    assert text.index("force_hook.accept_initialization(")<text.index("seed_q =")

def test_setup_readback_rejects_drift_before_seed_can_hide_it():
    from types import SimpleNamespace as NS
    m=mod();s=seed()
    from go2_pvcnn.control.dh116.linear_coupling import JOINTS
    from go2_pvcnn.control.dh116.tabletop_palm_mount import JOINT_NAMES,ANGULAR_JOINT_NAMES
    hands=[]
    for hi in range(2):
        names=list(JOINTS)+list(JOINT_NAMES+ANGULAR_JOINT_NAMES)
        hands.append(NS(joint_names=names,body_names=["base_link"],
            data=NS(joint_pos=np.r_[s["joints"][hi],np.zeros(6)][None],joint_vel=np.zeros((1,17)),
            body_link_pose_w=np.array(s["roots"][hi])[None,None])))
    state=dict(left_hands=hands[0],right_hands=hands[1],
        env_origins_np=np.zeros((1,3)),sim=NS(current_time_step_index=2,current_time=2/480))
    assert m.audit_seed_after_setup(state,s)["passed"] is True
    hands[1].data.joint_pos[0,1]+=.001
    receipt=m.audit_seed_after_setup(state,s)
    assert receipt["passed"] is False
    assert receipt["hands"]["right"]["joint_max_error_rad"]==pytest.approx(.001)
    assert receipt["physics_step"]==2
    assert receipt["native_success"] is False

    # A correct base pose is insufficient when a carriage joint drifted.
    hands[1].data.joint_pos[0,1]-=.001
    hands[1].data.joint_pos[0,11]=.0003
    receipt=m.audit_seed_after_setup(state,s)
    assert receipt["passed"] is False
    assert receipt["hands"]["right"]["wrist_max_translation_error_m"]==pytest.approx(.0003)
    hands[1].data.joint_pos[0,11]=0
    hands[1].data.joint_vel[0,11:]=1.
    assert m.audit_seed_after_setup(state,s)["passed"] is False


def test_nonidentity_roots_and_environment_origin_match_full_fk():
    m=mod();stage,sources=setup_stage();s=seed()
    from scipy.spatial.transform import Rotation
    from go2_pvcnn.control.dh116.hand_point_kinematics import HandPointFK
    from go2_pvcnn.control.dh116.linear_coupling import JOINTS,ACTIVE
    from pxr import UsdGeom
    for hi,angles in enumerate(([.73,-.41,1.2],[-1.31,.52,-.84])):
        quat=Rotation.from_euler("xyz",angles).as_quat()
        s["roots"][hi][3:]=np.r_[quat[3],quat[:3]].tolist()
    origin=np.array([1.23,-3.4,.11])
    m.author_seed_before_reset(stage,sources,s,origin=origin)
    for hi,side in enumerate(("left","right")):
        pose=np.array(s["roots"][hi]);pose[:3]+=origin
        q=np.asarray(s["joints"][hi])[[JOINTS.index(n) for n in ACTIVE]]
        for path,wanted in HandPointFK(sources[side],side,pose).poses(q).items():
            prim=stage.GetPrimAtPath("/World/envs/env_0/"+side+"Hand/"+path.rsplit("/",1)[-1])
            actual=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim)).T
            assert np.allclose(actual,wanted,atol=2e-9,rtol=0)
