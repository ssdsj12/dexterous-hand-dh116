import torch
from pxr import Usd,UsdGeom,UsdPhysics


def test_wrist_is_driven_prismatic_chain_and_original_base_is_moving_link():
    from go2_pvcnn.control.dh116.tabletop_palm_mount import author_mount,JOINT_NAMES
    s=Usd.Stage.CreateInMemory();UsdGeom.Xform.Define(s,'/World/Hand')
    base=UsdGeom.Xform.Define(s,'/World/Hand/base_link').GetPrim();UsdPhysics.RigidBodyAPI.Apply(base)
    root=UsdPhysics.FixedJoint.Define(s,'/World/Hand/root_joint')
    root.CreateBody1Rel().SetTargets([base.GetPath()]);UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    author_mount(s,'/World/Hand')
    assert not root.GetPrim().IsActive()
    assert not root.GetJointEnabledAttr().Get()
    assert not root.GetPrim().HasAPI(UsdPhysics.ArticulationRootAPI)
    assert s.GetPrimAtPath('/World/Hand/wrist_fixed').HasAPI(UsdPhysics.ArticulationRootAPI)
    for name,axis in zip(JOINT_NAMES,('X','Y','Z')):
        j=UsdPhysics.PrismaticJoint(s.GetPrimAtPath('/World/Hand/'+name))
        assert j.GetAxisAttr().Get()==axis
        drive=UsdPhysics.DriveAPI(j.GetPrim(),'linear')
        assert drive and drive.GetTypeAttr().Get()=='force'
        assert drive.GetStiffnessAttr().Get()==100000.
        assert j.GetLowerLimitAttr().Get()==-.1 or abs(j.GetLowerLimitAttr().Get()+.1)<1e-7
    j=UsdPhysics.PrismaticJoint(s.GetPrimAtPath('/World/Hand/'+JOINT_NAMES[-1]))
    assert str(j.GetBody1Rel().GetTargets()[0])=='/World/Hand/base_link'


def test_world_target_is_transformed_to_wrist_local_axes_without_teleport():
    from go2_pvcnn.control.dh116.tabletop_palm_mount import local_translation
    a=torch.tensor([[1.,2,3,1,0,0,0],[1.,2,3,2**-.5,0,0,2**-.5]])
    p=a.clone();p[:,0]+=.01
    out=local_translation(p,a)
    torch.testing.assert_close(out[0],torch.tensor([.01,0,0.]),atol=1e-7,rtol=1e-5)
    torch.testing.assert_close(out[1],torch.tensor([0,-.01,0.]),atol=1e-7,rtol=1e-5)


def test_drive_readback_checks_hand_and_wrist_limits_separately():
    import numpy as np
    import pytest
    from go2_pvcnn.control.dh116.tabletop_load_parameters import _verify_velocity_readback
    from go2_pvcnn.control.dh116.linear_coupling import JOINTS,ACTIVE
    from go2_pvcnn.control.dh116.tabletop_palm_mount import JOINT_NAMES
    names=list(JOINT_NAMES)+list(JOINTS)
    values=np.array([[.05 if n in JOINT_NAMES else (.3 if n in ACTIVE else 2.97) for n in names]])
    _verify_velocity_readback(values,names,1,2.97)
    values[0,0]=3.
    with pytest.raises(ValueError):_verify_velocity_readback(values,names,1,2.97)


def test_measured_palm_offset_keeps_tracking_error_out_of_finger_compensation():
    from go2_pvcnn.control.dh116.tabletop_palm_mount import measured_offset
    target=torch.zeros((2,2,7));actual=target.clone();micro=torch.zeros((2,2,3))
    micro[:,:,2]=.001;target[:,:,2]=.021;actual[:,:,2]=.0205
    torch.testing.assert_close(measured_offset(actual,target,micro)[:,:,2],torch.full((2,2),.0005),atol=1e-8,rtol=1e-5)


def test_carriage_guard_checks_actual_tracking_and_velocity():
    from go2_pvcnn.control.dh116.tabletop_palm_mount import transport_safe
    target=torch.zeros((3,2,7));actual=target.clone();velocity=torch.zeros((3,2,3))
    actual[1,0,0]=.004;velocity[2,1,2]=.07
    assert transport_safe(actual,target,velocity).tolist()==[True,False,False]


def test_selected_carriage_reset_does_not_change_other_environment_targets():
    from types import SimpleNamespace
    from go2_pvcnn.control.dh116.tabletop_palm_mount import MountView,JOINT_NAMES
    targets=torch.full((3,3),.02)
    def write(value,joint_ids,env_ids=None):
        targets[:] = value if env_ids is None else targets
        if env_ids is not None:targets[env_ids]=value
    hand=SimpleNamespace(joint_names=list(JOINT_NAMES),body_names=['base_link'],set_joint_position_target=write)
    mount=MountView(hand);mount.reset(torch.zeros((2,7)),torch.tensor([2,0]))
    assert targets[[2,0]].eq(0).all() and targets[1].eq(.02).all()


def test_host_preflight_drives_carriage_without_teleporting_anchor():
    import ast,json,os,time
    import numpy as np
    from pathlib import Path
    from types import SimpleNamespace
    path=Path(__file__).parents[1]/'src/go2_pvcnn/control/dh116/tabletop_gpu_isaac.py'
    cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='GpuWorld')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_step')
    namespace=dict(np=np,time=time,os=os,json=json,__package__='go2_pvcnn.control.dh116')
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),namespace)
    moves=[]
    def forbidden(*a,**kw):raise AssertionError('preflight teleported the carriage anchor')
    def noop(*a,**kw):pass
    hand=SimpleNamespace(write_root_pose_to_sim=forbidden,set_joint_position_target=noop,
        write_data_to_sim=noop,data=SimpleNamespace(body_link_pose_w=torch.zeros((1,1,7)),body_lin_vel_w=torch.zeros((1,1,3))))
    mount=SimpleNamespace(move=lambda p:moves.append(p.clone()),base_id=0)
    w=SimpleNamespace(hands=[hand],mounts=[mount],tensor=torch.as_tensor,root_targets=np.zeros((1,1,7)),
        commands=np.zeros((1,1,6)),ids=[[0]],gains=np.zeros((1,1)),previous_gains=np.zeros((1,1)),
        flex=None,sim=SimpleNamespace(step=noop),args=SimpleNamespace(headless=True),refresh=noop)
    namespace['_step'](w)
    assert len(moves)==1


def test_transport_checkpoint_migration_requires_exact_controller_and_transport():
    import pytest
    from go2_pvcnn.control.dh116.tabletop_tensor_training import checkpoint_contract_changes
    from go2_pvcnn.control.dh116.tabletop_coordinated_grasp import CONTRACT as CONTROL
    from go2_pvcnn.control.dh116.tabletop_palm_mount import CONTRACT as WRIST
    old=dict(num_envs=4096,physics_hz=480)
    new=dict(old,grasp_controller=CONTROL,palm_transport=WRIST)
    with pytest.raises(ValueError):checkpoint_contract_changes(old,new)
    assert checkpoint_contract_changes(new,new)==[]
    for bad in [dict(old,palm_transport=WRIST),dict(new,palm_transport=dict(WRIST,effort_limit_n=200)),dict(new,physics_hz=240)]:
        with pytest.raises(ValueError):checkpoint_contract_changes(old,bad)
    with pytest.raises(ValueError):checkpoint_contract_changes(new,old)
