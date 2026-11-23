"""Independent source-bound initial-height candidates never alter old physics."""
from copy import deepcopy
import numpy as np
import pytest

from go2_pvcnn.control.dh116.tabletop_dual_hand_reset import prepare_reset_state,_seed
from go2_pvcnn.control.dh116.linear_coupling import expand


def test_height_derivation_preserves_all_joint_data_and_original_source():
    from go2_pvcnn.control.dh116.tabletop_shared_training_height import derive_height_seed
    source=prepare_reset_state(hand_roots={s:[.5,y,.6,1.,0.,0.,0.]
        for s,y in (("left",.1),("right",-.1))},origin=[0.,0.,0.],
        root_xy=[[0.,0.],[0.,0.]],initial_offset_m=.04,
        joint_positions=[expand(np.full(6,.4)).tolist()]*2,
        joint_velocities=[np.zeros(11).tolist()]*2,source_description="explicit CPU fixture")
    saved=deepcopy(source)
    candidate,proof=derive_height_seed(source,grasp="pinch",delta_z_m=(.005,.005),
                                      evidence_sha256="a"*64)
    assert source==saved and _seed(candidate)==candidate
    assert candidate["sha256"]!=source["sha256"]
    assert candidate["joint_positions"]==source["joint_positions"]
    assert candidate["joint_velocities"]==source["joint_velocities"]
    assert candidate["initial_offset_m"]==source["initial_offset_m"]
    for key in ("hand_roots","authored_root_seed_w","reset_root_pose_w"):
        a,b=np.array(candidate[key]),np.array(source[key])
        np.testing.assert_array_equal(a[:,:2],b[:,:2])
        np.testing.assert_array_equal(a[:,3:],b[:,3:])
        np.testing.assert_allclose(a[:,2]-b[:,2],[.005,.005],atol=1e-15,rtol=0)
    assert proof["parent_seed_sha256"]==source["sha256"]
    assert proof["native_reset_qualified"] is False
    assert proof["rod_translation_m"]==[0.,0.,0.]


def test_initial_mesh_world_mapping_preserves_static_supports_and_lane_offsets():
    from go2_pvcnn.control.dh116.tabletop_shared_training_height import initial_mesh_world_poses
    plan={"entries":[{"name":"left.finger14_link"},{"name":"table"}]}
    measured=[({"left.finger14_link":([.1,.2,.405],[0.,0.,0.,1.])},{}),
              ({"left.finger14_link":([3.1,.2,.405],[0.,0.,0.,1.])},{})]
    class Solver:
        def mesh_state_snapshot(self):
            return {"current_positions_world_m":np.array([[[.1,.2,.4],[0.,0.,.2]],
                                                         [[3.1,.2,.4],[3.,0.,.2]]],np.float32),
                    "current_quaternions_xyzw":np.tile([0.,0.,0.,1.],(2,2,1)).astype(np.float32)}
    snapshot=Solver().mesh_state_snapshot()
    xyz,quat=initial_mesh_world_poses(Solver(),plan,measured)
    np.testing.assert_array_equal(xyz[:,1],snapshot["current_positions_world_m"][:,1])
    np.testing.assert_allclose(xyz[:,0,2],[.405,.405],rtol=0,atol=2e-8)
    np.testing.assert_array_equal(xyz[1,0,:2],[np.float32(3.1),np.float32(.2)])
    np.testing.assert_array_equal(quat,snapshot["current_quaternions_xyzw"])
    measured[1][0]["left.finger14_link"]=([3.1,.2,.405],[0.,0.,0.,2.])
    with pytest.raises(ValueError,match="quaternion"):
        initial_mesh_world_poses(Solver(),plan,measured)


def test_training_cli_forwards_verified_height_to_actual_coupling(monkeypatch,tmp_path):
    import contextlib,hashlib,json,sys,types
    from scripts import train_dh116_shared_native_teacher as train
    from scripts import audit_dh116_isaac_batch_scene as scene
    from scripts import audit_dh116_deformx_shared_pairs as pairs
    from scripts import run_dh116_work_packages as scheduler
    binary=tmp_path/"python/ropesim/_ropesim.so"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"CPU entrypoint routing fixture, not a native solver")
    backend=types.ModuleType("ropesim._ropesim")
    backend.__file__=str(binary)
    package=types.ModuleType("ropesim")
    package._ropesim=backend
    monkeypatch.setitem(sys.modules,"ropesim",package)
    failure=tmp_path/"failure.json";failure.write_text("{}")
    height={"schema":"dh116-shared-training-height-candidate-v1","grasp":"pinch",
            "failure_evidence":str(failure),
            "evidence_sha256":hashlib.sha256(failure.read_bytes()).hexdigest(),
            "delta_z_m":[.005,.005]}
    candidate=tmp_path/"candidate.json";candidate.write_text(json.dumps(height))
    calls=[]
    monkeypatch.setattr(scene,"load_grasp_archive",lambda *a:"CPU archive")
    monkeypatch.setattr(pairs,"load_inputs",lambda *a:(None,"CPU reset"))
    monkeypatch.setattr(scene,"_gpu_memory",lambda *a:0)
    monkeypatch.setattr(scene,"_run",lambda *a,**kw:calls.append(kw))
    monkeypatch.setattr(scheduler,"native_file_lock",lambda *a:contextlib.nullcontext())
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","3")
    monkeypatch.setattr(sys,"argv",["train","--grasp","pinch","--receipt",str(failure),
        "--receipt-sha256",height["evidence_sha256"],"--backend",str(binary),
        "--backend-sha256",hashlib.sha256(binary.read_bytes()).hexdigest(),
        "--gpu","3","--num-envs","2","--updates","2","--output",str(tmp_path/"output"),
        "--height-candidate",str(candidate),"--height-candidate-sha256",
        hashlib.sha256(candidate.read_bytes()).hexdigest()])
    train.main()
    assert len(calls)==1 and calls[0]["max_sdk_step"]==522
    assert calls[0]["extra_exercise"].keywords["training_height"]=={
        key:height[key] for key in ("delta_z_m","evidence_sha256")}
