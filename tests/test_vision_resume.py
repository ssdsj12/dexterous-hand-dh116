import copy

import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_vision_resume import prepare_visual_resume


VERSION = "dh116-vision-student-ppo-v2-bounded"
LABELS = ("pinch_thumb_index", "tripod_thumb_index_middle")


def source_checkpoint():
    n = 4096
    contract = dict(version=VERSION, actor_observation_size=3844,
        critic_observation_size=371, actor_uses_privileged_state=False,
        point_cloud_shape=[2, 128, 3], history_frames=4,
        point_frame="actual_palm_body_link", position_unit="m",
        num_envs=n, groups=[dict(label=label, num_envs=n//2) for label in LABELS],
        camera_batch_size=128,
        camera_batching="same_frozen_physics_state_sequential_real_RGBD",
        visibility_mode="all", pinned_source_environment=None,
        camera_config_sha256="unchanged-extrinsics", camera_resolution_wh=[320, 240],
        action_contract={"size": 24, "version": "same-action"},
        rollout_steps=32,
        memory_contract={"storage": "cpu_float32", "microbatch_size": 512,
            "inference_batch_size": 256},
        primary_reset_sha256="pinch", secondary_reset_sha256="tripod",
        privileged_critic_contract=dict(num_envs=n, pose_sha256="same-pose",
            reward="support_deficit", success_bottom_m=[.39+i*1e-9 for i in range(n)],
            surface_force_sharing={"lift_reference_bottom_m": [.36+i*1e-9 for i in range(n)]}))
    return dict(version=VERSION, contract=contract, iteration=5, timesteps=655360,
        policy={"actor.weight": torch.arange(24.).reshape(4, 6)},
        optimizer={"state": {0: {"step": torch.tensor(200.),
            "exp_avg": torch.arange(24.).reshape(4, 6)*.01,
            "exp_avg_sq": torch.arange(24.).reshape(4, 6)*.001}},
            "param_groups": [{"lr": .00045, "params": [0]}]},
        learning_rate=.00045, episodes=torch.arange(n, dtype=torch.long)+10,
        successes=torch.arange(n, dtype=torch.long)//2,
        torch_rng=torch.get_rng_state().clone(),
        cuda_rng=[torch.arange(64, dtype=torch.uint8)])


def target_contract(saved, n=2048, resolution=(320, 240)):
    c=copy.deepcopy(saved["contract"])
    indices=list(range(n//2))+list(range(2048, 2048+n//2))
    c["num_envs"]=n
    for group in c["groups"]: group["num_envs"]=n//2
    critic=c["privileged_critic_contract"];critic["num_envs"]=n
    critic["success_bottom_m"]=[critic["success_bottom_m"][i] for i in indices]
    ref=critic["surface_force_sharing"]["lift_reference_bottom_m"]
    critic["surface_force_sharing"]["lift_reference_bottom_m"]=[ref[i] for i in indices]
    c.update(camera_mode="fixed_per_env", camera_batch_size=n,
        camera_batching="fixed_per_environment_real_RGBD",
        camera_resolution_wh=list(resolution))
    return c


def assert_same(a, b):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.dtype==b.dtype and a.device==b.device
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys()==b.keys()
        for k in a: assert_same(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert type(a)==type(b) and len(a)==len(b)
        for x,y in zip(a,b): assert_same(x,y)
    else: assert a==b


def test_default_is_strict_and_exact_resume_does_not_alias_source():
    saved=source_checkpoint(); before=copy.deepcopy(saved)
    exact=prepare_visual_resume(saved, copy.deepcopy(saved["contract"]))
    assert_same(exact, saved)
    assert exact is not saved
    assert exact["policy"]["actor.weight"].data_ptr()!=saved["policy"]["actor.weight"].data_ptr()
    exact["policy"]["actor.weight"][0,0]=-100
    assert_same(saved, before)
    with pytest.raises(ValueError, match="contract"):
        prepare_visual_resume(saved, target_contract(saved))


@pytest.mark.parametrize("n", [2048, 1024])
@pytest.mark.parametrize("resolution", [(320, 240), (160, 120), (224, 96)])
def test_explicit_downsize_keeps_training_state_and_selects_each_style(n, resolution):
    saved=source_checkpoint(); before=copy.deepcopy(saved)
    current=target_contract(saved,n,resolution); current_before=copy.deepcopy(current)
    actual=prepare_visual_resume(saved,current,allow_downsize=True)
    assert_same(saved,before);assert_same(current,current_before)
    for key in ("version", "policy", "optimizer", "learning_rate", "iteration",
            "timesteps", "torch_rng", "cuda_rng"):
        assert_same(actual[key],saved[key])
    assert actual["contract"]==current
    indices=torch.cat((torch.arange(n//2),torch.arange(2048,2048+n//2)))
    for name in ("episodes", "successes"):
        assert torch.equal(actual[name],saved[name][indices])
        assert actual[name].data_ptr()!=saved[name].data_ptr()
    meta=actual["resume_migration"]
    assert meta["source_contract"]==saved["contract"]
    assert meta["source_num_envs"]==4096 and meta["target_num_envs"]==n
    assert meta["retained_ranges"]==[[0,n//2],[2048,2048+n//2]]
    for i,label in enumerate(LABELS):
        start=i*2048; group=meta["groups"][label]
        assert group["retained_num_envs"]==n//2
        assert group["discarded_num_envs"]==2048-n//2
        for name in ("episodes", "successes"):
            assert group["retained"][name]==int(saved[name][start:start+n//2].sum())
            assert group["discarded"][name]==int(saved[name][start+n//2:start+2048].sum())
            assert meta["source_totals"][name]==int(saved[name].sum())
            assert meta["retained_totals"][name]==int(actual[name].sum())
            assert meta["retained_totals"][name]+meta["discarded_totals"][name]==meta["source_totals"][name]
    paths={change["path"] for change in meta["changes"]}
    assert {"num_envs", "camera_mode", "privileged_critic_contract.num_envs"}<=paths
    if resolution!=(320,240): assert "camera_resolution_wh" in paths
    # The new fixed checkpoint resumes strictly without migration authorization.
    assert_same(prepare_visual_resume(actual,current),actual)


@pytest.mark.parametrize("path,value", [
    (("privileged_critic_contract", "reward"), "changed-reward"),
    (("privileged_critic_contract", "pose_sha256"), "changed-pose"),
    (("camera_config_sha256",), "changed-extrinsics"),
    (("memory_contract", "microbatch_size"), 128),
    (("action_contract", "size"), 25),
    (("rollout_steps",), 24),
    (("camera_batching",), "unknown-batching"),
    (("unknown_new_field",), True),
])
def test_explicit_permission_does_not_allow_other_contract_changes(path,value):
    saved=source_checkpoint(); current=target_contract(saved)
    at=current
    for key in path[:-1]: at=at[key]
    at[path[-1]]=value
    with pytest.raises(ValueError):
        prepare_visual_resume(saved,current,allow_downsize=True)


@pytest.mark.parametrize("n", [8192, 4096, 3072, 2047, 512])
def test_expansion_odd_or_unapproved_target_count_is_rejected(n):
    saved=source_checkpoint(); current=target_contract(saved,2048);current["num_envs"]=n
    with pytest.raises(ValueError):
        prepare_visual_resume(saved,current,allow_downsize=True)


def test_wrong_group_layout_or_reference_selection_is_rejected():
    saved=source_checkpoint(); current=target_contract(saved)
    current["groups"].reverse()
    with pytest.raises(ValueError): prepare_visual_resume(saved,current,allow_downsize=True)
    current=target_contract(saved)
    current["privileged_critic_contract"]["surface_force_sharing"]["lift_reference_bottom_m"][0]+=.0000001
    with pytest.raises(ValueError): prepare_visual_resume(saved,current,allow_downsize=True)


def test_invalid_payload_legacy_or_unknown_resolution_is_rejected():
    saved=source_checkpoint(); current=target_contract(saved)
    for name in ("episodes", "successes"):
        broken=copy.deepcopy(saved);broken[name]=broken[name][:-1]
        with pytest.raises(ValueError): prepare_visual_resume(broken,current,allow_downsize=True)
    broken=copy.deepcopy(saved);broken["version"]="legacy_371"
    with pytest.raises(ValueError): prepare_visual_resume(broken,current,allow_downsize=True)
    current=target_contract(saved,resolution=(80,60))
    with pytest.raises(ValueError): prepare_visual_resume(saved,current,allow_downsize=True)


@pytest.mark.parametrize("allow_downsize", [False, True])
def test_unknown_extra_integer_to_boolean_is_rejected(allow_downsize):
    saved=source_checkpoint()
    saved["contract"]["unknown_extra"]={"value": 0}
    current=target_contract(saved) if allow_downsize else copy.deepcopy(saved["contract"])
    current["unknown_extra"]["value"]=False
    with pytest.raises(ValueError, match="contract"):
        prepare_visual_resume(saved,current,allow_downsize=allow_downsize)


@pytest.mark.parametrize("path,value", [
    (("camera_batch_size",), 2048.0),
    (("camera_resolution_wh",), [160.0,120.0]),
    (("groups",0,"num_envs"), 1024.0),
    (("privileged_critic_contract","num_envs"), 2048.0),
])
def test_allowed_migration_fields_still_require_canonical_types(path,value):
    saved=source_checkpoint();current=target_contract(saved)
    at=current
    for key in path[:-1]: at=at[key]
    at[path[-1]]=value
    with pytest.raises(ValueError):
        prepare_visual_resume(saved,current,allow_downsize=True)


@pytest.mark.parametrize("resolution", [(224,95), (224.0,96), (224,96.0)])
def test_nearby_or_noncanonical_wide_resolution_is_rejected(resolution):
    saved=source_checkpoint();current=target_contract(saved,1024,resolution)
    with pytest.raises(ValueError):
        prepare_visual_resume(saved,current,allow_downsize=True)
