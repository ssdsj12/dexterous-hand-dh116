"""CPU numerical oracle; CUDA only when explicitly enabled by the operator."""
from copy import deepcopy
from dataclasses import replace
import importlib.util
import os

import numpy as np
import pytest
import torch

from test_native_teacher_context import backend_profile
from test_dh116_shared_teacher_source import fixture


MODULE = "go2_pvcnn.control.dh116.tabletop_tensor_native_inputs"


def native_fixture(old_profile, n):
    from go2_pvcnn.control.dh116.tabletop_shared_teacher_source import SharedBackendProfile, SharedTeacherSource
    from go2_pvcnn.control.dh116.tabletop_teacher_mounted_wrist_source import adapt_mounted_hands
    profile = SharedBackendProfile.load()
    origins = np.array([(3.*(i % 16), -6.*(i // 16), 0.) for i in range(n)])
    runtime, sim, common, contexts = fixture(old_profile, profile, origins)
    # Reconstruct the original adapter from the exact float32 native samples.
    # Authored reset roots retain their original precision and source checks.
    for i, c in enumerate(common):
        packets = deepcopy(c["source_packets"])
        for hi, p in enumerate(packets):
            for key in ("q", "qd", "body_link_pose_w", "body_lin_vel_w", "body_ang_vel_w"):
                a = np.asarray(p[key], dtype=np.float32)
                if key == "q": a += np.float32(i*.0001 + hi*.003)
                p[key] = a.tolist()
        common[i] = adapt_mounted_hands(packets, expected_identity=c["identity"])
    source = SharedTeacherSource(runtime, sim, backend_profile=profile, env_origins=origins,
        scope="cpu_fixture", device="cpu")
    oracle = source.capture(common, contexts)
    packets = [c["source_packets"] for c in common]
    kwargs = dict(
        joint_names=tuple(tuple(p["joint_names"]) for p in packets[0]),
        body_link_pose_w=torch.tensor([[p["body_link_pose_w"] for p in row] for row in packets]),
        body_com_velocity_w=torch.tensor([[p["body_lin_vel_w"]+p["body_ang_vel_w"] for p in row] for row in packets]),
        root_seed_w=torch.tensor([row[0]["reset_seed"]["root_seed_w"] for row in packets], dtype=torch.float64),
        env_origins=torch.tensor(origins, dtype=torch.float64),
        rod_positions_w=runtime.solver.positions(), rod_velocities_w=runtime.solver.velocities(),
        rest_lengths_m=torch.tensor(oracle.parent_parameters["rest_lengths_m"], dtype=torch.float32),
        task_context=torch.cat([c.values for c in contexts]),
        task_context_valid=torch.cat([c.valid for c in contexts]),
        source_identity=oracle.source,
        context_identities=tuple(c.metadata.identities[0] for c in contexts),
        backend_profile_sha256=profile.digest, grasp="pinch", scope="cpu_fixture")
    q = torch.tensor([[p["q"] for p in row] for row in packets])
    qd = torch.tensor([[p["qd"] for p in row] for row in packets])
    return q, qd, kwargs, oracle, source


def test_tensor_pack_api_is_available():
    assert importlib.util.find_spec(MODULE) is not None, "device-native shared observation pack is missing"


@pytest.mark.parametrize("n", [2, 128])
def test_entire_batch_exactly_matches_original_shared_oracle(backend_profile, n):
    from go2_pvcnn.control.dh116.tabletop_tensor_native_inputs import pack_shared_tensor_inputs
    q, qd, kwargs, oracle, source = native_fixture(backend_profile, n)
    packet = pack_shared_tensor_inputs(q, qd, **kwargs)
    assert packet.source_family == "shared_batched_solver_v1"
    assert packet.source == oracle.source
    assert packet.native_qualified is False
    assert packet.grasp == "pinch"
    packet.validate_fresh(source._boundary())
    for key, expected in oracle.inputs.items():
        torch.testing.assert_close(packet.inputs[key], expected, atol=0, rtol=0, equal_nan=True)
    obs = packet.inputs
    assert obs["common_valid"].sum().item() == n*66
    assert torch.isnan(obs["common371"][~obs["common_valid"]]).sum().item() == n*305
    assert obs["common371"][0,156].item() == -.5  # raw WXYZ sign is preserved
    saved = {k: v.clone() for k, v in obs.items()}
    q.fill_(8); qd.fill_(9)
    for key, value in kwargs.items():
        if isinstance(value, torch.Tensor): value.fill_(0)
    for key in saved:
        torch.testing.assert_close(obs[key], saved[key], atol=0, rtol=0, equal_nan=True)
    with pytest.raises(ValueError, match="epoch|clock|source"):
        packet.validate_fresh(replace(oracle.source, sdk_step=oracle.source.sdk_step+1))


@pytest.mark.parametrize("damage", ["quaternion", "names", "origin_z", "rod_nan", "degenerate", "rest",
    "context", "context_epoch", "source_scope", "profile", "parent_rows", "q_shape"])
def test_wrong_native_source_or_nonfinite_geometry_is_rejected(backend_profile, damage):
    from go2_pvcnn.control.dh116.tabletop_tensor_native_inputs import pack_shared_tensor_inputs
    q, qd, kwargs, _, _ = native_fixture(backend_profile, 2)
    if damage == "quaternion": kwargs["body_link_pose_w"][0,0,3:] *= 2
    elif damage == "names": kwargs["joint_names"] = (("fake",)*17, kwargs["joint_names"][1])
    elif damage == "origin_z": kwargs["env_origins"][0,2] = .1
    elif damage == "rod_nan": kwargs["rod_positions_w"][0,0,0] = torch.nan
    elif damage == "degenerate": kwargs["rod_positions_w"][0,1] = kwargs["rod_positions_w"][0,0]
    elif damage == "rest": kwargs["rest_lengths_m"][0,0] = 0
    elif damage == "context": kwargs["task_context_valid"][0,4] = True
    elif damage == "context_epoch":
        kwargs["context_identities"] = (replace(kwargs["context_identities"][0], reset_id=99), kwargs["context_identities"][1])
    elif damage == "source_scope": kwargs["scope"] = "owner_source_unqualified"
    elif damage == "profile": kwargs["backend_profile_sha256"] = "0"*64
    elif damage == "parent_rows": kwargs["source_identity"] = replace(kwargs["source_identity"], env_rows=(1,0))
    else: q = q[:1]
    with pytest.raises(ValueError): pack_shared_tensor_inputs(q, qd, **kwargs)


def test_new_packet_hooks_original_model_without_changing_legacy_pack_gate(backend_profile):
    from go2_pvcnn.control.dh116.tabletop_tensor_native_inputs import pack_shared_tensor_inputs
    from go2_pvcnn.control.dh116.tabletop_shared_native_teacher import SharedNativeTeacher
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import ROOT
    from go2_pvcnn.control.dh116.tabletop_training_reward import load_config
    from test_native_teacher_model import cpu_model
    q, qd, kwargs, oracle, source = native_fixture(backend_profile, 2)
    packet = pack_shared_tensor_inputs(q, qd, **kwargs)
    _, *_, config, _ = cpu_model(backend_profile)
    reward = load_config(ROOT/"configs/dh116_training/pinch_finger_goal_reward_v1.json",ppo_gamma=.995)
    model = SharedNativeTeacher(config,backend_profile=source.profile,num_envs=2,reward_config=reward)
    inputs = packet.for_teacher(model,current_identity=source._boundary())
    original = model.pack_shared(oracle,source)
    torch.testing.assert_close(model.latent_mean(inputs),model.latent_mean(original),atol=0,rtol=0)
    torch.testing.assert_close(model.value(inputs),model.value(original),atol=0,rtol=0)
    inputs["rod_nodes"].fill_(77)
    torch.testing.assert_close(packet.inputs["rod_nodes"],oracle.inputs["rod_nodes"],atol=0,rtol=0)
    with pytest.raises(ValueError): model.pack_shared(packet,source)
    with pytest.raises(ValueError): packet.for_teacher(object(),current_identity=source._boundary())
    with pytest.raises(ValueError): replace(packet,grasp="power").for_teacher(model,current_identity=source._boundary())
    with pytest.raises(ValueError): packet.for_teacher(model,current_identity=replace(oracle.source,sdk_step=9))


def test_pack_has_no_tensor_cpu_numpy_or_python_lane_materialization(backend_profile, monkeypatch):
    from go2_pvcnn.control.dh116.tabletop_tensor_native_inputs import pack_shared_tensor_inputs
    q, qd, kwargs, oracle, _ = native_fixture(backend_profile, 2)
    def forbidden(*args,**kw):
        pytest.fail("native batch must not be materialized on CPU or as Python lanes")
    with monkeypatch.context() as patch:
        for name in ("cpu","numpy","tolist"):
            patch.setattr(torch.Tensor,name,forbidden)
        packet = pack_shared_tensor_inputs(q,qd,**kwargs)
    torch.testing.assert_close(packet.inputs["rod_nodes"],oracle.inputs["rod_nodes"],atol=0,rtol=0)


@pytest.mark.skipif(os.environ.get("DH116_RUN_DEVICE_TESTS") != "1", reason="operator must allocate a CUDA device")
def test_cuda_pack_keeps_all_tensors_and_matches_cpu_oracle(backend_profile):
    from go2_pvcnn.control.dh116.tabletop_tensor_native_inputs import pack_shared_tensor_inputs
    q, qd, kwargs, oracle, _ = native_fixture(backend_profile, 128)
    device = os.environ.get("DH116_TEST_DEVICE", "cuda:0")
    kwargs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
    packet = pack_shared_tensor_inputs(q.to(device), qd.to(device), **kwargs)
    for key, expected in oracle.inputs.items():
        assert packet.inputs[key].device == torch.device(device)
        torch.testing.assert_close(packet.inputs[key].cpu(), expected, atol=2e-6, rtol=2e-6, equal_nan=True)
