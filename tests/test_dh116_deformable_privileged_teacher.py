import copy
import json
from pathlib import Path

import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_deformable_privileged_teacher import (
    DeformablePrivilegedTeacher, load_deformable_teacher_config, verify_deformable_teacher_identity,
)
from go2_pvcnn.control.dh116.tabletop_privileged_teacher import PrivilegedTeacher
from go2_pvcnn.control.dh116.tabletop_stereo_rod_state import VERSION as ROD_VERSION
from go2_pvcnn.control.dh116.tabletop_squashed_gaussian import squashed_log_prob


ROOT = Path(__file__).resolve().parents[1]
GRASPS = ("pinch", "tripod", "power")


def config_path(grasp):
    return ROOT / "configs/dh116_training" / f"privileged_deformable_teacher_{grasp}_v1.json"


@pytest.fixture
def teacher():
    config, identity = load_deformable_teacher_config(config_path("pinch"), ROOT)
    return DeformablePrivilegedTeacher(config, identity), identity


def inputs(identity, *, batch=2, nodes=5):
    env = list(range(batch))
    clock = dict(env_ids=env, episode_uids=[f"ep-{i}" for i in env], reset_ids=[0] * batch,
                 rod_uids=["rod0"] * batch, physics_step=24, rod_tick=288, timestamp_s=24 / 480)
    positions = torch.zeros(batch, nodes, 3)
    positions[:, :, 1] = torch.linspace(0, 1, nodes)
    positions[:, :, 2] = .31
    tangent = torch.zeros_like(positions)
    tangent[..., 1] = 1
    rod = dict(version=ROD_VERSION, task_kind="deformable_rod", coordinate_frame="world",
               source_scope="cpu_fixture", backend_sha256=identity["backend_sha256"], **clock,
               node_positions=positions, velocities=torch.randn_like(positions) * .01,
               tangent=tangent, curvature=torch.zeros(batch, nodes),
               rest_lengths_m=torch.full((batch, nodes - 1), 1 / (nodes - 1)),
               deformation=torch.zeros(batch, nodes - 1))
    return torch.zeros(batch, 371), rod, clock, torch.ones(batch, nodes, dtype=torch.bool)


def test_independent_configs_preserve_371_24_and_checkpoint_identity():
    found = {}
    for grasp in GRASPS:
        config, identity = load_deformable_teacher_config(config_path(grasp), ROOT)
        assert identity["grasp"] == grasp
        assert identity["common_size"] == 371 and identity["action_size"] == 24
        assert identity["native_ready"] is False
        assert config["use_cameras"] is False and config["expert_initialization"] == "optional"
        assert "qualified_reset_and_371d_common" in config["native_prerequisites"]
        assert "qualified_flexible_r0" in config["native_prerequisites"]
        verify_deformable_teacher_identity(identity, identity)
        found[grasp] = identity
    assert len({identity["config_sha256"] for identity in found.values()}) == 3
    assert len({identity["checkpoint_namespace"] for identity in found.values()}) == 3
    with pytest.raises(ValueError, match="identity"):
        verify_deformable_teacher_identity(found["tripod"], found["pinch"])
    with pytest.raises(ValueError, match="identity"):
        verify_deformable_teacher_identity({"policy_version": "dh116-privileged-teacher-tanh-v1"}, found["pinch"])
    assert PrivilegedTeacher().deterministic(torch.zeros(1, 371)).shape == (1, 24)


def test_config_rejects_unqualified_native_or_visual_dependency(tmp_path):
    config = json.loads(config_path("pinch").read_text())
    changed = tmp_path / "teacher.json"
    for key, value in (("native_ready", True), ("use_cameras", True), ("expert_initialization", "required")):
        changed.write_text(json.dumps({**config, key: value}))
        with pytest.raises(ValueError, match="camera-free"):
            load_deformable_teacher_config(changed, ROOT)


def test_actor_critic_rod_branches_match_existing_tanh_logprob(teacher):
    model, identity = teacher
    common, rod, clock, mask = inputs(identity)
    obs = model.pack(common, rod, clock, mask)
    assert obs["rod_nodes"].shape == (2, 5, 13)
    assert obs["common"].shape == (2, 371)
    sampled = model.sample(obs)
    evaluated = model.evaluate_latents(obs, obs, sampled["latent_actions"])
    assert sampled["actions"].shape == (2, 24) and evaluated["value"].shape == (2,)
    assert torch.equal(sampled["actions"], sampled["latent_actions"].tanh())
    assert torch.equal(model.deterministic(obs), model.latent_mean(obs).tanh())
    expected = squashed_log_prob(sampled["latent_actions"], sampled["latent_mean"], sampled["latent_std"])
    torch.testing.assert_close(sampled["log_prob"], expected)
    torch.testing.assert_close(evaluated["log_prob"], expected)


def test_rod_readback_identity_clock_and_source_cannot_be_substituted(teacher):
    model, identity = teacher
    common, rod, clock, mask = inputs(identity)
    for change in ({"rod_uids": ["rod0", "wrong"]}, {"physics_step": 25}, {"rod_tick": 287}, {"timestamp_s": .055},
                   {"episode_uids": ["ep-0", "stale"]}, {"reset_ids": [0, 1]}, {"env_ids": [1, 0]}):
        with pytest.raises(ValueError, match="identity"):
            model.pack(common, {**rod, **change}, clock, mask)
    with pytest.raises(ValueError, match="schema"):
        model.pack(common, {**rod, "version": "rigid_pose_v1"}, clock, mask)
    with pytest.raises(ValueError, match="371"):
        model.pack(torch.full_like(common, float("nan")), rod, clock, mask)
    with pytest.raises(ValueError, match="shape"):
        model.pack(common, rod, clock, mask[:, :-1])


def test_invalid_geometry_and_adjacent_strain_do_not_leak_and_storage_order_is_invariant(teacher):
    model, identity = teacher
    common, rod, clock, mask = inputs(identity)
    mask[:, 1] = False
    reference = model.pack(common, rod, clock, mask)
    changed = copy.deepcopy(rod)
    for key in ("node_positions", "velocities", "tangent", "curvature"):
        changed[key][:, 1] = float("nan")
    changed["deformation"][:, :2] = float("nan")
    other = model.pack(common, changed, clock, mask)
    torch.testing.assert_close(other["rod_nodes"], reference["rod_nodes"])
    noisy = other["rod_nodes"].clone()
    noisy[~mask] = float("nan")
    permutation = torch.tensor([4, 2, 0, 3, 1])
    permuted = dict(common=common, rod_nodes=noisy[:, permutation], rod_mask=mask[:, permutation])
    torch.testing.assert_close(model.latent_mean(reference), model.latent_mean(permuted))
    torch.testing.assert_close(model.value(reference), model.value(permuted))
    too_few = mask.clone()
    too_few[:, 2:4] = False
    with pytest.raises(ValueError, match="three"):
        model.pack(common, rod, clock, too_few)


def test_both_rod_branches_have_gradients_and_mask_blocks_invalid_gradient(teacher):
    model, identity = teacher
    common, rod, clock, mask = inputs(identity, nodes=7)
    mask[:, -1] = False
    obs = model.pack(common, rod, clock, mask)
    obs["rod_nodes"].requires_grad_()
    loss = model.latent_mean(obs).square().sum() + model.value(obs).square().sum()
    loss.backward()
    for encoder in (model.actor_rod, model.critic_rod):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.parameters())
    assert torch.count_nonzero(obs["rod_nodes"].grad[~mask]) == 0
    assert obs["rod_nodes"].grad[mask].abs().sum() > 0


def test_valid_tangent_and_legacy_weights_are_rejected_when_incompatible(teacher):
    model, identity = teacher
    common, rod, clock, mask = inputs(identity)
    rod["tangent"][:, 0] *= 2.
    with pytest.raises(ValueError, match="unit tangent"):
        model.pack(common, rod, clock, mask)
    with pytest.raises(RuntimeError, match="state_dict"):
        model.load_state_dict(PrivilegedTeacher().state_dict())
