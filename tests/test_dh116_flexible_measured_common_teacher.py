"""Original common371 ledger and independent masked-model semantics on CPU."""
from pathlib import Path

import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_flexible_measured_common_teacher import (
    MeasuredCommonTeacher, common_selection, model_contract,
)
from test_dh116_deformable_privileged_teacher import config_path, inputs
from go2_pvcnn.control.dh116.tabletop_deformable_privileged_teacher import (
    load_deformable_teacher_config, DeformablePrivilegedTeacher,
)


ROOT = Path(__file__).resolve().parents[1]


def setup(grasp="pinch"):
    config, base = load_deformable_teacher_config(config_path(grasp), ROOT)
    common, rod, clock, mask = inputs(base)
    fields = common_selection()
    selected = torch.zeros_like(common, dtype=torch.bool)
    selected[:, fields] = True
    common[~selected] = torch.nan
    return config, base, common, selected, rod, clock, mask


def test_measured_model_only_consumes_fields_with_unchanged_original_indices():
    config, base, common, valid, rod, clock, nodes = setup()
    model = MeasuredCommonTeacher(config, base)
    packed = model.pack_measured(common, valid, rod, clock, nodes)
    assert packed["common371"].shape == (2,371)
    assert torch.isnan(packed["common371"][~valid]).all()
    assert model.deterministic(packed).shape == (2,24)
    assert torch.isfinite(model.value(packed)).all()
    assert len(model.model_identity["common_indices"]) == 66
    assert model.model_identity["base_identity_sha256"] == base["model_identity_sha256"]
    assert model.model_identity["native_ready"] is False


@pytest.mark.parametrize("change", ["fabricated_na", "missing_required", "rigid_field", "foreign_mask"])
def test_model_rejects_filled_missing_wrong_or_mismatched_sources(change):
    config, base, common, valid, rod, clock, nodes = setup()
    model = MeasuredCommonTeacher(config, base)
    if change == "fabricated_na":
        common[:,64] = 0
    elif change == "missing_required":
        common[:,0] = torch.nan
        valid[:,0] = False
    elif change == "rigid_field":
        common[:,64] = 1
        valid[:,64] = True
    else:
        valid = valid[:,1:]
    with pytest.raises(ValueError, match="common"):
        model.pack_measured(common, valid, rod, clock, nodes)


def test_legacy_state_dict_and_independent_grasps_keep_separate_identity():
    configs = []
    for style in ("pinch","tripod","power"):
        config, base, *_ = setup(style)
        configs.append(model_contract(base))
    assert len({r["model_identity_sha256"] for r in configs}) == 3
    assert len({r["checkpoint_namespace"] for r in configs}) == 3
    config, base, *_ = setup()
    with pytest.raises(RuntimeError, match="state_dict"):
        MeasuredCommonTeacher(config, base).load_state_dict(
            DeformablePrivilegedTeacher(config, base).state_dict())


def test_original_ppo_update_uses_owned_measured_observation_tree(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_deformable_privileged_ppo import DeformablePPOTrainer
    from go2_pvcnn.control.dh116.tabletop_deformable_privileged_checkpoint import save_checkpoint
    from go2_pvcnn.control.dh116.tabletop_privileged_ppo import PPOConfig
    config, base, common, valid, rod, clock, nodes = setup()
    model = MeasuredCommonTeacher(config, base)
    with pytest.raises(ValueError, match="pack_measured"):
        model.pack(torch.zeros_like(common), rod, clock, nodes)
    obs = model.pack_measured(common, valid, rod, clock, nodes)
    trainer = DeformablePPOTrainer(model, PPOConfig(epochs=1,minibatches=1), rollout_steps=2,num_envs=2)
    for _ in range(2):
        sample = trainer.collect_step(obs,obs)
        trainer.record_step(torch.tensor([1.,2.]),torch.tensor([True,True]),torch.zeros(2),
            executed_actions=sample["actions"],collection_token=sample["collection_token"])
    stored = trainer.rollout.batch()
    assert torch.isnan(stored["actor_obs"]["common371"][:,64]).all()
    assert trainer.update()["minibatch_updates"] == 1
    assert trainer.counters["updates"] == 1
    with pytest.raises(ValueError,match="identity"):
        save_checkpoint(tmp_path/"old-format.pt", trainer)
    assert not (tmp_path/"old-format.pt").exists()
