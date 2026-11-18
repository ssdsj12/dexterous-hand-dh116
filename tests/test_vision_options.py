from types import SimpleNamespace
import pytest
from go2_pvcnn.control.dh116.tabletop_surface_exploration import validate_options


def options(**changes):
    values=dict(vision_student=True,surface_force_sharing=True,surface_feedback=True,
        free_wrist_grasp=True,surface_imitation_mode='legacy',surface_reward_profile='support_deficit',
        vertical_force_reward=True,phase_force_reward=True)
    values.update(changes)
    return SimpleNamespace(**values)


def test_visual_asymmetric_ppo_uses_reward_without_privileged_bc():
    validate_options(options())


def test_legacy_reward_baseline_and_native_feedback_remain_required():
    with pytest.raises(ValueError,match='stable-hold'):
        validate_options(options(vision_student=False))
    with pytest.raises(ValueError):
        validate_options(options(surface_feedback=False))
