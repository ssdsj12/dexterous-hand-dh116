"""CPU guards for the independent raised-rod Power native scene."""
import json
from pathlib import Path

import numpy as np
import pytest

from go2_pvcnn.control.dh116 import tabletop_power_shared_rod as power
from test_dual_hand_rod_runtime import Grid, runtime


def test_attach_both_palms_and_supports_after_original_reset_once(monkeypatch):
    original, base, bindings, poses, _ = runtime(monkeypatch, pairs=True)
    original.require_pairs = True
    original.reset(physics_step=2, episode_uid="reset-cpu:0", reset_id=1, pad_poses=poses)
    clock = original.clock()
    palms = {}
    for side in power.SIDES:
        key = f"{side}.base_link"
        palms[key] = dict(bindings[f"{side}.finger14_link"], body_name="base_link", body_index=6,
                          prim_path=f"/World/{side}Hand/base_link")
    pose = (np.zeros(3), np.array([0., 0., 0., 1.]))
    caps = []
    base.rope.enable_contact_pair_capture = lambda capacity: caps.append(capacity)
    values = dict(palm_bindings=palms, palm_grids={key:Grid() for key in palms},
                  palm_poses={key:pose for key in palms}, expected_clock=clock,
                  support_grids={key:Grid() for key in ("support_0", "support_1")},
                  support_poses={key:pose for key in ("support_0", "support_1")})
    attached = power.PowerSharedRodRuntime.from_original_reset(original, **values)
    assert attached.clock() == clock and original.clock()["clock_valid"] is False
    assert attached.rope is base.rope and base.rope.resets == 1 and base.rope.steps == 0
    assert caps == [len(base.nodes)*15*12]
    assert attached.mesh_names[11:] == ("left.base_link", "right.base_link", "support_0", "support_1")
    with pytest.raises(ValueError):
        power.PowerSharedRodRuntime.from_original_reset(original, **values)
    assert len(caps) == 1 and len(base.rope.slots) == 15


def test_native_support_guard_rejects_one_micrometre_intrusion():
    from go2_pvcnn.control.dh116.tabletop_power_supported_task import guard_hand_supports

    xyz = np.array([[x,y,z] for x in (0.01,0.02) for y in (-.002,.002) for z in (-.002,.002)])
    support = {"supports":[{"center_m":[0.,0.,0.], "size_m":[.02,.02,.02]},
                           {"center_m":[0.,.4,0.], "size_m":[.02,.02,.02]}]}
    clean = xyz + [1.e-6,0,0]
    bad = xyz - [1.e-6,0,0]
    assert guard_hand_supports({"left.source.test":clean}, support)["minimum_gap_m"] == pytest.approx(1.e-6)
    with pytest.raises(ValueError, match="left.source.test"):
        guard_hand_supports({"left.source.test":bad}, support)


def test_frozen_power_seed_keeps_free_rod_on_two_supports():
    from go2_pvcnn.control.dh116.tabletop_power_supported_task import prepare_initial_state

    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / "configs/dh116_tasks/flexible_power_supported_contact_v1.json").read_text())
    table = {"center_m":[.65,0.,.275], "size_m":[.8,1.,.05]}
    seed, active, nodes, poses = prepare_initial_state(cfg, table)
    assert seed["joint_velocities"] == [[0.]*11]*2
    assert seed["reset_root_pose_w"] == [cfg["hands"][side]["pose_wxyz"] for side in power.SIDES]
    assert active.shape == (2,6) and nodes.shape == (41,3)
    assert all(pose[1].tolist() == [0.,0.,0.,1.] for pose in poses.values())
    cfg["fixed_rod_nodes"] = [0]
    with pytest.raises(ValueError, match="free Power rod"):
        prepare_initial_state(cfg, table)
