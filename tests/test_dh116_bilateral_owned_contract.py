from copy import deepcopy
import importlib
from pathlib import Path
import numpy as np
import pytest

def module():
    path=Path(__file__).parents[1]/"scripts/prepare_dh116_bilateral_owned_pinch.py"
    assert path.exists(),"independent bilateral plan is not implemented"
    return importlib.import_module("scripts.prepare_dh116_bilateral_owned_pinch")

def test_exact_independent_24d_composition_and_original_reset():
    p=module()
    from scripts import prepare_dh116_right_pinch_owned_contact as right
    from scripts import prepare_dh116_pinch_margin_contact as left
    pair=p.derive_pair();a=p.action_schedule(pair)
    assert a.equal(right.action_schedule()+left.action_schedule())
    assert pair["reset_seed"]==right.load_pair()["reset_seed"]
    assert pair["left"]==left.load_pair()["left"]
    assert pair["right"]==right.load_pair()["right"]
    assert pair["rod_seed"]==right.load_pair()["rod_seed"]
    assert not pair["native_qualified"] and not pair["teacher_admission"]
    assert np.any(a.numpy()[:,:6]) and np.any(a.numpy()[:,6:12])
    assert not np.any(a.numpy()[:,18:]) and not np.any(a.numpy()[:,12:17])
    assert pair["schedule"]["total_steps"]==234
    assert pair["schedule"]["ppo_updates"]==0

@pytest.mark.parametrize("ordinal",[0,48,49,208,232])
def test_both_hand_targets_at_each_physical_boundary(ordinal):
    p=module();pair=p.derive_pair();r=p.expected_boundary(pair,ordinal)
    f=float(np.clip((ordinal-48)/160,0,1))
    for side in ("left","right"):
        c=pair[side]["candidate"]
        np.testing.assert_allclose(r[side+"_q_rad"],np.array(c["active_q_rad"])+f*np.array(c["active_delta_rad"]),atol=1e-15)
    assert r["root_poses_w"][1,2]==pytest.approx(pair["reset_seed"]["reset_root_pose_w"][1][2]-.0005*f)

@pytest.mark.parametrize("ordinal",[-1,233,True,1.5])
def test_no_budget_extension(ordinal):
    p=module()
    with pytest.raises(ValueError):p.expected_boundary(p.derive_pair(),ordinal)

def test_original_one_hand_schedules_are_not_mutated():
    p=module()
    from scripts import prepare_dh116_right_pinch_owned_contact as right
    from scripts import prepare_dh116_pinch_margin_contact as left
    a,b=right.action_schedule().clone(),left.action_schedule().clone()
    p.action_schedule()
    assert a.equal(right.action_schedule()) and b.equal(left.action_schedule())

def test_frozen_config_rejects_other_hand_change(tmp_path):
    import json
    p=module();pair=p.derive_pair();pair["left"]["candidate"]["active_delta_rad"][0]+=.001
    path=tmp_path/p.CONFIG;path.parent.mkdir(parents=True);path.write_text(json.dumps(pair))
    with pytest.raises(ValueError,match="frozen"):p.load_pair(tmp_path)
