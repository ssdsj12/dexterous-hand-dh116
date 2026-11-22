"""Independent wrench-work checks; no Isaac or GPU required."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation


def test_translation_work_and_lag_difference_are_in_joules():
    from scripts.audit_dh116_grasp_contact_work import endpoint_work
    q = [0., 0., 0., 1.]
    result = endpoint_work([0, 0, 0], [0.002, 0, 0], q, q,
                           [3, 0, 0], [0, 0, 0], [-1, 0, 0], [0, 0, 0])
    assert result["issued_j"] == pytest.approx(0.006)
    assert result["current_j"] == pytest.approx(-0.002)
    assert result["difference_j"] == pytest.approx(0.008)
    assert result["energy_qualified"] is False


def test_rotational_work_uses_world_axis_and_ignores_quaternion_sign():
    from scripts.audit_dh116_grasp_contact_work import endpoint_work
    q0 = Rotation.from_euler("x", 90, degrees=True)
    q1 = Rotation.from_rotvec([0, 0.01, 0]) * q0
    args = ([1, 2, 3], [1, 2, 3], q0.as_quat(), -q1.as_quat(),
            [0, 0, 0], [0, 2, 0], [0, 0, 0], [0, -1, 0])
    result = endpoint_work(*args)
    assert result["issued_j"] == pytest.approx(0.02)
    assert result["current_j"] == pytest.approx(-0.01)
    assert result["difference_j"] == pytest.approx(0.03)


def test_virtual_work_is_invariant_under_world_rotation():
    from scripts.audit_dh116_grasp_contact_work import endpoint_work
    world = Rotation.from_euler("zyx", [30, 20, 10], degrees=True)
    f, displacement, tau, turn = map(np.array, ([2., 3., 4.], [.01, 0., .02],
                                                   [.2, .3, .4], [0., .01, 0.]))
    value = endpoint_work([0, 0, 0], world.apply(displacement), world.as_quat(),
                         (world * Rotation.from_rotvec(turn)).as_quat(),
                         world.apply(f), world.apply(tau), world.apply(f), world.apply(tau))
    assert value["issued_j"] == pytest.approx(0.103)
    assert value["difference_j"] == pytest.approx(0., abs=1.e-14)


@pytest.mark.parametrize("bad", ["nan", "nonunit"])
def test_invalid_measurement_is_not_converted_to_zero_work(bad):
    from scripts.audit_dh116_grasp_contact_work import endpoint_work
    q, p = [0., 0., 0., 1.], [0., 0., 0.]
    q1 = [0., 0., 0., 2.] if bad == "nonunit" else q
    p1 = [float("nan"), 0., 0.] if bad == "nan" else p
    with pytest.raises(ValueError):
        endpoint_work(p, p1, q, q1, p, p, p, p)


def packets():
    from copy import deepcopy
    routes = {side: dict(body_ids=[7], forces=np.array([[[2., 0, 0]]]),
                         torques=np.zeros((1, 1, 3)), is_global=True)
              for side in ("left", "right")}
    row = dict(physics_step=8, reaction_source_physics_step=7,
        issued_composer_wrench=deepcopy(routes),
        rod=dict(physics_step=8, rod_tick=72, run_uuid="r", episode_uid="e",
                 reset_id=1, env_id=0, rod_uid="rod", hand_wrenches_com_world=deepcopy(routes),
                 mesh_bindings={side+".pad": dict(side=side, body_index=7, prim_path=side+"/pad")
                                for side in routes}),
        sdk_hand_geometry=dict(physics_step=8,
            centers_of_mass={side+".pad": np.zeros(3) for side in routes},
            poses={side+".pad": [np.zeros(3), np.array([0., 0, 0, 1])] for side in routes}))
    after = deepcopy(row)
    after["physics_step"] = after["rod"]["physics_step"] = after["sdk_hand_geometry"]["physics_step"] = 9
    after["rod"]["rod_tick"] = 84
    after["reaction_source_physics_step"] = 8
    for side in routes:
        after["sdk_hand_geometry"]["centers_of_mass"][side+".pad"][0] = .001
        after["rod"]["hand_wrenches_com_world"][side]["forces"] *= -1
    return row, after


def test_audit_compares_actual_previous_reaction_and_keeps_both_sides():
    from scripts.audit_dh116_grasp_contact_work import audit_pair
    values = audit_pair(*packets())
    assert set(values) == {"left.pad", "right.pad"}
    assert values["left.pad"]["issued_j"] == pytest.approx(.002)
    assert values["left.pad"]["difference_j"] == pytest.approx(.004)


@pytest.mark.parametrize("fault", ["reset", "force", "body", "skip"])
def test_audit_does_not_match_across_reset_or_wrong_reaction(fault):
    from scripts.audit_dh116_grasp_contact_work import audit_pair
    first, after = packets()
    if fault == "reset":
        after["rod"]["reset_id"] = 2
    elif fault == "force":
        after["issued_composer_wrench"]["left"]["forces"] *= 2
    elif fault == "body":
        after["rod"]["mesh_bindings"]["left.pad"]["body_index"] = 8
    else:
        after["physics_step"] = after["rod"]["physics_step"] = after["sdk_hand_geometry"]["physics_step"] = 10
    with pytest.raises(ValueError):
        audit_pair(first, after)
