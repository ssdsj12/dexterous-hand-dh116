import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scripts.audit_dh116_tripod_index_contact_loss import _tracking, gap_motion, index_loads, split_link_motion


def plane():
    return {"values": np.broadcast_to(np.arange(5.)[:, None, None], (5, 5, 5)) - 2.,
            "origin": np.full(3, -2.), "spacing": 1.}


def pose(x, angle=0.):
    return np.array([x, 0., 0.]), Rotation.from_euler("z", angle).as_quat()


@pytest.mark.parametrize(("link_shift", "node_shift", "expected"), [
    (0., .5, (.5, 0.)),
    (-.25, 0., (0., .25)),
    (.25, .5, (.5, -.25)),
])
def test_gap_decomposition_uses_same_material_node(link_shift, node_shift, expected):
    result = gap_motion(plane(), pose(0.), pose(link_shift),
                        np.array([[.1, 0., 0.]]), np.array([[.1 + node_shift, 0., 0.]]),
                        radius=.2)
    g00, g01, g10, g11 = result["gaps_m"][0]
    link = result["link_effect_m"][0]
    rod = result["rod_effect_m"][0]
    assert (rod, link) == pytest.approx(expected)
    assert result["valid"].tolist() == [True]
    assert g00 == pytest.approx(-.1)
    assert g11 - g00 == pytest.approx(rod + link)


def test_rotation_interaction_and_missing_lattice_point():
    result = gap_motion(plane(), pose(0.), pose(0., np.pi / 2),
                        np.array([[.9, .2, 0.], [3., 0., 0.]]),
                        np.array([[1.2, .7, 0.], [.4, 0., 0.]]), radius=.1)
    assert result["valid"].tolist() == [True, False]
    assert result["interaction_m"][0] == pytest.approx(.2)
    assert np.isnan(result["rod_effect_m"][1])
    with pytest.raises(ValueError, match="unit quaternion"):
        gap_motion(plane(), pose(0.), (np.zeros(3), np.zeros(4)),
                   np.zeros((1, 3)), np.zeros((1, 3)), radius=.1)


def test_index_loads_keep_material_identity_and_missing_substeps():
    def sample(substep, node_index, normal, mesh_name="left.finger23_link"):
        return dict(substep=substep, node_index=node_index, normal_force_on_rod_n=normal,
                    mesh_name=mesh_name)

    loads = index_loads([sample(0, 1, .2), sample(0, 2, .1), sample(5, 2, .4),
                         sample(1, 2, 1., "right.finger23_link")], node_count=3)
    assert loads.shape == (12, 3)
    assert loads[0].tolist() == pytest.approx([0., .2, .1])
    assert loads[5].tolist() == pytest.approx([0., 0., .4])
    assert loads[1].sum() == 0.
    with pytest.raises(ValueError, match="index contact"):
        index_loads([sample(12, 1, .1)], node_count=3)

def test_tracking_indexes_full_backend_targets_by_physx_dof():
    hand = dict(joint_names=["wrist_x", "finger21", "finger22"], active_indices=[1],
                q=np.array([[.003, .712, .66]]), qd=np.zeros((1, 3)),
                mounted=True, units_by_name=dict(wrist_x="m", finger21="rad", finger22="rad"),
                backend_targets=np.array([[.004, .715, 0.]]))
    row = dict(action_readback=dict(
        pre=dict(hands=dict(left=hand)), post=dict(hands=dict(left=hand)),
        request=dict(action24=np.zeros((1, 24))),
    ))
    result = _tracking(row)["joints"]
    assert result["finger21"]["backend_target"] == .715
    assert result["finger22"]["backend_target"] is None
    assert result["wrist_x"]["unit"] == "m"
    assert result["wrist_x"]["backend_target_before"] == .004
    assert result["wrist_x"]["backend_target"] == .004


@pytest.mark.parametrize(('end_palm', 'end_link', 'expected'), [
    (.2, .2, (-.2, 0.)),
    (0., .2, (0., -.2)),
    (.3, .2, (-.3, .1)),
])
def test_splits_link_gap_into_palm_and_palm_relative_motion(end_palm, end_link, expected):
    nodes0 = np.array([[.1, 0., 0.]])
    nodes1 = np.array([[.5, 0., 0.]])
    result = split_link_motion(plane(), (pose(0.), pose(end_palm)),
                               (pose(0.), pose(end_link)), (nodes0, nodes1), radius=.2)
    assert (result['palm_effect_m'][0], result['finger_effect_m'][0]) == pytest.approx(expected)


def test_palm_rotation_carries_link_lever_arm_and_checks_palm_quaternion():
    palm0, palm1 = pose(0.), pose(0., np.pi / 2)
    link0 = pose(.5)
    link1 = np.array([0., .5, 0.]), palm1[1]
    nodes = np.array([[.8, .3, 0.]])
    result = split_link_motion(plane(), (palm0, palm1), (link0, link1), (nodes, nodes), radius=.2)
    assert result["palm_effect_m"][0] == pytest.approx(-.5)
    assert result["finger_effect_m"][0] == pytest.approx(0.)
    bad_palm = (palm1[0], 2 * palm1[1])
    with pytest.raises(ValueError, match="unit quaternion"):
        split_link_motion(plane(), (palm0, bad_palm), (link0, link1), (nodes, nodes), radius=.2)
