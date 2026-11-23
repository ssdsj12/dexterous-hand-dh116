import numpy as np
import pytest


def test_motion_decomposition_shares_coupling_instead_of_claiming_an_order():
    from scripts.audit_dh116_tripod_contact_readiness import factor_effects

    values = np.zeros((2, 2, 2))
    for w, j, r in np.ndindex(values.shape):
        values[w, j, r] = 2*w - 3*j + 4*r + 6*w*j - 2*j*r + 3*w*j*r
    np.testing.assert_allclose(factor_effects(values), [6., 0., 4.])
    assert sum(factor_effects(values)) == pytest.approx(values[1, 1, 1] - values[0, 0, 0])


def test_source_thumb_constraint_prevents_a_false_coordinated_closure_direction():
    from scripts.audit_dh116_tripod_contact_readiness import contact_direction

    channels = [6, 8]
    gap = np.tile([[-2e-5, -1e-5]], (6, 1))
    # Channel 6 has the best free SDF reduction but worsens source-hull clearance.
    guard = np.array([[-1.e-5, 0.], [-1.e-5, 0.]])
    free = contact_direction(gap, channels)
    guarded = contact_direction(gap, channels, hull_guard=guard)
    assert free[6] > .9
    assert guarded[6] <= 1e-12 and guarded[8] > .9
    assert np.all(guard @ guarded[channels] >= -1e-12)
    assert np.all(gap @ guarded[channels] < 0.)



def test_coordinated_constraint_can_preserve_both_source_hulls():
    from scripts.audit_dh116_tripod_contact_readiness import contact_direction

    channels = [6, 8, 9]
    gap = np.tile([[-3e-5, -2e-5, -1e-5]], (6, 1))
    hulls = np.array([[-1e-5, 0., 0.], [-1e-5, 0., 0.],
                      [0., -1e-5, 0.], [0., -1e-5, 0.]])
    action = contact_direction(gap, channels, hull_guard=hulls)
    assert np.all(hulls @ action[channels] >= -1e-12)
    assert action[9] > .9
    assert np.all(gap @ action[channels] < 0.)


def test_reaction_summary_retains_transient_load_and_owning_side(monkeypatch):
    from scripts import audit_dh116_tripod_contact_readiness as audit

    def measurement(rod, *, grasp):
        assert grasp == "tripod"
        return {
            "wrong_side_contacts": [],
            "unexpected_hand_contacts": [],
            "external_support_force_norm_n": {"support_0": [0.12, 0.11], "support_1": [0.21, 0.2]},
            "per_hand": {
                side: {"required_pads": ["index"], "per_substep": [
                    {"per_pad": {"index": {"normal_load_n": 0.0}}},
                    {"per_pad": {"index": {"normal_load_n": rod.get(side, 0.0)}}},
                ]}
                for side in ("left", "right")
            },
        }

    monkeypatch.setattr(audit, "measure_supported_distal_contact", measurement)
    summary = audit._reaction_summary([{"rod": {}}, {"rod": {"right": .075}}])
    assert summary["maximum_pad_load_n"] == {"left": {"index": 0.0}, "right": {"index": .075}}
    assert summary["positive_pad_substeps"] == {"left": {"index": 0}, "right": {"index": 1}}
    assert summary["minimum_support_force_n"] == {"support_0": .11, "support_1": .2}
    assert summary["solver_substeps"] == 4
