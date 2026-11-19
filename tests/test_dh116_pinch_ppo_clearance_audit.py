from copy import deepcopy

import pytest


def row(step, clearances, *, failure=None, episode="episode0"):
    identity = dict(run_uuid="run0", env_id=0, episode_uid=episode, reset_id=1, rod_uid="rod0",
                    physics_step=step, clock_valid=True)
    left, right = clearances
    return dict(operation="physical", payload=dict(exchange=dict(physics_step=step, policy_step=1,
        substep=step-11, original_guard_failure=failure,
        post_step_clearance=dict(identity=identity, floor_m=.0021, clearance_m=[[left, right]],
                                 hand_table_contacts=None),
        closure_geometry=dict(source_complete=True, failure_reasons=(
            ["left:table:source:finger22_link"] if failure else []), sides={
            "left": dict(table_clearance_m={"source:finger22_link": left}),
            "right": dict(table_clearance_m={"source:base_link": right})}),
        wrist_projection=dict(safe_z_clipped={"__dh116_array__": [[False, False]]}))))


def test_first_clearance_crossing_keeps_physical_identity_and_contact_unknown():
    from scripts.audit_dh116_pinch_ppo_clearance import audit_clearance

    rows = [row(11, (.0021002334621673113, .00211648824)),
            row(12, (.0020976735901807597, .00211765340), failure="post_step_source_clearance:left")]
    result = audit_clearance(rows)
    found = result["first_clearance_failure"]
    assert found["identity"]["physics_step"] == 12
    assert found["link"] == "finger22_link"
    assert found["minimum_clearance_m"] == rows[1]["payload"]["exchange"]["post_step_clearance"]["clearance_m"][0][0]
    assert found["observed_floor_deficit_m"] == pytest.approx(2.32640981924e-6)
    assert found["previous_clearance_m"] == .0021002334621673113
    assert found["wrist_projection_clipped"] is False
    assert found["hand_table_contacts"] is None
    assert not result["teacher_admission"]
    changed = deepcopy(rows)
    changed[1]["payload"]["exchange"]["post_step_clearance"]["identity"]["episode_uid"] = "other"
    with pytest.raises(ValueError, match="same-episode"):
        audit_clearance(changed)
