import numpy as np
import pytest

from scripts import verify_dh116_supported_force_seating as seating


def test_coordinated_tripod_dose_keeps_the_screened_joint_ratios():
    from scripts.verify_dh116_tripod_paired_pad_balance import select_coordinated_command
    planned = np.zeros(24)
    planned[:4] = [.03, .02, .015, .011]
    # A high thumb alone may not cancel half of the screened coordinated command.
    row = dict(total_normal_load_n=.35,
               per_pad={pad:{"normal_load_n": force}
                        for pad, force in (("thumb",.19),("index",.04),("middle",.12))})
    evidence = {"wrong_side_contacts":[],"unexpected_hand_contacts":[],
                "per_hand":{"left":{"loaded":False,"per_substep":[row]*12}}}
    issued, minimum = select_coordinated_command(planned, evidence)
    np.testing.assert_array_equal(issued, planned)
    assert minimum == {"thumb":.19,"index":.04,"middle":.12}
    evidence["per_hand"]["left"]["loaded"] = True
    row["per_pad"]["index"]["normal_load_n"] = .06
    row["total_normal_load_n"] = .37
    row["effective"] = True
    row["opposition_cos"] = -.5
    issued, _ = select_coordinated_command(planned, evidence)
    assert not np.any(issued)
    with pytest.raises(ValueError, match="left Tripod"):
        select_coordinated_command(planned, evidence, side="right")
    planned[8] = .03
    with pytest.raises(ValueError, match="left Tripod"):
        select_coordinated_command(planned, evidence)
    assert seating.LEFT_PADS == {2:"index"} and seating.RIGHT_PADS == {8:"index",9:"middle"}


def test_coordinated_tripod_dose_requires_all_measured_solver_substeps():
    from scripts.verify_dh116_tripod_paired_pad_balance import select_coordinated_command
    evidence = {"wrong_side_contacts":[],"unexpected_hand_contacts":[],
                "per_hand":{"left":{"loaded":False,"per_substep":[{}]*11}}}
    with pytest.raises(ValueError, match="12-substep"):
        select_coordinated_command(np.zeros(24), evidence)


def test_coordinated_ray_never_continues_wrong_side_or_unknown_force_evidence():
    from scripts.verify_dh116_tripod_paired_pad_balance import select_coordinated_command
    planned=np.zeros(24);planned[:4]=[.03,.02,.015,.011]
    row=dict(total_normal_load_n=.35,per_pad={pad:{"normal_load_n":force}
        for pad,force in (("thumb",.19),("index",.04),("middle",.12))})
    evidence={"wrong_side_contacts":[{"link":"left.finger23_link"}],"unexpected_hand_contacts":[],
              "per_hand":{"left":{"loaded":False,"per_substep":[row]*12}}}
    with pytest.raises(ValueError,match="side"):
        select_coordinated_command(planned,evidence)
    evidence["wrong_side_contacts"]=[]
    row["per_pad"]["index"]["normal_load_n"]=float("nan")
    with pytest.raises(ValueError,match="force"):
        select_coordinated_command(planned,evidence)
