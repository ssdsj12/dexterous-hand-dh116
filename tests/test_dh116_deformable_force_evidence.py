import copy

import pytest

from go2_pvcnn.control.dh116.tabletop_deformable_force_evidence import (
    audit_solver_force_row,
)


def ledger():
    return {
        "env_id": 0, "episode_uid": "fixture", "reset_id": 1, "rod_uid": "rod0",
        "physics_step": 3, "sdk_step": 3, "timestamp_s": 3 / 480,
        "sdk_time_s": 3 / 480, "dt_s": 1 / 480,
        "rod_tick": 12, "rod_substep": 12,
        "rod_nodes_w": [[0, 0, 0], [0, 0, 1]],
        "rod_node_contact_forces_w": [[1, 0, 0], [-1, 0, 0]],
        "contact_node_ids": [0, 1],
        "actual_pad_bindings": {
            "finger14_link": {"mesh_index": 0, "body_index": 1, "body_path": "/World/Hand/finger14_link"},
            "finger23_link": {"mesh_index": 1, "body_index": 2, "body_path": "/World/Hand/finger23_link"},
        },
        "rod_mesh_wrenches_w": {
            "finger14_link": [-1, 0, 0, 0, 0, 0],
            "finger23_link": [1, 0, 0, 0, 0, 0],
            "table": [0, 0, 0, 0, 0, 0],
        },
        "pair_force_residual_n": 0.,
    }


def test_aggregate_opposing_wrenches_cannot_identify_contact_pairs():
    report = audit_solver_force_row(ledger(), physics_evidence=False)
    assert report["active_rod_node_ids"] == [0, 1]
    assert report["mesh_wrenches_on_mesh_world"]["finger14_link"]["force_n"] == [-1., 0., 0.]
    assert report["mesh_wrenches_on_mesh_world"]["finger14_link"]["body_path"] == "/World/Hand/finger14_link"
    assert report["rod_mesh_contact_pairs"] is None
    assert report["table_contact"] is None  # net-zero wrench does not prove absence
    assert report["valid_for_r0_contact_or_slip"] is False
    assert report["valid_for_r2"] is False
    assert report["native_readback"] is False


@pytest.mark.parametrize("edit", [
    lambda r: r.update(sdk_step=4),
    lambda r: r.update(timestamp_s=0.0),
    lambda r: r.update(contact_node_ids=[1]),
    lambda r: r["actual_pad_bindings"]["finger23_link"].update(mesh_index=0),
    lambda r: r["rod_mesh_wrenches_w"].pop("table"),
    lambda r: r["rod_mesh_wrenches_w"]["table"].__setitem__(0, 1.0),
    lambda r: r["rod_mesh_wrenches_w"]["table"].__setitem__(3, float("nan")),
    lambda r: r.update(rod_mesh_contact_pairs=[]),
])
def test_rejects_misaligned_corrupt_or_invented_contact_evidence(edit):
    row = copy.deepcopy(ledger())
    edit(row)
    with pytest.raises(ValueError):
        audit_solver_force_row(row, physics_evidence=False)
