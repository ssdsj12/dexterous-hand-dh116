import numpy as np
import pytest

from scripts.certify_dh116_tripod_distal_route import prove_segment, support_clearance_lower


def test_rejects_an_unsafe_midpoint_between_clear_endpoints():
    def margins(t):
        return {"rod": abs(t - .5) - .05, "support": .1}
    proof = prove_segment(margins, {"rod": 1., "support": 0.}, max_samples=64)
    assert proof["proved"] is False and proof["violations"][0]["margins"]["rod"] < 0.


def test_covers_whole_interval_with_positive_displacement_reserve():
    def margins(t):
        return {"rod": .025 + .1 * t, "table": .020}
    proof = prove_segment(margins, {"rod": .10, "table": .02}, max_samples=64)
    assert proof["proved"] is True
    assert min(box["minimum_margin_m"] for box in proof["certified_boxes"]) > 0.
    assert sum(np.asarray(box["upper"])[0] - np.asarray(box["lower"])[0]
               for box in proof["certified_boxes"]) == pytest.approx(1.)


def test_uses_convex_separation_when_world_aabbs_overlap():
    from scipy.spatial.transform import Rotation

    local = np.array([[x,y,z] for x in (-1.,1.) for y in (-.1,.1) for z in (-.1,.1)])
    vertices=Rotation.from_euler("z",45,degrees=True).apply(local)
    low=np.array([.5,-.6,-.05])
    high=np.array([.6,-.5,.05])
    assert np.all(vertices.max(0)>low) and np.all(vertices.min(0)<high)
    assert support_clearance_lower(vertices,low,high) > .5
    assert support_clearance_lower(vertices,-.02*np.ones(3),.02*np.ones(3)) < 0


def test_left_terminal_reference_cannot_select_the_right_native_link_transforms(tmp_path):
    import hashlib
    import json

    from go2_pvcnn.control.dh116.linear_coupling import ACTIVE
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import encode_owned
    from scripts.certify_dh116_tripod_distal_route import read_terminal

    sides = ("left", "right")
    joints = {side: {"joint_names": ACTIVE, "q": np.arange(6.)[None, :] + i}
              for i, side in enumerate(sides)}
    poses = {side: {"body_names": ["base_link"],
                    "link_poses_xyzw": np.array([[[float(i), 0., 0., 0., 0., 0., 1.]]])}
             for i, side in enumerate(sides)}
    record = dict(physics_step=522, tripod_near_geometry=dict(passed=True, hand_pairs=[], hands=poses),
                  action_readback=dict(post=dict(hands=joints)),
                  rod=dict(node_positions=np.zeros((1, 41, 3))))
    ledger = tmp_path / "physical.jsonl"
    ledger.write_text(json.dumps({"operation": "physical", "payload": encode_owned({"exchange": record})}) + "\n")
    contract = dict(side="left", native_sdk_step=522,
                    physical_ledger_sha256=hashlib.sha256(ledger.read_bytes()).hexdigest())
    q, body, _ = read_terminal(ledger, contract)
    assert q.tolist() == joints["left"]["q"][0].tolist()
    assert body["base_link"][0] == 0.


def test_archived_envelope_uses_all_material_nodes_and_contiguous_steps(monkeypatch):
    from types import SimpleNamespace
    from scripts import certify_dh116_tripod_distal_route as route

    current = SimpleNamespace(physics_step=240, node_positions=np.zeros((41, 3)),
        rest_lengths_m=np.ones(40), support_boxes=(), receipt_sha256="p", sample_sha256="now")
    requested = []

    def load(path, digest, sdk):
        assert path == "receipt" and digest == "p"
        requested.append(sdk)
        nodes = current.node_positions.copy()
        nodes[-1, 2] = .0001 * (240 - sdk)
        return SimpleNamespace(physics_step=sdk, node_positions=nodes, rest_lengths_m=np.ones(40),
                               support_boxes=(), receipt_sha256=digest, sample_sha256=str(sdk))

    monkeypatch.setattr(route, "load_parked_reference", load)
    envelope = route.archived_rod_envelope("receipt", current, 2)
    assert requested == [238, 239]
    assert envelope["max_point_displacement_m"] == pytest.approx(.0002)
    assert envelope["sample_sha256"] == ["238", "239", "now"]
    assert envelope["online_motion_bound"] is False
    with pytest.raises(ValueError):
        route.archived_rod_envelope("receipt", current, 0)


def test_envelope_debits_rod_only_and_preserves_entire_interval_requirement():
    def margins(_):
        return {"rod:source:finger52_link": .00012, "support:source:finger52_link": .0001}

    motion = {key: 0. for key in margins(0.)}
    proof = prove_segment(margins, motion, max_samples=4, rod_displacement_reserve_m=.0002)
    assert proof["proved"] is False
    assert set(proof["violations"][0]["margins"]) == {"rod:source:finger52_link"}
    good = prove_segment(margins, motion, max_samples=4, rod_displacement_reserve_m=.00005)
    assert good["proved"]
    assert good["certified_boxes"][0]["minimum_margin_m"] == pytest.approx(.00007)
    with pytest.raises(ValueError):
        prove_segment(margins, motion, max_samples=4, rod_displacement_reserve_m=-.00001)
