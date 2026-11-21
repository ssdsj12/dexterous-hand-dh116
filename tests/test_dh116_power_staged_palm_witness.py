
import hashlib

import numpy as np

from scripts import verify_dh116_power_staged_palm_witness as palm


def test_power_palm_permission_uses_only_the_exact_body_and_active_step(tmp_path, monkeypatch):
    shapes = [(0, 0, np.zeros((4,3))), (1, 0, np.zeros((4,3))), (0, 1, np.zeros((4,3)))]
    bindings = {}
    for side in ("left", "right"):
        p = tmp_path / f"{side}.base_link.npz"
        np.savez(p, dummy=np.zeros(1))
        bindings[side+".base_link"] = dict(side=side, body_name="base_link",
            prim_path=f"/World/{side}/base_link", body_index=0, sdf_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    probe = palm.SupportedPalmProbe(bindings, shapes,
       bake_dirs={side:tmp_path for side in ("left","right")},boundary_sdk=100,
       stations={"left.base_link":[31],"right.base_link":[9]})
    assert not probe.contact_shape(100,(0,0))
    assert probe.contact_shape(101,(0,0))
    assert probe.contact_shape(101,(1,0))
    assert not probe.contact_shape(101,(0,1))

    transfers = []
    def transfer(previous, current):
        transfers.append((previous["physics_step"],current["physics_step"]))
        return previous["physics_step"]>100

    monkeypatch.setattr(palm,"check_reaction_delivery",transfer)
    monkeypatch.setattr(palm,"measure_power_contact",lambda _: {"measured":True})
    assert probe.observe({"physics_step":100}) == (None,None)
    first = {"physics_step":101, "rod":{"rod_mesh_contact_pairs":{"pairs":[{"mesh_name":"left.base_link","node_index":31}]}}}
    evidence, stop = probe.observe(first)
    assert stop is None and evidence["actual_allowed_palm_pairs"]==1
    assert probe.summary()["first_contact_sdk_by_palm"]=={"left.base_link":101}
    second = {"physics_step":102, "rod":{"rod_mesh_contact_pairs":{"pairs":[{"mesh_name":"right.base_link","node_index":10}]}}}
    evidence, stop = probe.observe(second)
    assert stop == "unexpected_contact_outside_power_palm_station"
    assert transfers==[(100,101),(101,102)]
    assert evidence["violations"]==["right.base_link:10"]
    assert probe.summary()["first_reaction_sdk"]==102
