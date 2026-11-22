"""Exercise the held-out diameter, length and stiffness combinations."""
import json
from itertools import product
from pathlib import Path

import numpy as np
import pytest

from go2_pvcnn.control.dh116.tabletop_supported_wire_specs import build_variant, catalog_wire_specs
from scripts.verify_dh116_wire_variant_isaac import _choice

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "configs/dh116_training"
EVAL_CATALOG = TRAINING / "supported_wire_generalization_factorial_holdout_v1.json"


def test_factorial_holdouts_are_jointly_unseen_and_stay_out_of_training():
    catalog = json.loads(EVAL_CATALOG.read_bytes())
    rod = json.loads((ROOT / catalog["reference_rod"]["path"]).read_bytes())["rod"]
    reference = json.loads((TRAINING / "supported_wire_generalization_v2.json").read_bytes())
    states = catalog_wire_specs(catalog, rod)
    original = catalog_wire_specs(reference, rod)
    assert states["train"] == original["train"]
    assert states["validation"] == original["validation"]

    def key(case):
        return (case["rod"]["radius_m"], case["length_m"], case["elastic_modulus_scale"])

    expected = set(product((.0046, .00525), (.475, .525), (.7, 1.3)))
    assert {key(case) for case in states["test"]} == expected
    for axis, held_out in enumerate(zip(*expected)):
        assert set(held_out).isdisjoint(key(case)[axis] for case in states["train"])

    scene_path = ROOT / "configs/dh116_tasks/flexible_power_supported_contact_v1.json"
    scene = json.loads(scene_path.read_bytes())
    for case in states["test"]:
        with pytest.raises(ValueError, match="training-only"):
            _choice(catalog, case["id"])
        radius, length, stiffness = key(case)
        candidate = build_variant(scene, rod, radius_m=radius, length_m=length,
                                  elastic_modulus_scale=stiffness,
                                  grasp_material_ids={"left": 31, "right": 9})
        assert candidate["spec_sha256"] == case["spec_sha256"]
        nodes = np.asarray(candidate["scene"]["rod_nodes_w_m"])
        assert len(nodes) == 41
        assert np.linalg.norm(np.diff(nodes, axis=0), axis=1).sum() == pytest.approx(length)
        assert candidate["scene"]["support_scene"]["wire_radius_m"] == pytest.approx(radius)
