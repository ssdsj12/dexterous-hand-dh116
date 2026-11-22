"""Keep material stiffness separable from wire diameter and length."""
import json
from pathlib import Path

import numpy as np
import pytest

from go2_pvcnn.control.dh116.tabletop_supported_wire_specs import catalog_wire_specs

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs/dh116_tasks/flexible_pinch_candidate_v1.json"
CATALOG = ROOT / "configs/dh116_training/supported_wire_generalization_v2.json"


def read_catalog():
    baseline = json.loads(BASE.read_bytes())["rod"]
    catalog = json.loads(CATALOG.read_bytes())
    return catalog, baseline


def test_geometry_matched_pairs_change_stiffness_without_changing_mass_or_geometry():
    catalog, baseline = read_catalog()
    for cases in catalog_wire_specs(catalog, baseline).values():
        pair = [row for row in cases if row["id"].endswith(("-soft", "-stiff"))]
        soft, stiff = sorted(pair, key=lambda row: row["elastic_modulus_scale"])
        assert soft["spec_sha256"] != stiff["spec_sha256"]
        assert soft["length_m"] == stiff["length_m"]
        assert soft["rod"]["radius_m"] == stiff["rod"]["radius_m"]
        assert soft["rod"]["linear_density_kg_m"] == stiff["rod"]["linear_density_kg_m"]
        ratio = stiff["elastic_modulus_scale"] / soft["elastic_modulus_scale"]
        for field in ("axial_stiffness_n", "bend_stiffness_n_m2"):
            assert stiff["rod"][field] / soft["rod"][field] == pytest.approx(ratio)


def test_material_catalog_rejects_leakage_across_splits():
    catalog, baseline = read_catalog()
    train = next(row for row in catalog["variants"] if row["split"] == "train")
    validation, test = [next(row for row in catalog["variants"] if row["split"] == split)
                        for split in ("validation", "test")]
    old = dict(test)
    test.update(radius_m=train["radius_m"], length_m=train["length_m"])
    with pytest.raises(ValueError, match="geometry assigned across splits"):
        catalog_wire_specs(catalog, baseline)
    test.update(old, elastic_modulus_scale=float(np.nextafter(validation["elastic_modulus_scale"], 1.)))
    with pytest.raises(ValueError, match="stiffness assigned across splits"):
        catalog_wire_specs(catalog, baseline)
    test.update(old, elastic_modulus_scale=float("nan"))
    with pytest.raises(ValueError, match="finite and positive"):
        catalog_wire_specs(catalog, baseline)


def test_held_out_stiffness_cases_cannot_enter_the_native_training_pilot():
    from scripts.verify_dh116_wire_variant_isaac import _choice

    catalog, baseline = read_catalog()
    source, case = _choice(catalog, "train-stiff")
    assert source == baseline
    assert case["elastic_modulus_scale"] == 1.2
    for name in ("validation-soft", "test-stiff"):
        with pytest.raises(ValueError, match="training-only"):
            _choice(catalog, name)
