"""Pinned CPU geometry proposals, explicitly not additional native evidence."""
import hashlib,json
from pathlib import Path
import pytest
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
BASE=Path("/mnt/mydisk/xk/projects/dh116-autonomous-3c80166")
def read(name,sha):
    data=(BASE/name/"result.json").read_bytes()
    assert hashlib.sha256(data).hexdigest()==sha
    return decode_owned(json.loads(data))
@pytest.fixture(scope="module")
def target():
    return read("pinch-effective-target-20261114-run03","cc6fcfcbf001d597fcc46d0a27768974f8e0f4b1714a872074ac8a6a65d8ddbe")
@pytest.fixture(scope="module")
def path():
    return read("pinch-effective-path-20261114-run03","03b70ba97423665eac710ddd0fe3e868282885428efe1e0775bf304ccf9aa280")
def test_archived_commanded_geometry_already_misses_original_effective_region(target):
    for side in ("left","right"):
        assert all(r["ellipse_value"]>1 for r in target["sides"][side]["baseline"]["pads"].values())
def test_independent_right_endpoint_pass_does_not_promote_left_or_native(target):
    assert not target["sides"]["left"]["search"]["best"]["admission"]["static_constraints_passed"]
    right=target["sides"]["right"]["search"]["best"]
    assert right["admission"]["static_constraints_passed"]
    assert all(r["ellipse_value"]<=.95 and r["original_region_passed"] for r in right["pads"].values())
    assert not target["all_static_candidates_passed"]
    assert not target["native_ready"] and not target["teacher_admission"]
def test_path_includes1025_bounds_and_maximum_segment_lipschitz(path):
    r=path["sides"]["right"]
    assert r["path_samples"]==1025 and r["static_path_passed"]
    assert r["worst"]["table_m"]>=.0021 and r["worst"]["other_m"]>=0
    assert r["worst"]["depth_m"]>=-.0003
    assert r["bow_world_x_m"]==.0001
    assert r["root_lipschitz_per_fraction_m"]> .0006
    assert r["same_original_finger_action_amplitude"] and r["original_wrist_limits_preserved"]
    assert not r["dynamic_tracking_qualified"] and not r["rod_response_qualified"]
    assert not path["native_ready"] and not path["native_recipe_created"]
    assert path["new_physics_steps"]==path["ppo_updates"]==0
def test_failed_straight_path_is_retained_not_relabeled_passed():
    r=read("pinch-effective-path-20261114-run01","81340b52b641abdb23dbb46238f9543329fcf3bad1f1e4b7c22c8c1a7048e25c")["sides"]["right"]
    assert not r["static_path_passed"]
    assert r["worst_identity"]["other_m"]["index"]==703
    assert r["worst_identity"]["other_m"]["shape"]=="source:finger13_link"
    assert r["worst"]["other_m"]<0
