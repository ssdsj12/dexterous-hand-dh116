"""Immutable real CPU queries, distinct from native contact or Teacher evidence."""
import hashlib,json
from pathlib import Path
import pytest

BASE=Path("/mnt/mydisk/xk/projects/dh116-autonomous-3c80166")
FILES={
 "near":("tripod-hard-constraints-20261114-run01/result.json","a484586dd4e08cbdfedd740fba3df53b4bcf9d636b16a7162e624422665fb1df"),
 "contact":("tripod-hard-constraints-20261114-run02/result.json","858c72d463cda94f3416087dc6c6e7892a2e6f3b6450f28c288ec93558b6670f"),
 "fixed":("tripod-fixed-root-path-20261114-run01/result.json","5a95240e0b75e5dc5edc651489ba05ef104767db52d4b0bd99b282bdfe3c5535"),
 "pose":("tripod-pose-path-20261114-run01/result.json","a0aa3d147095585c63c03922b1e94278452530b2cca5671b0b1b75b6527035a0")}
def read(name):
    relative,sha=FILES[name];p=BASE/relative
    if not p.exists():pytest.skip("private actual CPU evidence unavailable")
    assert hashlib.sha256(p.read_bytes()).hexdigest()==sha
    return json.loads(p.read_text())

def test_near_contact_final_is_safe_geometry_but_all_sdf_gaps_are_positive():
    r=read("near");p=r["final_packet"];m=p["measurement"]
    assert r["geometry_queries"]==452
    assert r["original_endpoint_pass_count"]==284
    assert m["endpoint_geometric_gate"] is True
    assert min(p["constraints"])>=0
    assert all(g>0 for g in m["node_sdf_gap_m"])
    assert m["minimum_table_gap_m"]==pytest.approx(.002198028920044004)
    assert m["minimum_nonrequired_gap_lower_m"]==pytest.approx(5.5506813818669935e-6)
    assert r["solver_success"] is False

def test_contact_request_failure_is_preserved_and_not_promoted():
    r=read("contact")
    assert r["require_node_contact"] is True
    assert r["geometry_queries"]==442
    assert r["hard_constraint_pass_count"]==0
    assert r["final_packet"]["measurement"]["endpoint_geometric_gate"] is False
    assert r["teacher_admitted"] is r["native_ready"] is False

def test_fixed_root_opening_fails_clearance_without_claiming_penetration():
    r=read("fixed");q=r["quiet_start"]
    assert min(q["table_m"].values())==pytest.approx(.0005700779909862952)
    assert r["fixed_root_path"]["status"]=="measured_violation"
    assert r["reference_initial_and_path_passed"] is False
    assert min(q["table_m"].values())>0

def test_root_and_joint_path_is_continuously_covered_and_uses_full_motion_bounds():
    r=read("pose");p=r["pose_path"]
    assert r["reference_initial_and_path_passed"] is True
    assert len(p["samples"])==119
    leaves=sorted(p["certified_intervals"],key=lambda row:row["start"])
    assert len(leaves)==59
    assert leaves[0]["start"]==0 and leaves[-1]["end"]==1
    assert all(a["end"]==b["start"] for a,b in zip(leaves,leaves[1:]))
    assert all(min(a["margin_lower_m"].values())>=0 for a in leaves)
    assert all(v>r["joint_only_point_bound_m"][k] for k,v in r["complete_point_motion_bound_m"].items())
    assert p["minimum_clearance_margin_m"]==pytest.approx(9.104268377338173e-7)
    assert r["sdk_action_path_validated"] is False
    assert r["native_ready"] is r["teacher_admitted"] is False
    assert r["new_physics_steps"]==r["ppo_updates"]==0
