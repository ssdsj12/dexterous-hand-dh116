"""Regression of actual bounded CPU records; not a new native qualification."""
import hashlib,json
from pathlib import Path
import numpy as np
import pytest

BASE=Path("/mnt/mydisk/xk/projects/dh116-autonomous-3c80166")
FILES={
 "trf":("tripod-node-closure-20261114-run01/result.json","c4ea21c8eb95900612afee1ddd08088b3f4f25fc761ac533797c70ef7e8a9783"),
 "dogbox":("tripod-node-closure-20261114-run02/result.json","f0e36fb782299ee48e41e9282f5c6943563c1aef51332e15201ef6981772e339"),
 "witness":("tripod-node-closure-20261114-run02/nonrequired-witness.json","0b156293f517c3a201d5f43c1c2d55ca327a0ad1056e0c29151b8d14a3f13f36")}

def read(name):
    path,sha=FILES[name];p=BASE/path
    if not p.exists():pytest.skip("private actual CPU evidence unavailable")
    assert hashlib.sha256(p.read_bytes()).hexdigest()==sha
    return json.loads(p.read_text())

def test_actual_budget_and_failure_evidence_remain_separate():
    for name,count in (("trf",484),("dogbox",312)):
        r=read(name)
        assert r["evaluations"]==count<=500
        assert len(r["completed_rows"])==count
        assert r["solver_success"] is False
        assert r["endpoint_pass_count"]==0
        assert r["physics_steps"]==r["ppo_updates"]==0
        assert r["native_ready"] is r["teacher_admitted"] is False
    assert read("dogbox")["solver_method"]=="dogbox"

def test_selected_probe_is_not_mislabeled_optimizer_final_or_valid_reset():
    r=read("dogbox");b=r["best"]
    assert b["evaluation"]==300
    assert not np.array_equal(b["x"],r["solver_x"])
    assert b["measurement"]["node_sdf_gap_m"]==pytest.approx(
        [4.705052112937303e-5,-1.0336641918374904e-5,3.8202191201474126e-5],abs=1e-12)
    assert b["measurement"]["minimum_table_gap_m"]<.0021
    assert b["measurement"]["endpoint_geometric_gate"] is False

def test_finger13_overlap_is_proven_not_only_loose_lower_bound():
    w=read("witness")
    assert w["selected_was_fd_probe"] is True
    assert w["witnesses"]["finger13_link"]["feasible_gap_upper_m"]==pytest.approx(-5.1705660961549626e-5)
    for n in ("finger22_link","finger32_link"):
        assert w["witnesses"][n]["full_3d_certificate"]["lower_m"]>0
    assert w["new_physics_steps"]==0
    assert w["native_ready"] is False
