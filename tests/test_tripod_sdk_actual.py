"""Frozen original-kernel CPU results are not actual SDK or native evidence."""
import hashlib,json
from pathlib import Path
import numpy as np
import pytest

BASE=Path("/mnt/mydisk/xk/projects/dh116-autonomous-3c80166")
CASES={"intrinsic_linear":"f0726bb65f7ed3d2e645fb9d93226dd0e83f2100ee1cc928c95fa60fc74156ae",
       "geodesic_knots":"13afddee4057d00b0c51881aa6b9a251cd49d7b52eb048517536f2c6f151a42a"}

@pytest.mark.parametrize("mode",CASES)
def test_real_cpu_target_reconstruction_is_qualified_only_for_reference_and_exceeds_old_budget(mode):
    p=BASE/("tripod-sdk-path-20261114-"+mode+"-run04/result.json")
    if not p.exists():pytest.skip("private CPU evidence unavailable")
    assert hashlib.sha256(p.read_bytes()).hexdigest()==CASES[mode]
    r=json.loads(p.read_text());plan=r["plan"];replay=r["target_replay"];proof=r["target_path"]
    assert r["status"]=="complete" and r["summary"]["motion_steps"]==200
    assert r["unchanged_stage_budget"]["total_sdk_steps"]==274
    assert r["unchanged_stage_budget"]["required_extra_steps"]==34
    assert not r["unchanged_stage_budget"]["within_budget"]
    assert len(replay["rows"])==201 and len(plan["actions"])==25
    for row in replay["rows"][1:]:
        assert row["policy_index"]==(row["physics_index"]-1)//8
        assert row["action24"]==plan["actions"][row["policy_index"]]
        assert np.linalg.norm(row["projection_delta_m"])<1e-12
    assert proof["proved"] is True
    assert proof["required_rod_minimum_threshold_m"]==-.00005
    assert all("rod:cooked:finger33_link" in s for s in proof["samples"].values())
    assert all("rod:source:finger33_link" not in s for s in proof["samples"].values())
    leaves=sorted(proof["certified_intervals"],key=lambda x:x["start"])
    assert leaves[0]["start"]==0 and leaves[-1]["end"]==1
    assert all(a["end"]==b["start"] for a,b in zip(leaves,leaves[1:]))
    assert r["new_physics_steps"]==r["ppo_updates"]==0
    assert r["native_ready"] is r["teacher_admitted"] is r["actual_tracking_validated"] is False
    assert replay["ideal_tracking_assumption"] is True
