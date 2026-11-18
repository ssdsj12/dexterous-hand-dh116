"""Regression of the exact immutable native record; never a new physics trial."""
import hashlib,json
from pathlib import Path
import pytest
from scripts.audit_dh116_bilateral_effective_pad_regions import validate_binding
P=Path("/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/effective-pad-region-20261114-run01/result.json")
@pytest.fixture(scope="module")
def result():
    raw=P.read_bytes()
    assert hashlib.sha256(raw).hexdigest()=="275c7f6359f668288efd8b120ba921e524c691d40bccf7321d682b71e792934d"
    return json.loads(raw)

def test_all_actual_hand_pairs_fail_original_region_without_dropping_samples(result):
    assert result["status"]=="complete"
    assert (result["physical_rows"],result["solver_rows"])==(232,2784)
    assert (result["accepted_hand_pairs"],result["rejected_hand_pairs"])==(0,2111)
    expected={"left.finger14_link":526,"left.finger23_link":495,"right.finger14_link":530,"right.finger23_link":560}
    for key,count in expected.items():
        assert result["by_pad"][key]==dict(total=count,accepted=0,rejected=count,
            **{"first:outside_inner_ellipse":count,"overlap:outside_inner_ellipse":count})

def test_first_failed_contact_preserves_native_side_and_action_clock(result):
    row=result["first_pair_by_pad"]["right.finger23_link"]
    c=row["identity"]
    assert (c["env_id"],c["physics_step"],c["policy_step"],c["substep"])==(0,188,23,1)
    assert row["solver_substep"]==4
    assert row["region"]["ellipse_value"]==pytest.approx(1.1942522477919726)
    assert row["region"]["normal_cosine_on_hand"]<-.99
    assert abs(row["region"]["plane_signed_distance_m"])<.001

def test_output_evidence_unchanged_and_no_depth_slip_success_invented(result):
    for n,sha in result["output_sha256"].items():
        assert hashlib.sha256((P.parent/n).read_bytes()).hexdigest()==sha
    for key in ("contact_depth_qualified","material_slip_qualified","positive_effective_contact_qualified",
                "native_qualification","teacher_admission"):
        assert result[key] is False
    assert result["new_physics_steps"]==result["ppo_updates"]==0

def test_binding_validates_sdf_bytes_and_original_usd_source(tmp_path):
    sdf=tmp_path/"sdf.npz";sdf.write_bytes(b"fixture-only")
    g={"source_colliders":{"finger14_link":{"source":{"geometry_sha256":"geometry"}}},
       "cooked_pads":{"finger14_link":{"capture_sha256":"capture"}},"usd":{"sha256":"usd"}}
    binding=dict(source_geometry_sha256="geometry",source_usd_sha256="usd",
        capture_sha256="capture",sdf_sha256=hashlib.sha256(sdf.read_bytes()).hexdigest(),sdf_path=str(sdf))
    assert validate_binding(binding,g,"finger14_link")["capture_sha256"]=="capture"
    for key in ("source_geometry_sha256","source_usd_sha256","capture_sha256","sdf_sha256"):
        with pytest.raises(ValueError):validate_binding(dict(binding,**{key:"foreign"}),g,"finger14_link")
    sdf.write_bytes(b"changed")
    with pytest.raises(ValueError):validate_binding(binding,g,"finger14_link")
