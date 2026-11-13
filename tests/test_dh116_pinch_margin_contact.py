"""Independent CPU tests; no Isaac or native qualification."""
import importlib.util
from pathlib import Path
from copy import deepcopy
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
def api():
    assert importlib.util.find_spec("scripts.prepare_dh116_pinch_margin_contact"), "new bounded contact operation missing"
    from scripts import prepare_dh116_pinch_margin_contact as m
    return m

def test_234_step_schedule_only_changes_left_two_fingers():
    m=api();a=m.action_schedule().numpy()
    assert (m.SETUP_STEPS,m.MEASURED_STEPS,m.TOTAL_STEPS)==(2,232,234)
    assert a.shape==(29,24) and np.all(a[:6]==0) and np.all(a[26:]==0)
    assert np.all(a[:,[i for i in range(24) if i not in (1,2)]]==0)
    p=m.load_pair()
    delta=np.array(p["left"]["candidate"]["active_delta_rad"])
    np.testing.assert_allclose(a[:,0:6].sum(0)*.3*8/480,delta,rtol=0,atol=1e-16)
    assert np.max(np.abs(a))<=1

def test_new_reset_uses_real_proof_right_rod_and_coupling_unchanged():
    m=api();p=m.load_pair();old=m.read_locked(m.OLD_PAIR,m.OLD_PAIR_SHA)
    from go2_pvcnn.control.dh116.linear_coupling import expand
    c=p["left"]["candidate"]
    assert p["right"]==old["right"] and p["rod_seed"]==old["rod_seed"]
    for field in ("hand_roots","authored_root_seed_w","reset_root_pose_w"):
        assert p["reset_seed"][field][0]==c["base_pose_wxyz"]
        assert p["reset_seed"][field][1]==old["reset_seed"][field][1]
    np.testing.assert_array_equal(p["reset_seed"]["joint_positions"][0],expand(c["active_q_rad"]))
    assert p["reset_seed"]["joint_positions"][1]==old["reset_seed"]["joint_positions"][1]
    assert p["wrist_clearance_floor_m"]==.0021 and not p["native_qualified"]
    assert not p["teacher_admission"] and p["reset_seed"]["sha256"]!=old["reset_seed"]["sha256"]

@pytest.mark.parametrize("gap,valid",[(0.,True),(-.000299,True),(-.000301,False),(None,False)])
def test_depth_limit_is_additional_and_never_relaxes_original(gap,valid):
    m=api();row=dict(minimum_gap_m=gap,gradient_norm_at_minimum=1.,in_lattice_nodes=[3])
    assert m.required_sdf_valid(row) is valid
    row["gradient_norm_at_minimum"]=0
    assert m.required_sdf_valid(row) is False

def test_scope_restores_all_old_defaults_even_after_failure():
    m=api()
    from scripts import prepare_dh116_dual_pinch_closure_interval as closure
    from go2_pvcnn.control.dh116 import tabletop_dual_pinch_builder as b
    from go2_pvcnn.control.dh116 import tabletop_measured_closure_geometry as g
    original=(b.PAIR,b.PAIR_SHA,b.SEED_SHA,closure.TOTAL_STEPS,closure.action_schedule,g.required_sdf_valid)
    with pytest.raises(RuntimeError,match="injected"):
        with m.scenario_scope(ROOT):
            assert b.load_pair_proposal()==m.load_pair()
            assert closure.TOTAL_STEPS==234
            assert closure.action_schedule().shape==(29,24)
            assert g.required_sdf_valid is m.required_sdf_valid
            raise RuntimeError("injected")
    assert original==(b.PAIR,b.PAIR_SHA,b.SEED_SHA,closure.TOTAL_STEPS,closure.action_schedule,g.required_sdf_valid)

def test_independent_entry_has_exact_parent_and_closed_budget():
    assert importlib.util.find_spec("scripts.verify_dh116_pinch_margin_contact"), "new worker missing"
    from scripts import verify_dh116_pinch_margin_contact as entry
    c=entry.preflight(ROOT)
    assert c["physics_steps"]==234 and c["parameter_endpoint_count"]==233
    assert c["native_parent_receipt_sha256"]=="f0767fc970b02d0e4024f57558982eb01bd3c7c6a9698e2289adef71f4974901"
    assert c["native_execution_performed"] is False

def test_scoped_audit_preserves_all_original_checks_except_explicit_budget():
    import ast,inspect
    from scripts import dh116_pinch_margin_contact_audit as new
    from scripts import validate_dh116_dual_pinch_closure_interval as old
    source=inspect.getsource(old.validate_artifacts)
    source=source.replace("prepare_dh116_dual_pinch_closure_interval as operation",
                          "prepare_dh116_pinch_margin_contact as operation")
    source=source.replace("(2,16,18)","(2,232,234)")
    for field in ("physics_steps","expected_steps","total_steps"):
        source=source.replace(field+"=18",field+"=234")
    for field in ("measurement_steps","rod_steps","physics_rows","measurement_physics_steps"):
        source=source.replace(field+"=16",field+"=232")
    assert ast.dump(ast.parse(source))==ast.dump(ast.parse(inspect.getsource(new.validate_artifacts)))

@pytest.mark.parametrize("field",["right","rod_seed","reset_seed","sources"])
def test_frozen_config_rejects_modified_proposal(tmp_path,field):
    import json
    m=api();p=m.load_pair();p[field]={}
    f=tmp_path/m.CONFIG;f.parent.mkdir(parents=True);f.write_text(json.dumps(p))
    with pytest.raises(ValueError,match="frozen"):
        m.load_pair(tmp_path)
