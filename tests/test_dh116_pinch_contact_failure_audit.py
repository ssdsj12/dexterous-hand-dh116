import copy
import pytest

def rows():
    def make(step):
        ident=dict(run_uuid="test",env_id=0,episode_uid="test:1",reset_id=1,rod_uid="rod0",
                   physics_step=step,rod_tick=(step-2)*12,timestamp_s=step/480)
        return dict(physics_step=step,policy_step=(step-3)//8,substep=(step-3)%8,
            guard_failure=None,original_guard_failure=None,
            rod=dict(ident,rod_mesh_contact_pairs=dict(substep_dt_s=1/5760,pairs=[]),
                pair_force_residual_n=0),
            action_readback=dict(post=dict(identity=[dict(ident,policy_step=(step-3)//8,substep=(step-3)%8)])),
            closure_geometry=dict(failure_reasons=[],sides={s:dict(
                minimum_table_clearance_m=.003,minimum_non_target_gap_lower_m=.001)
                for s in ("left","right")}))
    return [make(3),make(4)]

def pair(mesh,substep=0):
    return dict(mesh_name=mesh,substep=substep,node_index=1,
        force_on_rod_world_n=[0,0,1],normal_force_on_rod_n=1,
        impulse_on_rod_ns=[0,0,1/5760])

def audit(data):
    from scripts.audit_dh116_pinch_contact_failure import summarize_exchanges
    return summarize_exchanges(data)

def test_sdf_prediction_does_not_count_as_contact():
    out=audit(rows())
    assert out["hand_contact_pairs"]=={} and out["both_left_contact_solver_steps"]==0
    assert out["teacher_admission"] is False

def test_counts_pairs_and_solver_substeps_separately():
    data=rows();pairs=data[0]["rod"]["rod_mesh_contact_pairs"]["pairs"]
    pairs.extend([pair("left.finger14_link"),pair("left.finger23_link"),
                  pair("left.finger14_link"),pair("table")])
    out=audit(data)
    assert out["hand_contact_pairs"]["left.finger14_link"]["count"]==2
    assert out["both_left_contact_solver_steps"]==1
    assert out["both_left_contiguous_s"]==pytest.approx(1/5760)
    assert out["table_pair_count"]==1

@pytest.mark.parametrize("field",["timestamp_s","episode_uid","physics_step"])
def test_cross_identity_or_bad_clock_rejected(field):
    data=rows();data[1]["rod"][field]=0
    with pytest.raises(ValueError):audit(data)

def test_duplicate_or_skipped_physical_step_rejected():
    data=rows();data[1]=copy.deepcopy(data[0])
    with pytest.raises(ValueError):audit(data)

def test_terminal_cannot_be_overwritten_by_later_noop():
    data=rows();data[0]["guard_failure"]="closure_geometry_guard"
    with pytest.raises(ValueError):audit(data)

def test_force_impulse_units_checked():
    data=rows();p=pair("left.finger14_link");p["impulse_on_rod_ns"]=[0,0,1]
    data[0]["rod"]["rod_mesh_contact_pairs"]["pairs"]=[p]
    with pytest.raises(ValueError):audit(data)

def test_unregistered_mesh_cannot_be_promoted():
    data=rows();data[0]["rod"]["rod_mesh_contact_pairs"]["pairs"]=[pair("left.fake")]
    with pytest.raises(ValueError):audit(data)
