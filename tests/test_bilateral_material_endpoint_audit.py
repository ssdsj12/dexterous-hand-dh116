import numpy as np
import pytest

def module():
    from scripts import audit_dh116_bilateral_material_endpoints as m
    return m

def test_actual_pose_contract_is_xyzw_and_rigid():
    T=module().pose_matrix([1.,2.,3.,0.,0.,0.,1.])
    np.testing.assert_array_equal(T[:3,3],[1,2,3])
    np.testing.assert_array_equal(T[:3,:3],np.eye(3))
    with pytest.raises(ValueError):module().pose_matrix([0,0,0,0,0,0,0])

def test_one_episode_and_monotonic_full_sdk_endpoints_required():
    prior=dict(run_uuid="r",env_id=0,episode_uid="e",reset_id=1,rod_uid="rod",
        physics_step=2,rod_tick=0,timestamp_s=2/480)
    now=dict(prior,physics_step=3,rod_tick=12,timestamp_s=3/480)
    module().same_endpoint_sequence(prior,now)
    for changed in (dict(now,physics_step=4),dict(now,rod_tick=24),
                    dict(now,episode_uid="other"),dict(now,timestamp_s=4/480)):
        with pytest.raises(ValueError):module().same_endpoint_sequence(prior,changed)

def test_boundary_cannot_be_promoted_to_whole_substep_history_or_slip():
    s=module().qualification()
    assert not s["contact_marker_correspondence_verified"]
    assert not s["collision_surface_model_equivalence_verified"]
    assert not s["substep_director_history_available"]
    assert s["r0_tangential_slip_m"] is None and s["physical_error_bound_m"] is None
    assert s["new_physics_steps"]==s["ppo_updates"]==0
    assert not s["teacher_admission"]

def test_final_endpoint_partition_retains_rejection_counts():
    pairs=[dict(substep=s,mesh_name=name) for s,name in [(0,"left.finger14_link"),(11,"table"),(11,"right.finger23_link")]]
    final,reasons=module().partition_pairs(pairs)
    assert final==[pairs[-1]]
    assert reasons=={"no_same_substep_director":1,"table_outside_hand_material_scope":1}
    with pytest.raises(ValueError):module().partition_pairs([dict(substep=12,mesh_name="left.finger14_link")])

def test_only_exact_archived_config_path_alias_is_allowed(tmp_path,monkeypatch):
    import hashlib
    m=module();cfg="configs/original.json";current=tmp_path/"new"/cfg;archived=tmp_path/"old"/cfg
    for p in (current,archived):p.parent.mkdir(parents=True);p.write_text("immutable")
    digest=hashlib.sha256(current.read_bytes()).hexdigest()
    monkeypatch.setattr(m.native_source,"ROOT",tmp_path/"new")
    monkeypatch.setattr(m.native_source,"CONFIG",cfg)
    monkeypatch.setattr(m.native_source,"CONFIG_SHA",digest)
    monkeypatch.setattr(m,"FROZEN_CONFIG",archived)
    expected={"version":"locked","sources":{str(current):digest,"backend":"unchanged"}}
    recorded={"version":"locked","sources":{str(archived):digest,"backend":"unchanged"}}
    assert m.archived_source(expected)==recorded
    archived.write_text("tampered")
    with pytest.raises(ValueError):m.archived_source(expected)
