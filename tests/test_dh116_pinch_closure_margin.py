from copy import deepcopy
import numpy as np
import pytest
from scripts import prepare_dh116_pinch_closure_margin as m


def plane():
    origin=np.array([-.002,-.002,-.002],np.float32)
    xyz=np.stack(np.meshgrid(*[origin[i]+.001*np.arange(5) for i in range(3)],indexing="ij"),-1)
    return dict(values=(xyz[...,0]+2*xyz[...,1]+3*xyz[...,2]+.01).astype(np.float32),
                origin=origin,spacing=.001)


@pytest.mark.parametrize("dtype",[np.float32,np.float64])
def test_grid_uses_xyz_and_exact_interpolant_gradient(dtype):
    p=np.array([[.00013,-.00024,.00037]],dtype)
    phi,grad,inside=m.sample_grid(plane(),p,dtype=dtype)
    assert inside.tolist()==[True]
    np.testing.assert_allclose(phi,.01+p[:,0]+2*p[:,1]+3*p[:,2],atol=2e-9,rtol=0)
    np.testing.assert_allclose(grad,[[1,2,3]],atol=2e-6,rtol=0)


def test_upper_lattice_boundary_and_missing_support_are_not_zero_gap():
    phi,grad,inside=m.sample_grid(plane(),[[.0021,0,0],[-.0021,0,0]])
    assert inside.tolist()==[False,False] and np.isnan(phi).all()
    assert np.isnan(grad).all()


def test_constant_grid_has_no_contact_normal_even_if_negative_gap():
    grid=plane();grid["values"][:]=.001
    r=m.sample_contact(grid,np.eye(4),[[0,0,0]],radius=.005)
    assert r["negative_gap_nodes"]==[0] and r["contact_predicate_nodes"]==[]


def test_measurement_anchor_is_identity_at_measured_q():
    actual=np.eye(4);actual[:3,3]=[.4,.2,.3]
    fk=np.eye(4);fk[:3,3]=[.4001,.1999,.3002]
    assert np.allclose(m.relative_pose(actual,fk,fk),actual)
    new=fk.copy();new[0,3]+=.001
    assert m.relative_pose(actual,fk,new)[0,3]==pytest.approx(.401)


def test_pinch_trajectory_preserves_quiet_hold_and_original_speed_cap():
    q=np.array([.28,.45,.66,1.15,1.15,1.15]);saved=q.copy()
    rows=m.command_path(q,[.04,.05])
    assert rows.shape==(239,6) and np.array_equal(q,saved)
    np.testing.assert_array_equal(rows[:49],np.tile(q,(49,1)))
    np.testing.assert_allclose(rows[208:],[q+[0,.04,.05,0,0,0]]*31)
    assert np.max(np.abs(np.diff(rows,axis=0)))*480<=.3+1e-12
    assert np.array_equal(rows[:,[0,3,4,5]],np.tile(q[[0,3,4,5]],(239,1)))


@pytest.mark.parametrize("delta",[[.1001,.025],[-.001,.025],[np.nan,.025]])
def test_illegal_closure_is_rejected(delta):
    with pytest.raises(ValueError):m.command_path([.28,.45,.66,1.15,1.15,1.15],delta)


def test_margin_gate_requires_both_pads_and_finite_normal():
    rows={n:dict(minimum_gap_m=-.0002,gradient_norm_at_minimum=1.) for n in m.REQUIRED}
    assert m.margin_gate(rows)
    rows[m.REQUIRED[1]]["minimum_gap_m"]=-.000099
    assert not m.margin_gate(rows)
    rows[m.REQUIRED[1]]["minimum_gap_m"]=-.0002
    rows[m.REQUIRED[1]]["gradient_norm_at_minimum"]=0
    assert not m.margin_gate(rows)


def test_nonfinite_grid_refused():
    g=plane();g["values"][0,0,0]=np.nan
    with pytest.raises(ValueError):m.sample_grid(g,[[0,0,0]])


def test_existing_output_is_never_overwritten(tmp_path):
    out=tmp_path/"out";out.mkdir();(out/"keep").write_text("original")
    with pytest.raises(FileExistsError):m.main(["--output",str(out)])
    assert (out/"keep").read_text()=="original"


def test_path_bound_keeps_static_and_runtime_thresholds_separate():
    row=m.path_gate(.002,0.001,0.002)
    assert row["static_geometry_passed"] and not row["runtime_2p1mm_passed"]
    assert not row["native_ready"] and not row["native_recipe_created"]
    assert not m.path_gate(.00189,.001,.002)["static_geometry_passed"]
    assert not m.path_gate(.002,-1e-9,.002)["static_geometry_passed"]
    assert not m.path_gate(.002,.001,-1e-9)["static_geometry_passed"]


def test_partial_preserves_completed_endpoint_records(tmp_path,monkeypatch):
    import json
    monkeypatch.setattr(m,"load_inputs",lambda p: ({},{},{},[],{},tmp_path))
    monkeypatch.setattr(m,"measured_anchors",lambda *a:{})
    monkeypatch.setattr(m,"prepare_metadata",lambda *a:{})
    monkeypatch.setattr(m,"measured_grid_audit",lambda *a:{})
    calls=[]
    def endpoint(*a):
        calls.append(1)
        if len(calls)>1:raise m.BudgetExpired()
        return {"completed_fixture":True}
    monkeypatch.setattr(m,"endpoint",endpoint)
    out=tmp_path/"out"
    assert m.main(["--output",str(out)])==0
    r=json.loads((out/"result.json").read_text())
    assert r["status"]=="partial_timeout" and r["endpoint_queries"]==[{"completed_fixture":True}]
    assert r["native_ready"] is False


def test_all_five_grid_rows_are_required_for_measured_replay():
    with pytest.raises(ValueError):
        m.measured_grid_audit({},[],{"finger14_link":plane()})


def test_refinement_changes_only_initial_spread_and_thumb_closure():
    cfg={"active_q_rad":[.28,.45,.66,1.15,1.15,1.15]}
    new,extra=m.refinement_parameters(cfg,[.006,.007],[.02,.003])
    assert cfg["active_q_rad"][0]==.28
    np.testing.assert_allclose(new["active_q_rad"],[.30,.45,.66,1.15,1.15,1.15])
    np.testing.assert_allclose(extra,[.009,.007])
    with pytest.raises(ValueError):m.refinement_parameters(cfg,[.006,.007],[.051,0])


def test_parent_refinement_rejects_changed_evidence_before_load(tmp_path):
    p=tmp_path/"result.json";p.write_text("{}")
    with pytest.raises(ValueError):m.load_refinement_parent(p)


def test_safe_suffix_requires_every_record_and_future_protection():
    rows=[dict(path_index=i,table_lower_m={"a":.002 if i>=40 else .0018},
               nonrequired_lower_m={"b":.001}) for i in range(161)]
    assert m.safe_suffix_start(rows)==40
    rows[80]["nonrequired_lower_m"]["b"]=-.001
    assert m.safe_suffix_start(rows)==81
    rows.pop(20)
    with pytest.raises(ValueError):m.safe_suffix_start(rows)


def test_suffix_cannot_relabel_changed_path_as_parent_subset():
    q=np.array([.28,.45,.66,1.15,1.15,1.15])
    start,delta=m.suffix_parameters(q,[.025,.032],40)
    np.testing.assert_allclose(start,q+[0,.00625,.008,0,0,0])
    np.testing.assert_allclose(delta,[.01875,.024])
    old=m.command_path(q,[.025,.032]);new=m.command_path(start,delta)
    np.testing.assert_allclose(new[208],old[208])
    assert np.max(abs(np.diff(new,axis=0)))*480<.3


def test_suffix_selection_buffer_cannot_relax_static_gate():
    rows=[dict(path_index=i,table_lower_m={"a":.002},nonrequired_lower_m={"b":.001}) for i in range(161)]
    with pytest.raises(ValueError):m.safe_suffix_start(rows,selection_clearance=.0018)


def test_export_cannot_promote_native_or_accept_unbound_result(tmp_path):
    p=tmp_path/"fake.json";p.write_text("{}")
    with pytest.raises(ValueError):m.export_candidate(p,tmp_path/"candidate.json")
    assert not (tmp_path/"candidate.json").exists()
