"""CPU source-law counterexamples and same-clock endpoint diagnostics."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]


def module():
    spec=importlib.util.spec_from_file_location("material_audit",ROOT/"scripts/audit_dh116_contact_material_observability.py")
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def test_transition_force_has_two_distinct_speeds_and_plateau_is_unbounded():
    m=module()
    assert m.friction_ratio(.0009,mu_s=1.,mu_k=.8,v_slip=.001)==pytest.approx(.9)
    assert m.friction_ratio(.0015,mu_s=1.,mu_k=.8,v_slip=.001)==pytest.approx(.9)
    assert m.friction_ratio(.002,mu_s=1.,mu_k=.8,v_slip=.001)==pytest.approx(.8)
    assert m.friction_ratio(3.,mu_s=1.,mu_k=.8,v_slip=.001)==pytest.approx(.8)
    proof=m.force_inversion_counterexamples(mu_s=1.,mu_k=.8,v_slip=.001,substep_dt=1/5760)
    assert proof["unique_displacement_inversion"] is False
    assert proof["physical_slip_upper_bound_m"] is None
    assert proof["plateau_same_force_displacements_m"][-1]>.0005


def pair(node=2,substep=11):
    return dict(node_index=node,mesh_index=0,substep=substep,
        contact_point_world_m=[.01,0,-.005],outward_normal_world=[0.,0.,1.],
        normal_force_on_rod_n=1.)


def row():
    nodes=np.zeros((41,3));nodes[2]=[.01,0,0]
    return dict(phase_step=1,physics_step=3,rod_tick=12,
        rod_nodes_w=nodes.tolist(),post_state=dict(link_pos=[[0.,0,0]]*5,
        link_quat=[[1.,0,0,0]]*5))


def test_final_substep_center_is_bound_to_actual_node_and_pad_frame():
    m=module()
    result=m.final_substep_endpoint(pair(),row(),radius=.005)
    assert result["node_index"]==2
    assert result["pad_local_node_m"]==pytest.approx([.01,0,0])
    assert result["pair_reconstructed_center_error_m"]==pytest.approx(0)
    assert result["physical_slip_m"] is None
    assert result["continuous_material_s_m"] is None


@pytest.mark.parametrize("fault",["wrong_node","not_final","bad_pose","nonfinite","wrong_radius"])
def test_endpoint_rejects_unknown_or_mismatched_source(fault):
    m=module();p=pair();r=row();radius=.005
    if fault=="wrong_node":p["node_index"]=3
    if fault=="not_final":p["substep"]=10
    if fault=="bad_pose":r["post_state"]["link_quat"][0]=[0.,0,0,0]
    if fault=="nonfinite":r["rod_nodes_w"][2][0]=float("nan")
    if fault=="wrong_radius":radius=.006
    with pytest.raises(ValueError):m.final_substep_endpoint(p,r,radius=radius)


def test_rigidly_moving_pad_and_node_preserve_material_centerline_coordinate():
    m=module();p=pair();r=row()
    first=m.final_substep_endpoint(p,r,radius=.005)
    r["rod_nodes_w"][2]=[.11,.2,.3]
    r["post_state"]["link_pos"][0]=[.1,.2,.3]
    p["contact_point_world_m"]=[.11,.2,.295]
    second=m.final_substep_endpoint(p,r,radius=.005)
    np.testing.assert_allclose(first["pad_local_node_m"],second["pad_local_node_m"],atol=1e-12)


def test_rod_cross_section_rotation_is_unobservable_from_node_sphere_pair():
    # The captured tuple can remain identical under different cross-section
    # material director histories: the contact kernel never receives q.
    m=module();p=pair();r=row()
    result=m.final_substep_endpoint(p,r,radius=.005)
    assert result["physical_slip_m"] is None
    proof=m.rolling_counterexample(radius=.005,angle_rad=.2)
    assert proof["identical_node_sphere_geometry"] is True
    assert proof["surface_material_travel_m"]==pytest.approx(.001)
    assert proof["exceeds_r0_limit"] is True

def test_existing_evidence_is_refused_before_reading_inputs(tmp_path,monkeypatch):
    m=module();out=tmp_path/"existing.json";out.write_text("original")
    monkeypatch.setattr(m,"audit",lambda *a,**k:pytest.fail("must refuse before audit"))
    with pytest.raises(FileExistsError):
        m.main(["--native-output","missing","--source-root","missing","--output",str(out)])
    assert out.read_text()=="original"


def test_source_hash_guard_rejects_changed_header(tmp_path,monkeypatch):
    m=module();p=tmp_path/"contact.cuh";p.write_text("changed")
    monkeypatch.setattr(m,"SOURCE_HASHES",{"contact.cuh":"0"*64})
    with pytest.raises(ValueError,match="source changed"):m.source_contract(tmp_path)
