"""Independent vector oracles; all new contact fixtures are explicitly CPU."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from go2_pvcnn.control.dh116.tabletop_deformable_pair_source_basis import (
    analyze_pair_capture,material_station_resolution,compare_contacts,verify_contact_source,
)

NAMES=["left.finger14_link","left.finger23_link","table"]
def packet(pairs,substeps=12):
    ids=np.asarray([p[:3] for p in pairs],dtype=np.int64).reshape(-1,3)
    vectors=np.asarray([p[3:] for p in pairs],dtype=float).reshape(-1,12)
    forces=np.zeros((3,3));wrenches=np.zeros((3,6))
    for ids1,v in zip(ids,vectors):
        forces[ids1[1]]+=v[6:9]/substeps
        wrenches[ids1[2],:3]-=v[6:9]/substeps
        wrenches[ids1[2],3:]+=v[9:12]/substeps
    return dict(substeps=substeps,mesh_count=3,capacity=128,attempted=len(ids),overflow=False,
        indices=ids.tolist(),vectors=vectors.tolist(),node_forces_on_rod_world_n=forces.tolist(),
        mesh_wrenches_on_mesh_world=wrenches.tolist())

def contact(sub,node,mesh,point,normal,force):
    return [sub,node,mesh,*point,*normal,*force,0.,0.,0.]

def opposing():
    return [contact(0,1,0,[-.005,0,0],[1,0,0],[2,0,0]),
            contact(0,1,1,[.005,0,0],[-1,0,0],[-3,0,0])]

def run(raw,**kw):
    return analyze_pair_capture(raw,mesh_names=NAMES,rest_lengths_m=[.01,.01],
        rest_source="explicit CPU rest-segment fixture",radius_sample={"value":.005,"source":"CPU radius"},
        capture_source="explicit CPU pair fixture",**kw)

def test_source_sign_and_same_node_radius_identity():
    result=run(packet(opposing()))
    pair=result["substeps"][0]["comparisons"][0]
    assert pair["normal_dot"]==pytest.approx(-1)
    assert pair["force_cosine"]==pytest.approx(-1)
    assert pair["inward_normal_cosines"]==pytest.approx([1,1])
    assert pair["inward_force_projections_n"]==pytest.approx([2,3])
    assert pair["same_node_reconstructed_center_residual_m"]==pytest.approx(0)
    assert pair["geometric_opposition_sign_candidate"] is True
    assert pair["effective_opposed_contact"] is None
    assert result["effective_opposed_contact"] is result["material_anchor_slip"] is None
    assert result["native_qualified"] is False
    assert result["substeps"][1]["comparisons"]==[]

@pytest.mark.parametrize("variant",["points_reversed","parallel","zero_force","coincident"])
def test_wrong_or_undefined_direction_never_becomes_opposition(variant):
    values=opposing()
    if variant=="points_reversed":values[0][3:6],values[1][3:6]=values[1][3:6],values[0][3:6]
    elif variant=="parallel":values[1][6:9]=[1,0,0];values[1][9:12]=[3,0,0]
    elif variant=="zero_force":values[1][9:12]=[0,0,0]
    else:values[1][3:6]=values[0][3:6]
    pair=run(packet(values))["substeps"][0]["comparisons"][0]
    if variant in ("zero_force","coincident"):
        assert pair["geometric_opposition_sign_candidate"] is None
    else:assert pair["geometric_opposition_sign_candidate"] is False
    if variant=="zero_force":assert pair["force_cosine"] is None
    if variant=="coincident":assert pair["inward_force_projections_n"] is None

def test_cross_substeps_and_table_are_never_combined():
    values=opposing();values[1][0]=1
    values.append(contact(0,1,2,[0,0,-.005],[0,0,1],[0,0,10]))
    result=run(packet(values))
    assert result["comparison_count"]==0
    assert result["table_pair_count"]==1
    assert all(s["comparisons"]==[] for s in result["substeps"])

def test_different_material_nodes_keep_station_gap_and_no_shared_center_claim():
    values=opposing();values[1][1]=2
    pair=run(packet(values))["substeps"][0]["comparisons"][0]
    assert pair["material_station_gap_m"]==pytest.approx(.01)
    assert pair["same_node_reconstructed_center_residual_m"] is None
    assert pair["material_anchor_slip"] is None

def test_absent_vs_complete_empty_packets_are_distinct():
    absent=run(None);empty=run(packet([]))
    assert absent["substeps"] is None and absent["comparison_count"] is None
    assert empty["comparison_count"]==0 and len(empty["substeps"])==12
    assert absent["effective_opposed_contact"] is empty["effective_opposed_contact"] is None

@pytest.mark.parametrize("damage",["aggregates","normal","negative_force","duplicate","overflow","names","budget"])
def test_reuse_raw_pair_integrity_and_bounded_comparisons(damage):
    values=opposing()
    if damage=="normal":values[0][6:9]=[2,0,0]
    if damage=="negative_force":values[0][9:12]=[-2,0,0]
    if damage=="duplicate":values.append(values[0])
    raw=packet(values)
    if damage=="aggregates":raw["node_forces_on_rod_world_n"][0][0]=1
    if damage=="overflow":raw["overflow"]=True
    args={"max_comparisons":0} if damage=="budget" else {}
    if damage=="names":
        with pytest.raises(ValueError):
            analyze_pair_capture(raw,mesh_names=["left.fake",*NAMES[1:]],rest_lengths_m=[.01,.01],
                rest_source="fixture",radius_sample={"value":None,"source":None},capture_source="fixture")
    else:
        with pytest.raises(ValueError):run(raw,**args)

def test_material_resolution_bound_is_conditional_not_measured_slip_error():
    rest=np.array([.01,.025,.005])
    report=material_station_resolution(rest,source="CPU nonuniform rest readback")
    grid=np.linspace(0,rest.sum(),8001)
    nodes=np.r_[0,np.cumsum(rest)]
    assert np.min(abs(grid[:,None]-nodes),axis=1).max()==pytest.approx(.0125)
    assert report["nearest_station_quantization_bound_m"]==pytest.approx(.0125)
    assert report["physical_material_anchor_error_bound_m"] is None
    assert report["tangential_slip_error_bound_m"] is None
    assert report["conditional_bound_applies_to_captured_contact"] is False

def test_missing_radius_is_not_imputed():
    raw=packet(opposing())
    report=analyze_pair_capture(raw,mesh_names=NAMES,rest_lengths_m=[.01,.01],rest_source="CPU fixture",
        radius_sample={"value":None,"source":None},capture_source="CPU fixture")
    assert report["substeps"][0]["comparisons"][0]["same_node_reconstructed_center_residual_m"] is None

def test_pair_comparison_has_swap_and_rigid_transform_invariance():
    clean=run(packet(opposing()))["substeps"][0]["comparisons"][0]
    values=opposing();q=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
    for p in values:
        p[3:6]=(q@p[3:6]+[.2,-.3,1.]).tolist()
        p[6:9]=(q@p[6:9]).tolist();p[9:12]=(q@p[9:12]).tolist()
    rotated=run(packet(values))["substeps"][0]["comparisons"][0]
    for key in ("normal_dot","force_cosine","inward_normal_cosines","inward_force_projections_n"):
        assert rotated[key]==pytest.approx(clean[key])

def test_actual_locked_contact_source_and_mutation_rejection(tmp_path):
    source=Path("/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/teacher-pair-build-zQEenV/source/third_party/yarnball/KittenEngine/YarnBall/sim/contact.cuh")
    proof=verify_contact_source(source)
    assert proof["source_sha256"]=="8d90d9d18c853fbc3931a69435000726f49d36c887e62debe10225e8aa77d2bb"
    changed=tmp_path/"contact.cuh";changed.write_bytes(source.read_bytes()+b"\n")
    with pytest.raises(ValueError):verify_contact_source(changed)

def test_real_cpu_cli_preserves_evidence_and_no_native_import(tmp_path):
    root=Path(__file__).resolve().parents[1]
    bundle=dict(raw=packet(opposing()),mesh_names=NAMES,rest_lengths_m=[.01,.01],
        rest_source="CPU fixture",radius_sample={"value":.005,"source":"CPU fixture"},capture_source="CPU fixture")
    src=tmp_path/"input.json";src.write_text(json.dumps(bundle))
    out=tmp_path/"out.json"
    cmd=[sys.executable,str(root/"scripts/audit_dh116_flexible_pair_basis.py"),"--input",str(src),"--output",str(out)]
    proc=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
    assert proc.returncode==0,proc.stderr
    record=json.loads(out.read_text())
    assert record["native_imports"]==[] and record["analysis"]["comparison_count"]==1
    before=out.read_bytes()
    proc=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
    assert proc.returncode==2 and out.read_bytes()==before


@pytest.mark.parametrize("grasp",["pinch","tripod","power"])
def test_current_dual_hand_row_is_consumed_without_slot_or_clock_invention(monkeypatch,grasp):
    from test_dh116_flexible_pair_contact_diagnostics import sample
    from go2_pvcnn.control.dh116.tabletop_deformable_pair_source_basis import analyze_rod_pair_basis
    row=sample(monkeypatch)
    result=analyze_rod_pair_basis(row,grasp=grasp,radius_sample={"value":None,"source":None})
    assert result["pair_count"]==24 and result["comparison_count"]==0
    assert result["boundary_identity"]["physics_step"]==3
    assert result["boundary_identity"]["rod_tick"]==12
    assert result["effective_opposed_contact"] is None

def test_current_runtime_row_clock_mismatch_is_rejected(monkeypatch):
    from test_dh116_flexible_pair_contact_diagnostics import sample
    from go2_pvcnn.control.dh116.tabletop_deformable_pair_source_basis import analyze_rod_pair_basis
    row=sample(monkeypatch);row["rod_mesh_contact_pairs"]["physics_step"]+=1
    with pytest.raises(ValueError):
        analyze_rod_pair_basis(row,grasp="pinch",radius_sample={"value":None,"source":None})

def test_actual_comparison_budget_rejects_before_partial_result():
    values=opposing()+[contact(0,2,1,[.005,.01,0],[-1,0,0],[-2,0,0])]
    with pytest.raises(ValueError,match="exceed"):run(packet(values),max_comparisons=1)


def test_many_same_pad_rows_do_not_bypass_cpu_bound_with_quadratic_enumeration(monkeypatch):
    import go2_pvcnn.control.dh116.tabletop_deformable_pair_source_basis as basis
    original=basis.combinations
    def bounded(values,r):
        values=list(values)
        assert len(values)<=5,"pair enumeration must be across at most five mesh groups"
        return original(values,r)
    monkeypatch.setattr(basis,"combinations",bounded)
    n=1000
    raw=dict(substeps=1,mesh_count=3,capacity=n,attempted=n,overflow=False,
        indices=[[0,i,0] for i in range(n)],vectors=[[0,0,0,0,0,1,0,0,1,0,0,0]]*n,
        node_forces_on_rod_world_n=[[0,0,1]]*n,mesh_wrenches_on_mesh_world=[[0,0,-n,0,0,0],[0]*6,[0]*6])
    result=analyze_pair_capture(raw,mesh_names=NAMES,rest_lengths_m=[.001]*(n-1),
        rest_source="CPU dense-node fixture",radius_sample={"value":None,"source":None},
        capture_source="CPU dense single-pad fixture",max_comparisons=1)
    assert result["comparison_count"]==0 and result["substeps"][0]["comparisons"]==[]
