import numpy as np
import pytest
from scipy.spatial import ConvexHull
from scripts import audit_dh116_power_alternative_sdf_interval as m


def grid(value=None):
    origin=np.array([-.002,-.002,-.002],np.float32)
    xyz=np.stack(np.meshgrid(*[origin[i]+.001*np.arange(5) for i in range(3)],indexing="ij"),-1)
    values=(xyz[...,0]+.005).astype(np.float32) if value is None else np.full((5,5,5),value,np.float32)
    return dict(values=values,origin=origin,spacing=.001)


def test_contact_interpolant_region_contains_all_negative_sample_points():
    from scripts.prepare_dh116_pinch_closure_margin import sample_grid
    g=grid();r=m.contact_region(g,.005)
    p=np.array([[-.0017,.0003,.0004],[-.0001,-.0004,.0002]])
    phi,_,inside=sample_grid(g,p,dtype=np.float64)
    assert inside.all() and np.all(phi<.005)
    planes=ConvexHull(r["vertices"]).equations
    assert np.max(p@planes[:,:3].T+planes[:,3])<=1e-10


def test_full_cell_is_included_when_only_one_corner_can_cross_radius():
    g=grid(.02);g["values"][2,2,2]=.001
    r=m.contact_region(g,.005)
    assert r["active_cell_count"]==8
    v=np.array(r["vertices"])
    np.testing.assert_allclose(v.min(0),[-.001]*3,atol=1e-9)
    np.testing.assert_allclose(v.max(0),[.001]*3,atol=1e-9)


def test_all_far_grid_has_no_contact_region():
    r=m.contact_region(grid(.02),.005)
    assert r["active_cell_count"]==0 and r["vertices"]==[]


def test_region_uses_radius_once_not_subtracting_it_again():
    r=m.interval_certificate(np.linspace(0,.1,9),[.001]*9,.008)
    assert r["continuous_axis_to_contact_region_lower_m"]==pytest.approx(.000498)
    assert r["native_ready"] is False and r["fixed_root_contact_excluded"]


def test_nine_positive_samples_are_not_a_continuous_exclusion_without_travel_margin():
    r=m.interval_certificate(np.linspace(0,.1,9),[.0002]*9,.008)
    assert not r["fixed_root_contact_excluded"]


@pytest.mark.parametrize("amounts",[np.linspace(0,.2,9),np.linspace(.01,.1,9),[0,.1]])
def test_interval_keeps_original_velocity_duration_and_full_coverage(amounts):
    with pytest.raises(ValueError):m.interval_certificate(amounts,[.001]*len(amounts),.001)


def test_nonfinite_grid_refused():
    g=grid();g["values"][0,0,0]=np.nan
    with pytest.raises(ValueError):m.contact_region(g,.005)


def test_selection_is_nonrepeated_original_orientation():
    def row(i,gap):
        return dict(archive_start_index=i,mode="minimum_table_clearance_root",
            initial_rod_settling_gaps_m={"source":{"a":.001},"cooked":{"a":gap}})
    assert m.select_unused([row(15,.001),row(6,.01),row(2,.03)],"left")["archive_start_index"]==6
    assert m.select_unused([row(1,.001),row(15,.002),row(6,.01)],"right")["archive_start_index"]==6


def test_output_exclusive(tmp_path):
    with pytest.raises(FileExistsError):m.main(["--output",str(tmp_path)])


def test_only_repeated_or_initially_overlapping_seeds_cannot_be_selected():
    row=dict(archive_start_index=15,mode="minimum_table_clearance_root",
             initial_rod_settling_gaps_m={"source":{"a":.001},"cooked":{"a":.001}})
    with pytest.raises(ValueError):m.select_unused([row],"left")
    row["archive_start_index"]=6;row["initial_rod_settling_gaps_m"]["source"]["a"]=-.001
    with pytest.raises(ValueError):m.select_unused([row],"left")


def test_partial_setup_never_grants_reset(tmp_path,monkeypatch):
    import json
    def checked(path,sha):
        project=m.Path(__file__).resolve().parents[1]
        return {"preparation_source_sha256":m.digest(project/"scripts/prepare_dh116_flexible_style_poses.py"),
            "geometry_audit_sha256":m.digest(project/"scripts/audit_dh116_pinch_initial_pose.py")}
    monkeypatch.setattr(m,"checked_json",checked)
    monkeypatch.setattr(m,"load_power_archive",lambda *a:{})
    monkeypatch.setattr(m,"load_geometry",lambda *a,**k:(_ for _ in ()).throw(m.BudgetExpired()))
    out=tmp_path/"partial"
    assert m.main(["--output",str(out)])==0
    r=json.loads((out/"result.json").read_text())
    assert r["status"]=="partial_timeout" and not r["reset_proposal_created"]
    assert not r["native_ready"] and r["completed_pad_samples"]==[]
