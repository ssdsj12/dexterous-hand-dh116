"""Distance reserve certificates are distinct from native contact qualification."""
import itertools
import numpy as np
import pytest
from scipy.spatial.transform import Rotation


def api():
    from scripts import audit_dh116_902_tracking_reserve
    return audit_dh116_902_tracking_reserve


def rod_fixture(x=.00498):
    rot=Rotation.from_euler("y",3.75,degrees=True).as_matrix()
    v=np.array([[a,b,c] for a in (x,x+.002) for b in (-.015,.015) for c in (-.02,.02)])@rot.T
    nodes=np.c_[np.zeros(41),np.linspace(-.2,.2,41),np.zeros(41)]
    return v,np.repeat(nodes[None],9,axis=0),rot


def test_near_threshold_rod_is_refined_without_requiring_positive_contact_gap():
    v,n,_=rod_fixture()
    out=api().rod_reserve_certificate(v,n,requested_reserve_m=.00012)
    assert out["fast_lower_m"] < -.00003
    assert out["guard_certified"] and out["requested_reserve_certified"]
    assert out["lower_m"] <= -.00002 <= out["upper_m"]
    assert out["upper_m"]-out["lower_m"] < 1e-8
    assert 0<out["qp_calls"]<=40
    assert out["native_qualification"] is False


def test_true_insufficient_reserve_and_collision_are_distinguished():
    for x,guard in [(.0049,True),(.0047,False)]:
        v,n,_=rod_fixture(x)
        out=api().rod_reserve_certificate(v,n,requested_reserve_m=.0001)
        assert out["guard_certified"] is guard
        assert not out["requested_reserve_certified"]
        assert out["upper_proves_insufficient"]


def test_earlier_frames_cannot_be_replaced_by_last_frame_witness():
    v,n,rot=rod_fixture()
    n[:8]+=.001*rot[:,0]
    out=api().rod_reserve_certificate(v,n,requested_reserve_m=.0001)
    assert not out["requested_reserve_certified"]
    assert not out["guard_certified"]
    assert out["frame_count"]==9


def test_box_lower_bound_is_tightened_without_changing_original_floor():
    half=.01
    v=np.array(list(itertools.product((-half,half),repeat=3)))
    rot=Rotation.from_euler("z",45,degrees=True).as_matrix()
    centre=.01+np.sqrt(2)*half+.001000245/np.sqrt(2)
    v=v@rot.T+np.array([centre,centre,0.])
    out=api().box_reserve_certificate(v,[-.01]*3,[.01]*3,requested_reserve_m=.001)
    assert abs(out["fast_lower_m"]-.001000245)<1e-10
    assert out["lower_m"]>.01
    assert out["guard_certified"] and out["requested_reserve_certified"]
    assert out["lower_m"]<=out["upper_m"]
    out=api().box_reserve_certificate(np.array(list(itertools.product((-half,half),repeat=3)))+[.0205,0,0],
        [-.01]*3,[.01]*3,requested_reserve_m=.001)
    assert not out["guard_certified"] and out["upper_proves_insufficient"]

@pytest.mark.parametrize("field",["missing_frame","nan","degenerate","negative_reserve","infinite_reserve","budget41","bool_budget"])
def test_invalid_complete_geometry_or_budget_is_rejected(field):
    v,n,_=rod_fixture();opts=dict(requested_reserve_m=.0001)
    if field=="missing_frame":n=n[:-1]
    if field=="nan":n[0,0,0]=np.nan
    if field=="degenerate":v[:,2]=0.
    if field=="negative_reserve":opts["requested_reserve_m"]=-1.
    if field=="infinite_reserve":opts["requested_reserve_m"]=np.inf
    if field=="budget41":opts["max_qp_calls"]=41
    if field=="bool_budget":opts["max_qp_calls"]=True
    with pytest.raises(ValueError):api().rod_reserve_certificate(v,n,**opts)


def test_budget_or_failed_optimizer_does_not_grant_reserve(monkeypatch):
    v,n,_=rod_fixture()
    out=api().rod_reserve_certificate(v,n,requested_reserve_m=.00012,max_qp_calls=0)
    assert out["qp_calls"]==0 and out["reserve_status"]=="unknown"
    def failure(*a):raise ValueError("no independent certificate")
    monkeypatch.setattr(api(),"segment_hull_bounds",failure)
    out=api().rod_reserve_certificate(v,n,requested_reserve_m=.00012)
    assert not out["requested_reserve_certified"]
    assert out["reserve_status"]=="unknown" and out["qp_calls"]==1


def test_translation_preserves_full_frame_certificate():
    v,n,_=rod_fixture();shift=np.array([4.,-.8,.9])
    a=api().rod_reserve_certificate(v,n,requested_reserve_m=.00012)
    b=api().rod_reserve_certificate(v+shift,n+shift,requested_reserve_m=.00012)
    assert a["reserve_status"]==b["reserve_status"]
    assert abs(a["lower_m"]-b["lower_m"])<1e-10
    assert abs(a["upper_m"]-b["upper_m"])<1e-10


def test_failed_bound_interval_cannot_be_accepted():
    with pytest.raises(ValueError):api()._summary(.01,.009,-.00015,.001)
    for value in [np.nan,np.inf,-np.inf]:
        with pytest.raises(ValueError):api()._summary(value,.009,-.00015,.001)


def test_witness_cli_rejects_changed_file_before_geometry(tmp_path):
    packet=tmp_path/"witness.json";packet.write_text("{}")
    output=tmp_path/"result.json"
    with pytest.raises(ValueError,match="SHA"):
        api().main(["--witnesses",str(packet),"--sha256","a"*64,"--output",str(output)])
    assert not output.exists()

def test_witness_cli_keeps_sample_scope_and_binds_distance_implementations(tmp_path):
    import json,hashlib
    v,n,_=rod_fixture()
    names=[group+":"+pad for group in ("source_colliders","cooked_pads")
        for pad in ("finger14_link","finger23_link","finger33_link")]
    shapes={name:v.tolist() for name in names}
    shapes.update({"aux"+str(i):(v+[1.,0.,0.]).tolist() for i in range(16)})
    witness=dict(ordinal=1,shapes=shapes,rod_samples=n.tolist(),
        boxes=[([-.1,-.1,-1.],[.1,.1,-.9]),([2.,2.,2.],[2.1,2.1,2.1])])
    packet=tmp_path/"input.json";output=tmp_path/"result.json"
    packet.write_text(json.dumps(dict(scope="staged_ideal_command_CPU_only_not_native_receipt",
        SDK_substeps=1032,witness_geometry=[witness])))
    digest=hashlib.sha256(packet.read_bytes()).hexdigest()
    assert api().main(["--witnesses",str(packet),"--sha256",digest,"--output",str(output)])==0
    result=json.loads(output.read_bytes())
    assert result["sampled_substeps"]==1 and result["total_substeps"]==1032
    assert not result["full_path_certified"] and not result["robust_route_qualified"]
    assert not result["teacher_admission"] and not result["native_qualification"]
    assert {"audit_dh116_902_tracking_reserve.py","audit_dh116_certified_segment_distance.py",
            "audit_dh116_pinch_initial_pose.py","screen_dh116_tripod_supported_reset.py"} <= set(result["source_sha256"])
