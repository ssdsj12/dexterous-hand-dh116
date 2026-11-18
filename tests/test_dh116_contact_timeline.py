from copy import deepcopy
import numpy as np
import pytest
from test_dh116_flexible_pair_contact_diagnostics import sample
from go2_pvcnn.control.dh116.tabletop_deformable_pair_source_basis import ContactTimeline
from go2_pvcnn.control.dh116.tabletop_dual_hand_rod_runtime import MESH_NAMES

def prepared(monkeypatch, mode="opposed"):
    row=sample(monkeypatch)
    pairs=[]
    for sub in range(12):
        for mesh,n,p,f in [(0,[1,0,0],[-.005,0,0],[1,0,0]),
                           (1,[-1,0,0],[.005,0,0],[-2,0,0])]:
            if mode=="alternating" and (sub%2)!=mesh:continue
            if mode=="parallel" and mesh==1:n=[1,0,0];f=[2,0,0]
            pairs.append(dict(substep=sub,node_index=0,mesh_index=mesh,mesh_name=MESH_NAMES[mesh],
                contact_point_world_m=p,outward_normal_world=n,force_on_rod_world_n=f,
                torque_on_mesh_about_origin_world_nm=[0,0,0]))
    row["rod_mesh_contact_pairs"]["pairs"]=pairs
    row["distributed_contact"]=np.zeros_like(row["distributed_contact"])
    row["named_wrenches"]={name:np.zeros(6) for name in MESH_NAMES}
    for p in pairs:
        f=np.array(p["force_on_rod_world_n"])
        row["distributed_contact"][0,p["node_index"]]+=f/12
        row["named_wrenches"][p["mesh_name"]][:3]-=f/12
    return row

def identity(row,control=0,sub=0):
    keys=("run_uuid","env_id","episode_uid","reset_id","rod_uid","physics_step","rod_tick",
          "timestamp_s","solver_time_s","clock_valid")
    return dict({k:row[k] for k in keys},policy_step=control,substep=sub)

def advance(row):
    r=deepcopy(row);r["physics_step"]+=1;r["rod_tick"]+=12
    r["timestamp_s"]=r["physics_step"]/480;r["solver_time_s"]+=1/480
    if r["rod_mesh_contact_pairs"] is not None:
        r["rod_mesh_contact_pairs"].update(physics_step=r["physics_step"],rod_tick=r["rod_tick"])
    return r

def timeline(grasp="pinch",max_rows=240):
    return ContactTimeline(grasp=grasp,radius_sample={"value":.005,"source":"CPU fixture"},
                           max_physics_rows=max_rows)

def test_opposed_candidate_keeps_exact_substep_identity_and_never_qualifies(monkeypatch):
    row=prepared(monkeypatch);t=timeline()
    out=t.record(row,action_identity=identity(row))
    assert len(out)==12
    assert out[0]["identity"]["physics_step"]==3
    assert out[0]["identity"]["solver_substep"]==0
    assert out[0]["identity"]["solver_tick"]==1
    assert out[-1]["identity"]["solver_tick"]==12
    assert out[0]["hands"]["left"]["all_required_loaded"] is True
    assert out[0]["hands"]["left"]["opposition_sign_candidate"] is True
    assert out[0]["hands"]["right"]["all_required_loaded"] is False
    s=t.summary()
    assert s["metrics"]["left.opposition_sign_candidate"]["longest_interval_s"]==pytest.approx(1/480)
    assert s["effective_opposed_contact"] is s["stable"] is s["material_anchor_slip"] is None
    assert not s["teacher_admission"] and not s["native_qualified"]
    assert s["new_physics_steps"]==0

def test_required_pad_load_is_not_opposition(monkeypatch):
    row=prepared(monkeypatch,"parallel");t=timeline();out=t.record(row,action_identity=identity(row))
    assert out[0]["hands"]["left"]["all_required_loaded"] is True
    assert out[0]["hands"]["left"]["opposition_sign_candidate"] is False
    assert out[0]["comparisons"][0]["normal_dot"]==pytest.approx(1)

def test_nonoverlapping_pad_contacts_do_not_create_same_time_opposition(monkeypatch):
    row=prepared(monkeypatch,"alternating");t=timeline();t.record(row,action_identity=identity(row))
    assert t.summary()["metrics"]["left.all_required_loaded"]["true_substeps"]==0
    assert t.summary()["comparison_count"]==0

def test_runs_continue_only_across_contiguous_identical_episode(monkeypatch):
    a=prepared(monkeypatch);b=advance(a);t=timeline()
    t.record(a,action_identity=identity(a));t.record(b,action_identity=identity(b,sub=1))
    m=t.summary()["metrics"]["left.all_required_loaded"]
    assert m["longest_interval_s"]==pytest.approx(2/480)
    assert m["first_true"]["physics_step"]==3 and m["last_true"]["physics_step"]==4

def test_missing_packet_is_unknown_and_breaks_observed_interval(monkeypatch):
    a=prepared(monkeypatch);b=advance(a);c=advance(b)
    b["rod_mesh_contact_pairs"]=None;t=timeline()
    t.record(a,action_identity=identity(a));t.record(b,action_identity=identity(b,sub=1))
    t.record(c,action_identity=identity(c,sub=2))
    m=t.summary()["metrics"]["left.all_required_loaded"]
    assert m["true_substeps"]==24 and m["unknown_substeps"]==12
    assert m["longest_interval_s"]==pytest.approx(1/480)
    assert m["true_interval_count"]==2

@pytest.mark.parametrize("damage",["episode","reset","env","rod","skip","action","solver_time","duplicate"])
def test_rejects_identity_clock_changes_without_committing_partial_rows(monkeypatch,damage):
    a=prepared(monkeypatch);b=advance(a);t=timeline();t.record(a,action_identity=identity(a))
    if damage=="episode":b["episode_uid"]+="new"
    if damage=="reset":b["reset_id"]+=1
    if damage=="env":b["env_id"]+=1
    if damage=="rod":b["rod_uid"]+="new"
    if damage=="skip":b=advance(b)
    if damage=="solver_time":b["solver_time_s"]+=.01
    if damage=="duplicate":b=deepcopy(a)
    ident=identity(b,sub=1)
    if damage=="action":ident["policy_step"]=1
    with pytest.raises(ValueError):t.record(b,action_identity=ident)
    assert t.summary()["physics_rows"]==1

def test_radius_and_budget_are_explicit_and_enforced(monkeypatch):
    a=prepared(monkeypatch);b=advance(a);t=timeline(max_rows=1)
    t.record(a,action_identity=identity(a))
    with pytest.raises(ValueError,match="budget"):t.record(b,action_identity=identity(b,sub=1))
    assert t.summary()["physics_rows"]==1

@pytest.mark.parametrize("grasp",["tripod","power"])
def test_extra_required_digits_are_not_inferred_from_two_pad_data(monkeypatch,grasp):
    a=prepared(monkeypatch);t=timeline(grasp);out=t.record(a,action_identity=identity(a))
    assert out[0]["hands"]["left"]["all_required_loaded"] is False
    assert out[0]["hands"]["left"]["opposition_sign_candidate"] is False

def test_tampered_action_clock_is_rejected_before_first_append(monkeypatch):
    a=prepared(monkeypatch);t=timeline();i=identity(a);i["rod_tick"]+=1
    with pytest.raises(ValueError):t.record(a,action_identity=i)
    assert t.summary()["physics_rows"]==0

def test_caller_cannot_mutate_retained_timeline(monkeypatch):
    a=prepared(monkeypatch);t=timeline();out=t.record(a,action_identity=identity(a))
    out[0]["hands"]["left"]["all_required_loaded"]=False
    summary=t.summary()
    assert summary["metrics"]["left.all_required_loaded"]["true_substeps"]==12
    summary["metrics"]["left.all_required_loaded"]["first_true"]["physics_step"]=999
    assert t.summary()["metrics"]["left.all_required_loaded"]["first_true"]["physics_step"]==3
