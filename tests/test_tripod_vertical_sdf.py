import importlib
import json
import numpy as np
import pytest

def module():
    name="scripts.audit_dh116_tripod_vertical_sdf"
    assert importlib.util.find_spec(name) is not None, "source-bound Tripod vertical audit missing"
    return importlib.import_module(name)

def grid():
    return dict(values=np.indices((3,3,3))[2].astype(float), origin=np.zeros(3), spacing=1.)

def test_all_nodes_preserved_including_outside_domain():
    a=np.array([[.5,.5,1.5],[10,10,10.]])
    b=a.copy();b[:,2]-=1
    r=module().audit_paths({"pad":grid()},{"pad":np.eye(4)},a,b,.1)
    assert len(r["nodes"]) == 2
    assert r["nodes"][1]["classification"] == "outside_sampler_lattice"
    assert r["minimum_reference_gap_m"] == pytest.approx(.4)
    assert r["native_ready"] is False
    assert r["physics_steps"] == 0

def test_cpu_contact_witness_never_means_native_contact():
    a=np.array([[.5,.5,.2]]);b=np.array([[.5,.5,.01]])
    r=module().audit_paths({"pad":grid()},{"pad":np.eye(4)},a,b,.1)
    row=r["nodes"][0]
    assert row["classification"] == "cpu_contact_witness"
    assert row["float32_witness"]["contact_predicate"] is True
    assert r["reference_no_contact"] is False
    assert r["teacher_admitted"] is False

def test_near_zero_reference_is_not_cleared_by_guard():
    r=module().audit_paths({"pad":grid()},{"pad":np.eye(4)},
                          np.array([[.5,.5,.100001]]),np.array([[.5,.5,.100001]]),.1)
    assert r["nodes"][0]["classification"] == "reference_margin_unresolved"

def test_world_transform_does_not_change_local_sdf_measurement():
    from scipy.spatial.transform import Rotation
    p=np.eye(4);p[:3,:3]=Rotation.from_rotvec([.2,.4,.6]).as_matrix();p[:3,3]=[3,7,9]
    a=np.array([[.5,.5,1.5]]);b=np.array([[.5,.5,.5]])
    r=module().audit_paths({"pad":grid()},{"pad":np.eye(4)},a,b,.1)
    s=module().audit_paths({"pad":grid()},{"pad":p},a@p[:3,:3].T+p[:3,3],b@p[:3,:3].T+p[:3,3],.1)
    assert s["minimum_reference_gap_m"]==pytest.approx(r["minimum_reference_gap_m"],abs=1e-12)

def test_nonrigid_transform_is_rejected():
    p=np.eye(4);p[0,0]=2
    with pytest.raises(ValueError):
        module().audit_paths({"pad":grid()},{"pad":p},np.zeros((1,3)),np.zeros((1,3)),.1)

def test_mismatched_nodes_and_pad_identities_rejected():
    with pytest.raises(ValueError):
        module().audit_paths({"pad":grid()},{},np.zeros((1,3)),np.zeros((1,3)),.1)
    with pytest.raises(ValueError):
        module().audit_paths({"pad":grid()},{"pad":np.eye(4)},np.zeros((1,3)),np.zeros((2,3)),.1)

def test_source_digest_mismatch_rejected(tmp_path):
    p=tmp_path/"evidence.json";p.write_text("{}")
    with pytest.raises(ValueError):module().checked_json(p,"f"*64)

def test_existing_output_is_not_overwritten(tmp_path):
    p=tmp_path/"run";p.mkdir();(p/"result.json").write_text("original")
    with pytest.raises(FileExistsError):module().main(["--output",str(p)])
    assert (p/"result.json").read_text()=="original"
