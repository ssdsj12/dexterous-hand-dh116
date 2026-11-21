import hashlib
import json

import numpy as np
import pytest

from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import encode_owned
from go2_pvcnn.control.dh116.tabletop_supported_rod_reference import load_parked_reference


def write(path, data):
    path.write_text(json.dumps(encode_owned(data)))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive(tmp_path, *, signal=None, bad_pair=False):
    out=tmp_path/"native"
    out.mkdir()
    identity=dict(run_uuid="run", env_id=0, episode_uid="run:0", reset_id=1, rod_uid="rod0")
    scene=dict(reset_clock=identity, supports=[
        {"name":"negative_y","center_m":[.436,-.2385,.33],"size_m":[.024,.01,.06]},
        {"name":"positive_y","center_m":[.436,.2385,.33],"size_m":[.024,.01,.06]},
    ])
    pair_names=["support_0","support_1"]
    if bad_pair:
        pair_names[0]="left.finger14_link"
    pairs=[dict(substep=i, mesh_name=name, node_index=1 if name=="support_0" else 39, normal_force_on_rod_n=.04)
           for i in range(12) for name in pair_names]
    positions=np.zeros((1,41,3))
    positions[0,:,1]=np.linspace(-.25,.25,41)
    positions[0,:,2]=.365
    velocities=np.full_like(positions,.01)
    rest=np.full(40,.0125)
    row=dict(rod=dict(**identity, physics_step=240, rod_tick=2856, timestamp_s=.5,
        clock_valid=True, pair_capture_status="validated_solver_samples",
        node_positions=positions, velocities=velocities, rest_lengths_m=rest,
        rod_mesh_contact_pairs=dict(physics_step=240,rod_tick=2856,pairs=pairs)))
    files={"power_support_scene.json":scene, "power_exchange_240.json":row,
        "power_shared_result.json":dict(scope="measured_power_support_settling_without_hand", sdk_steps=240,
        measured_exchanges=238, solver_substeps=2856),
        "result.json":dict(physics_steps=240,failure=None,lifetime_released_before_close=True)}
    manifest={name:write(out/name,data) for name,data in files.items()}
    receipt=tmp_path/"outer.json"
    digest=write(receipt,dict(exit_code=0, debugger_inferior_exit_code=0, debugger_inferior_signal=signal,
        debugger_status_verified=True,forced_exit=False,exceeded_budget=False,artifact_sha256=manifest))
    return receipt,digest


def test_loads_positions_velocities_and_original_rest_lengths_without_a_reset(tmp_path):
    path, digest = archive(tmp_path)
    ref=load_parked_reference(path,digest,240)
    assert ref.physics_step == 240 and ref.rod_tick == 2856
    assert ref.node_positions.shape == ref.node_velocities.shape == (41,3)
    assert not ref.node_positions.flags.writeable
    assert np.all(ref.node_velocities==.01)
    assert np.all(ref.rest_lengths_m==.0125)


def test_rejects_changed_or_mislabeled_native_source(tmp_path):
    path, digest=archive(tmp_path, bad_pair=True)
    with pytest.raises(ValueError, match="isolated support pairs"):
        load_parked_reference(path,digest,240)
    (tmp_path/"native/power_exchange_240.json").write_text("{}")
    with pytest.raises(ValueError, match="changed recorded artifact"):
        load_parked_reference(path,digest,240)


def test_requires_outer_natural_exit(tmp_path):
    path,digest=archive(tmp_path,signal=11)
    with pytest.raises(ValueError, match="natural exit"):
        load_parked_reference(path,digest,240)


def test_explicit_tripod_source_keeps_identity_clock_and_contact_guards(tmp_path):
    path,_=archive(tmp_path)
    outer=json.loads(path.read_bytes());out=tmp_path/"native"
    for suffix in ["support_scene.json","exchange_240.json","shared_result.json"]:
        old="power_"+suffix;new="tripod_"+suffix
        row=json.loads((out/old).read_bytes())
        if suffix=="shared_result.json":row["scope"]="measured_tripod_support_settling_without_hand"
        outer["artifact_sha256"][new]=write(out/new,row)
    digest=write(path,outer)
    ref=load_parked_reference(path,digest,240,task="tripod")
    assert ref.node_positions.shape==(41,3)
    with pytest.raises(ValueError):
        load_parked_reference(path,digest,240,task="wrong")
