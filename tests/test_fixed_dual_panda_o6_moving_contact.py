from __future__ import annotations

import hashlib
from pathlib import Path
from types import MethodType

import numpy as np
import open3d as o3d
import pytest


EXTERNAL = Path(__file__).resolve().parents[1]/"vendor/DeformX"
if not EXTERNAL.is_dir():
    pytest.skip("PyElastica-Mesh checkout is absent", allow_module_level=True)
pytest.MonkeyPatch().syspath_prepend(str(EXTERNAL))

from co_sim.models import CoSimConfig
from co_sim.moving_contact import MeshAsset, MovingContactRodEngine
from co_sim.moving_contact_models import MeshMotion
from co_sim.table_rod import TableRodEngine


CHANNELS = ("left_thumb", "left_index", "right_thumb", "right_index")


def test_restore_fresh_contact_backend_without_replaying_impulses(assets):
    def make(**kw):
        return MovingContactRodEngine(_config(),table_center=(0,0,-1),
            table_size=(2,2,.1),meshes=assets,**kw)
    source=make()
    motions={c:_motion(position=(10+i,10,10)) for i,c in enumerate(CHANNELS)}
    source.advance_with_meshes(motions,.001)
    saved=source.export_physical_state()
    restored=make();restored.restore_physical_state(saved)
    assert restored._motion_cache=={}
    assert restored._active_motions is None
    assert restored._accumulators is None
    assert restored.last_contact_diagnostics is None
    assert restored._prior_rod_force is None
    assert not restored.rod.external_forces.any()
    _,impulses=restored.advance_with_meshes(motions,.001)
    assert all(np.linalg.norm(i.linear_w)==0 for i in impulses.values())
    with pytest.raises(RuntimeError):source.restore_physical_state(saved)
    with pytest.raises(ValueError):make(contact_k=12345.).restore_physical_state(saved)


def test_restored_contact_response_matches_continuing_rod(assets):
    def make():return MovingContactRodEngine(_config(py_dt=1e-5),table_center=(0,0,-1),
        table_size=(2,2,.1),meshes=assets,contact_subsamples=16)
    source=make()
    motions={c:_motion(position=(10+i,10,10)) for i,c in enumerate(CHANNELS)}
    motions['left_thumb']=_motion(position=(0,0,.183))
    _,first=source.advance_with_meshes(motions,.0001)
    assert first['left_thumb'].peak_force_n>0
    restored=make();restored.restore_physical_state(source.export_physical_state())
    a,ia=source.advance_with_meshes(motions,.0001)
    b,ib=restored.advance_with_meshes(motions,.0001)
    for field in ('rod_position','rod_velocity','rod_director','rod_omega'):
        np.testing.assert_allclose(a[field],b[field],atol=1e-12,rtol=1e-10)
    for channel in CHANNELS:
        np.testing.assert_allclose(ia[channel].linear_w,ib[channel].linear_w,atol=1e-12,rtol=1e-10)


def test_contact_broad_phase_preserves_integrated_state_and_impulses(assets):
    engines=[MovingContactRodEngine(_config(py_dt=1e-5),table_center=(0,0,-1),
        table_size=(2,2,.1),meshes=assets,contact_subsamples=16,contact_broad_phase=b) for b in (False,True)]
    motions={c:_motion(position=(10+i,10,10)) for i,c in enumerate(CHANNELS)}
    motions['left_thumb']=_motion(position=(0,0,.183),velocity_w=np.array([.01,.02,0]))
    positive=False
    for _ in range(10):
        a,ia=engines[0].advance_with_meshes(motions,.0001)
        b,ib=engines[1].advance_with_meshes(motions,.0001)
        for field in ('rod_position','rod_velocity','rod_director','rod_omega'):
            np.testing.assert_array_equal(a[field],b[field])
        for c in CHANNELS:
            np.testing.assert_array_equal(ia[c].linear_w,ib[c].linear_w)
            np.testing.assert_array_equal(ia[c].angular_com_w,ib[c].angular_com_w)
            assert ia[c].contact_duration_s==ib[c].contact_duration_s
            ga=engines[0].contact_geometry(c);gb=engines[1].contact_geometry(c)
            np.testing.assert_array_equal(ga['gaps'],gb['gaps'])  # observations remain exact, no sentinel
        positive|=ia['left_thumb'].peak_force_n>0
    assert positive


@pytest.mark.parametrize('value',[1,None,'yes'])
def test_contact_broad_phase_rejects_nonboolean(value):
    with pytest.raises(ValueError,match='contact_broad_phase'):
        TableRodEngine(_config(),table_center=(0,0,-1),table_size=(2,2,.1),contact_broad_phase=value)


def test_refined_contact_engine_geometry_matches_rigid_comotion(assets):
    engine=MovingContactRodEngine(_config(),table_center=(0,0,-1),table_size=(2,2,.1),
        meshes=assets,contact_subsamples=16)
    assert engine.contact_subsamples==16
    body=engine.mesh_bodies['left_thumb']
    omega=np.array([.3,-.2,.1]);translation=np.array([.01,.02,-.03])
    engine.rod.velocity_collection[:]=translation[:,None]+np.cross(omega,engine.rod.position_collection.T).T
    engine.rod.omega_collection[:]=np.einsum('ijk,j->ik',engine.rod.director_collection,omega)
    body.velocity_collection[:,0]=translation+np.cross(omega,body.position_collection[:,0])
    body.omega_collection[:,0]=body.director_collection[:,:,0]@omega
    g=engine.contact_geometry('left_thumb')
    assert g['gaps'].shape==(64,)
    np.testing.assert_array_equal(g['element_ids'],np.repeat(np.arange(4),16))
    np.testing.assert_allclose(g['relative_normal_speeds'],0,atol=1e-12)
    assert engine.rod.position_collection.shape==(3,5)  # no extra dynamic DOFs


@pytest.mark.parametrize('count',[0,True,1.2,129])
def test_refined_contact_engine_rejects_invalid_count(count):
    with pytest.raises(ValueError,match='contact_subsamples'):
        TableRodEngine(_config(),table_center=(0,0,-1),table_size=(2,2,.1),contact_subsamples=count)


def test_refinement_reaches_registered_pad_contacts_not_only_observer(assets):
    peaks=[]
    for count in (1,16):
        engine=MovingContactRodEngine(_config(),table_center=(0,0,-1),table_size=(2,2,.1),
            meshes=assets,contact_subsamples=count)
        motions={c:_motion(position=(10+i,10,10)) for i,c in enumerate(CHANNELS)}
        motions['left_thumb']=_motion(position=(0,0,.183))
        _,impulses=engine.advance_with_meshes(motions,.001)
        peaks.append(impulses['left_thumb'].peak_force_n)
        assert all(impulses[c].peak_force_n==0 for c in CHANNELS[1:])
    assert peaks[0]==0 and peaks[1]>0


def test_contact_diagnostics_bound_every_integrator_position_segment(assets):
    engine=MovingContactRodEngine(_config(),table_center=(0,0,-1),table_size=(2,2,.1),meshes=assets)
    motions={c:_motion(position=(10+i,10,10)) for i,c in enumerate(CHANNELS)}
    start=engine.snapshot()['rod_position']
    sample,impulses=engine.advance_with_meshes(motions,.001)
    d=engine.last_contact_diagnostics
    assert d['rod_path_bound_m']>=np.linalg.norm(sample['rod_position']-start,axis=0).max()-1e-12
    assert d['rod_path_bound_m']>0
    assert d['max_rod_radius_m']>=float(engine.rod.radius.max())
    assert d['position_segment_samples']==21  # two position drifts per fine step, plus initial


def test_motion_prediction_reuses_same_time_immutable_result(monkeypatch):
    import co_sim.moving_contact as module
    original=module.predict_motion;calls=[]
    def counted(value,dt):
        calls.append(dt);return original(value,dt)
    monkeypatch.setattr(module,'predict_motion',counted)
    engine=MovingContactRodEngine.__new__(MovingContactRodEngine)
    engine._active_motions={'left_thumb':_motion(velocity_w=np.array([.01,0,0]))}
    engine._call_start=1.;engine._motion_cache={}
    a=engine._motion_at('left_thumb',1.1);b=engine._motion_at('left_thumb',1.1)
    assert a is b and len(calls)==1
    np.testing.assert_array_equal(a.position_w,original(engine._active_motions['left_thumb'],1.1-engine._call_start).position_w)
    engine._motion_at('left_thumb',1.2)
    assert len(calls)==2 and len(engine._motion_cache)==1
    engine._active_motions['left_thumb']=_motion(position=(1,2,3),velocity_w=np.array([.01,0,0]))
    c=engine._motion_at('left_thumb',1.2)
    assert c.position_w[0]>1 and len(calls)==3


def test_stationary_validated_prediction_is_exact_without_reconstruction(monkeypatch):
    import co_sim.moving_contact as module
    def forbidden(*args):raise AssertionError('stationary immutable motion need not be reconstructed')
    monkeypatch.setattr(module,'predict_motion',forbidden)
    engine=MovingContactRodEngine.__new__(MovingContactRodEngine)
    initial=_motion();engine._active_motions={'left_thumb':initial}
    engine._call_start=0.;engine._motion_cache={}
    assert engine._motion_at('left_thumb',.001) is initial


def _config(**changes):
    values = dict(
        py_dt=1.0e-4,
        isaac_dt=1.0e-3,
        n_elem=4,
        base_length=0.20,
        base_radius=0.008,
        density=1000.0,
        youngs_modulus=1.0e5,
        damping_constant=0.0,
        rod_start=np.array([-0.10, 0.0, 0.20]),
        rod_direction=np.array([1.0, 0.0, 0.0]),
        rod_normal=np.array([0.0, 0.0, 1.0]),
    )
    values.update(changes)
    return CoSimConfig(**values)


def _motion(position=(0.0, 0.0, 2.0), **changes):
    values = dict(
        position_w=np.asarray(position, dtype=float),
        rotation_wb=np.eye(3),
        velocity_w=np.zeros(3),
        omega_w=np.zeros(3),
        acceleration_w=np.zeros(3),
        alpha_w=np.zeros(3),
    )
    values.update(changes)
    return MeshMotion(**values)


@pytest.fixture
def assets(tmp_path):
    mesh = o3d.geometry.TriangleMesh.create_box(0.02, 0.02, 0.02)
    mesh.translate([-0.01, -0.01, -0.01])
    mesh.compute_triangle_normals()
    path = tmp_path / "box.ply"
    assert o3d.io.write_triangle_mesh(str(path), mesh, write_ascii=True)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {name: MeshAsset(path, digest) for name in CHANNELS}


def _engine(assets, **kwargs):
    return MovingContactRodEngine(
        _config(),
        table_center=np.array([0.0, 0.0, 0.05]),
        table_size=np.array([0.6, 0.4, 0.1]),
        meshes=assets,
        **kwargs,
    )


def _engine_dt(assets, py_dt):
    return MovingContactRodEngine(
        _config(py_dt=py_dt),
        table_center=np.array([0.0, 0.0, 0.05]),
        table_size=np.array([0.6, 0.4, 0.1]),
        meshes=assets,
    )


def test_cached_motion_matches_uncached_contact_trajectory_exactly(assets):
    from co_sim.moving_contact_models import predict_motion
    cached=_engine(assets);baseline=_engine(assets)
    def uncached(self,channel,time):
        if self._active_motions is None:return None
        elapsed=max(0.,time-self._call_start)
        return self._active_motions[channel] if elapsed==0 else predict_motion(self._active_motions[channel],elapsed)
    baseline._motion_at=MethodType(uncached,baseline)
    motions={name:_motion() for name in CHANNELS}
    motions['left_thumb']=_motion(position=(.025,0,.185),velocity_w=np.array([0,.02,0]),
                                 omega_w=np.array([0,0,.03]))
    saw_contact=False
    for _ in range(8):
        a,ia=cached.advance_with_meshes(motions,1e-4)
        b,ib=baseline.advance_with_meshes(motions,1e-4)
        for field in ('rod_position','rod_velocity','rod_director'):
            np.testing.assert_array_equal(a[field],b[field])
        for channel in CHANNELS:
            np.testing.assert_array_equal(ia[channel].linear_w,ib[channel].linear_w)
            np.testing.assert_array_equal(ia[channel].angular_com_w,ib[channel].angular_com_w)
            assert ia[channel].peak_force_n==ib[channel].peak_force_n
            assert ia[channel].contact_duration_s==ib[channel].contact_duration_s
        saw_contact|=ia['left_thumb'].peak_force_n>0
        assert cached._motion_cache=={}
    assert saw_contact


def test_no_contact_returns_four_zero_channels_and_matches_table_engine(assets):
    moving = _engine(assets)
    table = TableRodEngine(
        _config(),
        table_center=np.array([0.0, 0.0, 0.05]),
        table_size=np.array([0.6, 0.4, 0.1]),
    )

    snapshot, impulses = moving.advance_with_meshes(
        {name: _motion() for name in CHANNELS}, 3.5e-4
    )
    expected = table.advance(3.5e-4)

    assert snapshot["mode"] == "FREE_TABLE_MOVING_CONTACT"
    assert tuple(impulses) == CHANNELS
    for impulse in impulses.values():
        np.testing.assert_array_equal(impulse.linear_w, np.zeros(3))
        np.testing.assert_array_equal(impulse.angular_com_w, np.zeros(3))
        assert impulse.peak_force_n == 0.0
        assert impulse.contact_duration_s == 0.0
    for field in ("rod_position", "rod_velocity", "rod_director", "rod_omega"):
        np.testing.assert_allclose(snapshot[field], expected[field], atol=1e-15, rtol=0.0)
    assert moving.last_contact_diagnostics["evaluation_count"] == 4 * 4
    assert moving.last_contact_diagnostics["integrated_duration_s"] == pytest.approx(3.5e-4)


def test_second_call_has_no_residual_impulse(assets):
    engine = _engine(assets)
    motions = {name: _motion() for name in CHANNELS}
    _, first = engine.advance_with_meshes(motions, 2.0e-4)
    _, second = engine.advance_with_meshes(motions, 2.0e-4)
    for name in CHANNELS:
        np.testing.assert_array_equal(first[name].linear_w, np.zeros(3))
        np.testing.assert_array_equal(second[name].linear_w, np.zeros(3))


def test_only_contacting_channel_reacts_and_balances_common_origin(assets):
    engine = _engine(assets)
    motions = {name: _motion() for name in CHANNELS}
    motions["left_thumb"] = _motion(position=(0.025, 0.0, 0.185))

    _, impulses = engine.advance_with_meshes(motions, 1.0e-4)

    assert impulses["left_thumb"].peak_force_n > 0.0
    assert impulses["left_thumb"].linear_w[2] < 0.0
    for name in CHANNELS[1:]:
        np.testing.assert_array_equal(impulses[name].linear_w, np.zeros(3))
    diagnostics = engine.last_contact_diagnostics["channels"]["left_thumb"]
    np.testing.assert_allclose(diagnostics["linear_balance_w"], 0.0, atol=1e-15)
    np.testing.assert_allclose(
        diagnostics["angular_origin_balance_w"], 0.0, atol=1e-15
    )


def test_separated_mesh_has_no_attractive_force(assets):
    engine = _engine(assets)
    motions = {name: _motion() for name in CHANNELS}
    motions["left_thumb"] = _motion(
        position=(0.025, 0.0, 0.17), velocity_w=np.array([0.0, 0.0, -1.0])
    )
    _, impulses = engine.advance_with_meshes(motions, 1.0e-4)
    np.testing.assert_array_equal(impulses["left_thumb"].linear_w, np.zeros(3))


def test_tangential_mesh_reaction_opposes_prescribed_slip(assets):
    engine = _engine(assets)
    motions = {name: _motion() for name in CHANNELS}
    motions["left_thumb"] = _motion(
        position=(0.025, 0.0, 0.185), velocity_w=np.array([0.0, 0.2, 0.0])
    )
    _, impulses = engine.advance_with_meshes(motions, 1.0e-4)
    assert impulses["left_thumb"].linear_w[1] < 0.0


def test_nonidentity_mesh_rotation_preserves_world_impulse_balance(assets):
    engine = _engine(assets)
    motions = {name: _motion() for name in CHANNELS}
    motions["left_thumb"] = _motion(
        position=(0.025, 0.0, 0.185),
        rotation_wb=np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        velocity_w=np.array([0.0, 0.2, 0.0]),
    )
    engine.advance_with_meshes(motions, 1.0e-4)
    diagnostics = engine.last_contact_diagnostics["channels"]["left_thumb"]
    np.testing.assert_allclose(diagnostics["linear_balance_w"], 0.0, atol=1e-15)
    np.testing.assert_allclose(diagnostics["angular_origin_balance_w"], 0.0, atol=1e-15)


def test_contact_impulse_converges_when_fine_step_is_halved(assets):
    motions = {name: _motion() for name in CHANNELS}
    motions["left_thumb"] = _motion(
        position=(0.025, 0.0, 0.185), velocity_w=np.array([0.0, 0.2, 0.0])
    )
    values = []
    for dt in (1.0e-4, 5.0e-5, 2.5e-5):
        _, impulses = _engine_dt(assets, dt).advance_with_meshes(motions, 4.0e-4)
        values.append(impulses["left_thumb"].linear_w)
    coarse_error = np.linalg.norm(values[0] - values[2])
    half_error = np.linalg.norm(values[1] - values[2])
    assert half_error < coarse_error


@pytest.mark.parametrize(
    "motions",
    [
        {},
        {name: _motion() for name in CHANNELS[:-1]},
        {**{name: _motion() for name in CHANNELS}, "extra": _motion()},
    ],
)
def test_invalid_channel_set_is_atomically_rejected(assets, motions):
    engine = _engine(assets)
    before = engine.snapshot()
    with pytest.raises(ValueError, match="channels"):
        engine.advance_with_meshes(motions, 1.0e-4)
    assert engine.snapshot()["sim_time"] == before["sim_time"]
    assert engine.last_contact_diagnostics is None


@pytest.mark.parametrize("duration", [True, 0.0, -1.0, np.nan, np.inf, "0.1"])
def test_invalid_duration_is_atomically_rejected(assets, duration):
    engine = _engine(assets)
    with pytest.raises((TypeError, ValueError)):
        engine.advance_with_meshes({name: _motion() for name in CHANNELS}, duration)
    assert engine.time == 0.0
    assert engine.last_contact_diagnostics is None


def test_direct_advance_cannot_reuse_stale_mesh_motion(assets):
    engine = _engine(assets)
    with pytest.raises(RuntimeError, match="advance_with_meshes"):
        engine.advance(1.0e-4)


def test_midstep_failure_latches_engine_and_preserves_last_diagnostics(assets):
    engine = _engine(assets)
    motions = {name: _motion() for name in CHANNELS}
    engine.advance_with_meshes(motions, 1.0e-4)
    previous = engine.last_contact_diagnostics

    class BrokenStepper:
        def step(self, sim, time, dt):
            raise FloatingPointError("injected")

    engine.stepper = BrokenStepper()
    with pytest.raises(FloatingPointError, match="injected"):
        engine.advance_with_meshes(motions, 1.0e-4)
    assert engine.last_contact_diagnostics is previous
    with pytest.raises(RuntimeError, match="unusable"):
        engine.advance_with_meshes(motions, 1.0e-4)


def test_asset_sha_mismatch_is_rejected_before_physical_initialization(assets, monkeypatch):
    broken = dict(assets)
    broken["left_thumb"] = MeshAsset(assets["left_thumb"].path, "0" * 64)
    reached = False

    def unexpected(*args, **kwargs):
        nonlocal reached
        reached = True

    monkeypatch.setattr(TableRodEngine, "__init__", unexpected)
    with pytest.raises(ValueError, match="SHA-256"):
        _engine(broken)
    assert not reached


def test_geometry_is_parsed_from_exact_hashed_bytes_when_source_changes(assets, monkeypatch):
    payload = assets["left_thumb"].path.read_bytes()
    isolated = {}
    for name in CHANNELS:
        path = assets[name].path.parent / f"{name}.ply"
        path.write_bytes(payload)
        isolated[name] = MeshAsset(path, hashlib.sha256(payload).hexdigest())
    target = isolated["left_thumb"].path
    original = Path.read_bytes

    def read_then_replace(path):
        payload = original(path)
        if Path(path) == target:
            target.write_text("not the hashed mesh", encoding="utf-8")
        return payload

    monkeypatch.setattr(Path, "read_bytes", read_then_replace)
    engine = _engine(isolated)
    assert engine.mesh_bodies["left_thumb"]._is_watertight


def test_correct_sha_invalid_geometry_is_rejected_before_physical_initialization(tmp_path):
    path = tmp_path / "open.ply"
    path.write_text(
        "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\n"
        "property float z\nelement face 1\nproperty list uchar int vertex_indices\n"
        "end_header\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n",
        encoding="ascii",
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    invalid = {name: MeshAsset(path, digest) for name in CHANNELS}
    with pytest.raises(ValueError, match="closed and orientable"):
        _engine(invalid)


def test_nonfinite_forged_motion_is_atomically_rejected(assets):
    forged = object.__new__(MeshMotion)
    for field, value in _motion().__dict__.items():
        object.__setattr__(forged, field, value)
    object.__setattr__(forged, "position_w", np.array([np.nan, 0.0, 0.0]))
    motions = {name: _motion() for name in CHANNELS}
    motions["left_thumb"] = forged
    engine = _engine(assets)
    with pytest.raises(ValueError, match="finite"):
        engine.advance_with_meshes(motions, 1.0e-4)
    assert engine.time == 0.0
    assert engine.last_contact_diagnostics is None


def test_postadvance_output_failure_latches_engine(assets, monkeypatch):
    import co_sim.moving_contact as module

    engine = _engine(assets)
    motions = {name: _motion() for name in CHANNELS}

    def fail_output(*args, **kwargs):
        raise ArithmeticError("injected output failure")

    monkeypatch.setattr(module, "ContactImpulse", fail_output)
    with pytest.raises(ArithmeticError, match="injected output failure"):
        engine.advance_with_meshes(motions, 1.0e-4)
    assert engine.time == pytest.approx(1.0e-4)
    assert engine.last_contact_diagnostics is None
    with pytest.raises(RuntimeError, match="unusable"):
        engine.advance_with_meshes(motions, 1.0e-4)


def test_stale_sampler_call_is_rejected_immediately(assets):
    engine = _engine(assets)
    motions = {name: _motion() for name in CHANNELS}
    engine.advance_with_meshes(motions, 1.0e-4)
    with pytest.raises(RuntimeError, match="Stale"):
        engine._sample_contact("left_thumb", float(engine.time))


def test_out_of_order_sampler_call_is_rejected_and_latches(assets):
    engine = _engine(assets)
    motions = {name: _motion() for name in CHANNELS}
    real = engine._sample_contact

    def wrong_first(self, channel, time):
        if channel == "left_thumb":
            return real("right_index", time)
        return real(channel, time)

    engine._sample_contact = MethodType(wrong_first, engine)
    with pytest.raises(RuntimeError, match="order"):
        engine.advance_with_meshes(motions, 1.0e-4)
    assert engine._unusable


def test_duplicate_sampler_call_is_rejected_immediately(assets):
    engine = _engine(assets)
    motions = {name: _motion() for name in CHANNELS}
    real = engine._sample_contact

    def duplicate_first(self, channel, time):
        result = real(channel, time)
        if channel == "left_thumb":
            real(channel, time)
        return result

    engine._sample_contact = MethodType(duplicate_first, engine)
    with pytest.raises(RuntimeError, match="order|Repeated"):
        engine.advance_with_meshes(motions, 1.0e-4)
    assert engine._unusable


def test_missing_final_sampler_call_is_detected_after_physical_step(assets):
    engine = _engine(assets)
    motions = {name: _motion() for name in CHANNELS}
    real = engine._sample_contact

    def omit_final(self, channel, time):
        if channel == "right_index":
            return None
        return real(channel, time)

    engine._sample_contact = MethodType(omit_final, engine)
    with pytest.raises(RuntimeError, match="final fine step"):
        engine.advance_with_meshes(motions, 1.0e-4)
    assert engine.time == pytest.approx(1.0e-4)
    assert engine._unusable


def test_constant_synthetic_contact_uses_full_and_partial_step_weights(assets):
    engine = _engine_dt(assets, 1.0e-4)
    motions = {name: _motion() for name in CHANNELS}
    real = engine._sample_contact
    force = np.array([2.0, -3.0, 4.0])

    def inject(self, channel, time):
        if channel == "left_thumb":
            self.mesh_bodies[channel].external_forces[:, 0] += force
            self.rod.external_forces[:, 0] -= force
        return real(channel, time)

    engine._sample_contact = MethodType(inject, engine)
    _, impulses = engine.advance_with_meshes(motions, 2.5e-4)
    np.testing.assert_allclose(impulses["left_thumb"].linear_w, force * 2.5e-4)
    assert engine.last_contact_diagnostics["channels"]["left_thumb"]["evaluations"] == 3


def test_late_channel_prediction_failure_does_not_partially_mutate_meshes(assets):
    engine = _engine(assets)
    motions = {name: _motion(position=(0.1, 0.2, 2.0)) for name in CHANNELS}
    motions["right_index"] = _motion(
        velocity_w=np.array([np.finfo(float).max, 0.0, 0.0]),
        acceleration_w=np.array([np.finfo(float).max, 0.0, 0.0]),
    )
    before = {
        name: (
            body.position_collection.copy(),
            body.director_collection.copy(),
            body.velocity_collection.copy(),
            body.omega_collection.copy(),
        )
        for name, body in engine.mesh_bodies.items()
    }

    with pytest.raises(ValueError, match="finite"):
        engine.advance_with_meshes(motions, 1.0e-4)

    for name, body in engine.mesh_bodies.items():
        for actual, expected in zip(
            (
                body.position_collection,
                body.director_collection,
                body.velocity_collection,
                body.omega_collection,
            ),
            before[name],
        ):
            np.testing.assert_array_equal(actual, expected)
    assert engine.time == 0.0
    assert engine._unusable
