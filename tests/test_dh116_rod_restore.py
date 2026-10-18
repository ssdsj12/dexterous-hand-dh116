from fixture_paths import ARCHIVE, ASSET, BACKEND, CONTACT, CONTACT_SHA, SURFACE_REFS
"""Real Cosserat backend tests; no Isaac instance or grasp qualification."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(BACKEND))
from co_sim.free_rod import FreeRodEngine
from co_sim.models import CoSimConfig


def engine(**kw):
    return FreeRodEngine(CoSimConfig(n_elem=4, base_length=.5, base_radius=.008,
                                    isaac_dt=.0001, **kw))


def test_deformed_dynamic_state_keeps_original_reference_and_response():
    source=engine()
    source.rod.position_collection[2] += np.array([0.,.001,.002,.001,0.])
    source.rod.velocity_collection[1] = .001
    source.rod.compute_internal_forces_and_torques(0.)
    saved=source.export_physical_state()
    restored=engine()
    restored.restore_physical_state(saved)
    np.testing.assert_array_equal(restored.rod.rest_lengths, source.rod.rest_lengths)
    np.testing.assert_allclose(restored.rod.sigma, source.rod.sigma)
    np.testing.assert_allclose(restored.rod.internal_forces, source.rod.internal_forces)
    assert np.linalg.norm(restored.rod.sigma)>0
    assert float(restored.time)==0.
    source.advance()
    restored.advance()
    np.testing.assert_allclose(restored.rod.position_collection,source.rod.position_collection,atol=1e-14)
    np.testing.assert_allclose(restored.rod.velocity_collection,source.rod.velocity_collection,atol=1e-12)


@pytest.mark.parametrize('bad', ['reference','model','director','nan','collapsed'])
def test_bad_restore_rejected_before_mutation(bad):
    e=engine();saved=e.export_physical_state();before=e.rod.position_collection.copy()
    if bad=='reference':saved['reference']['rest_lengths'][0]*=2
    if bad=='model':saved['model_sha256']='0'*64
    if bad=='director':saved['dynamic']['rod_director'][0][0][0]=2.
    if bad=='nan':saved['dynamic']['rod_velocity'][0][0]=float('nan')
    if bad=='collapsed':
        for axis in saved['dynamic']['rod_position']:axis[1]=axis[0]
    with pytest.raises(ValueError):e.restore_physical_state(saved)
    np.testing.assert_array_equal(before,e.rod.position_collection)


def test_restore_is_fresh_instance_only_and_owned():
    e=engine();saved=e.export_physical_state();e.restore_physical_state(saved)
    saved['dynamic']['rod_position'][0][0]=100
    assert e.rod.position_collection[0,0]!=100
    with pytest.raises(RuntimeError):e.restore_physical_state(saved)
    other=engine();other.advance()
    with pytest.raises(RuntimeError):other.restore_physical_state(e.export_physical_state())


def test_model_change_rejected():
    saved=engine().export_physical_state()
    with pytest.raises(ValueError):engine(youngs_modulus=2e6).restore_physical_state(saved)
