import numpy as np
import pytest
from go2_pvcnn.control.dh116.clearance_certificate import ClearanceCertificate


def geometry():
    return {'a':np.array([[0.,0,0],[1,1,1]]),'b':np.array([[2.,0,0],[3,1,1]])}


def test_reuse_debits_current_vertex_and_wire_displacements_from_original_certificate():
    world=geometry();wire=np.array([[0.,2,0],[1,2,0]])
    cert=ClearanceCertificate(world,wire,lower_bound=.01,wire_radius=.008,wire_motion=0.)
    world['a']+=.001
    bound=cert.lower_bound(world,wire+.001,wire_radius=.009,wire_motion=.002)
    assert bound==pytest.approx(.01-3*np.sqrt(3)*.001-.001-.002)
    # Repeated reuse cannot reset accumulated displacement to the last frame.
    world['a']+=.001
    assert cert.lower_bound(world,wire+.001,wire_radius=.009,wire_motion=.002)<bound


def test_input_arrays_are_snapshotted_and_topology_change_rejected():
    world=geometry();wire=np.array([[0.,2,0],[1,2,0]])
    cert=ClearanceCertificate(world,wire,lower_bound=.01,wire_radius=.008,wire_motion=.001)
    assert cert.lower_bound(world,wire,wire_radius=.007,wire_motion=0.)==.01
    with pytest.raises(ValueError):cert.lower_bound({'a':world['a']},wire,wire_radius=.008,wire_motion=0.)
    bad=geometry();bad['a'][0,0]=np.nan
    with pytest.raises(ValueError):cert.lower_bound(bad,wire,wire_radius=.008,wire_motion=0.)


def test_rotation_and_wire_shape_changes_cannot_reuse_stale_clearance():
    world=geometry();wire=np.array([[0.,2,0],[1,2,0]])
    cert=ClearanceCertificate(world,wire,lower_bound=.01,wire_radius=.008,wire_motion=0.)
    world['a'][1]=[1.,1.,1.02]
    assert cert.lower_bound(world,wire,wire_radius=.008,wire_motion=0.)<0
    assert cert.lower_bound(geometry(),wire+[0,0,.02],wire_radius=.008,wire_motion=0.)<0
