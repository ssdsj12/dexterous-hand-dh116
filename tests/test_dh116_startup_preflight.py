"""Direct learning permits missed contact but never unsafe simulation."""
from types import SimpleNamespace
import numpy as np
import pytest


class Scene:
    def __init__(self, failure=None):
        self.num_envs = 2
        self.world = SimpleNamespace(force=np.zeros((2,4)))
        self.calls = 0
        self.failure = failure

    def step(self, actions):
        self.calls += 1
        return np.zeros((2,136)), np.zeros(2), np.zeros(2,dtype=bool), [
            dict(safety_failure=self.failure), {}]


def test_failed_contacts_can_produce_direct_training_data():
    from go2_pvcnn.control.dh116.tabletop_preflight import contact_preflight
    scene = Scene()
    report = contact_preflight(scene,scene.world,direct=True)
    assert report['mode'] == 'finite_bounded_exploration_contact_not_required'
    assert not report['all_hands_detect_contact']
    assert scene.calls == 2


def test_explicit_contact_diagnostic_still_rejects_missing_collision_response():
    from go2_pvcnn.control.dh116.tabletop_preflight import contact_preflight
    scene = Scene()
    with pytest.raises(ValueError,match='missing measured'):
        contact_preflight(scene,scene.world,direct=False)


def test_direct_training_still_blocks_physical_failure():
    from go2_pvcnn.control.dh116.tabletop_preflight import contact_preflight
    scene = Scene('joint_limit')
    with pytest.raises(ValueError,match='physical failure'):
        contact_preflight(scene,scene.world,direct=True)
    assert scene.calls == 1


def test_nonfinite_contact_force_blocks_direct_learning():
    from go2_pvcnn.control.dh116.tabletop_preflight import contact_preflight
    scene = Scene()
    scene.world.force[1,3] = np.nan
    with pytest.raises(ValueError,match='physical failure'):
        contact_preflight(scene,scene.world,direct=True)
