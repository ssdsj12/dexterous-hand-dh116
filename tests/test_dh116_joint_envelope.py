import numpy as np
import pytest
from go2_pvcnn.control.dh116 import tabletop_joint_envelope as envelope
from go2_pvcnn.control.dh116.linear_coupling import active_limits


def profile():
    lo, hi = (np.tile(x, 2) for x in active_limits())
    center = (lo + hi) / 2
    return dict(version='dh116-joint-envelope-v1', lower_rad=(center-.1).tolist(),
                center_rad=center.tolist(), upper_rad=(center+.2).tolist())


def test_asymmetric_mapping_and_batch_match():
    p = profile()
    actions = np.stack([-np.ones(12), np.zeros(12), np.ones(12)])
    expected = np.array([p[k] for k in ('lower_rad','center_rad','upper_rad')])
    np.testing.assert_allclose(envelope.targets(p, actions), expected)
    for a, q in zip(actions, expected):
        np.testing.assert_allclose(envelope.targets(p, a), q)
    np.testing.assert_allclose(envelope.targets(p, np.full(12,.5)),
                               np.asarray(p['center_rad'])+.1)


@pytest.mark.parametrize('field,value', [('lower_rad',[-1.]*12),
    ('center_rad',[float('nan')]*12), ('upper_rad',[3.]*12), ('upper_rad',[.1]*11)])
def test_illegal_or_missing_bounds_rejected(field,value):
    p=profile();p[field]=value
    with pytest.raises(ValueError): envelope.targets(p,np.zeros(12))


def test_force_feedback_clips_to_same_envelope():
    p=profile()
    np.testing.assert_allclose(envelope.clip_targets(p,np.full((4,12),100.)),
                               np.tile(p['upper_rad'],(4,1)))
    with pytest.raises(ValueError): envelope.targets(p,np.full(12,1.01))
    with pytest.raises(ValueError): envelope.targets(p,np.zeros(20))
