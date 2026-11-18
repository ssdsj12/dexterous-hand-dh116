import numpy as np
import pytest

from go2_pvcnn.control.dh116.coordinated_closure import coordinated_bimanual_increment
from go2_pvcnn.control.dh116.contact_runtime import BODY_NAMES, load_mesh_manifest

def test_six_order_and_middle_active_mask():
    assert tuple(BODY_NAMES) == ('left_thumb','left_index','left_middle','right_thumb','right_index','right_middle')
    rng = np.random.default_rng(4)
    J = rng.normal(size=(2, 3, 3, 10))
    n = np.tile(np.eye(3)[None, None, :, 0], (2, 3, 1))
    W = np.zeros((2, 3, 10)); W[:, :, :7] = rng.normal(size=(2, 3, 7))
    result, diag = coordinated_bimanual_increment(
        J, n, W, np.zeros((2, 3)), 1e-4,
        target_loads_n=np.ones((2, 3)), transport_velocity_w=np.zeros(3),
        normal_speeds_m_s=np.full((2, 3), .001),
        active_pad_mask=np.array([[True, True, False], [True, True, False]]),
        return_diagnostics=True)
    assert result.shape == (2, 10)
    assert np.asarray(diag['requested_pad_velocity_w_m_s']).shape == (2, 3, 3)

def test_contact_manifest_rejects_wrong_middle_body(tmp_path):
    src = {
        'schema': 'dh116-contact-hulls-v1', 'asset_sha256': 'a' * 64, 'layers': {},
        'meshes': {'left_middle': {'path': 'left_middle.ply', 'sha256': 'b' * 64,
                                   'body_name': 'finger23_link'}}}
    p = tmp_path / 'manifest.json'; p.write_text(__import__('json').dumps(src))
    with pytest.raises(ValueError, match='channels|body'):
        load_mesh_manifest(p, __import__('hashlib').sha256(p.read_bytes()).hexdigest())
