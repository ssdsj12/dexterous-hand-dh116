import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parents[1]

def test_middle_references_and_meshes_are_pinned():
    refs = json.loads((ROOT / 'configs/symmetric_grasp_middle_v1/point_references.json').read_text())
    middle = [r for r in refs['references'] if r['digit'] == 'middle']
    assert {r['body_name'] for r in middle} == {'finger33_link'}
    for ref in middle:
        point = np.asarray(ref['point'], dtype=float)
        normal = np.asarray(ref['normal'], dtype=float)
        assert point.shape == normal.shape == (3,)
        assert np.isfinite(point).all() and np.isfinite(normal).all()
        assert np.isclose(np.linalg.norm(normal), 1., atol=1e-6)
        ray = np.asarray(ref['ray_direction'], dtype=float)
        assert np.dot(normal, ray) < 0.0  # pad normal faces the wire-facing side
    manifest = json.loads((ROOT / 'configs/symmetric_grasp_middle_v1/contact/manifest.json').read_text())
    assert set(manifest['meshes']) == {
        'left_thumb','left_index','left_middle','right_thumb','right_index','right_middle'}
    for side in ('left', 'right'):
        mesh = manifest['meshes'][f'{side}_middle']
        assert mesh['body_name'] == 'finger33_link'
        assert len(mesh['sha256']) == 64
        assert (ROOT / 'configs/symmetric_grasp_middle_v1/contact' / mesh['path']).exists()
