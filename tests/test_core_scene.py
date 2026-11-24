import hashlib
import importlib
import json
from pathlib import Path
import numpy as np
import pytest

def package(tmp_path):
    root = tmp_path / 'root'
    root.mkdir()
    hand = root / 'assets/left.usd'
    hand.parent.mkdir()
    hand.write_bytes(b'asset-original')
    bundle = root / 'build/scene'
    bundle.mkdir(parents=True)
    np.savez_compressed(bundle / 'arrays.npz', nodes=np.zeros((41, 3), dtype=np.float32))
    data = {'schema': 'dh116-materialized-scene-v1', 'grasp': 'pinch', 'assets': {'left': 'assets/left.usd'}, 'prepared': {'hands': {'left': {'usd_sha256': hashlib.sha256(hand.read_bytes()).hexdigest()}}}, 'qualification': {'native': False, 'teacher': False, 'grasp': False}, 'reset': {'nodes': {'__array__': 'nodes', 'dtype': '<f4', 'shape': [41, 3]}}, 'arrays_sha256': hashlib.sha256((bundle / 'arrays.npz').read_bytes()).hexdigest()}
    manifest = bundle / 'scene.json'
    manifest.write_text(json.dumps(data))
    return (root, manifest, hashlib.sha256(manifest.read_bytes()).hexdigest())

def test_local_materialized_arrays_keep_dtype_shape_and_asset_binding(tmp_path):
    module = importlib.import_module('dh116.env.scene')
    (root, path, digest) = package(tmp_path)
    bundle = module.load_scene_inputs(path, digest, root=root)
    assert bundle['reset']['nodes'].dtype == np.float32
    assert bundle['reset']['nodes'].shape == (41, 3)
    (root / 'assets/left.usd').write_bytes(b'changed')
    with pytest.raises(ValueError, match='SHA'):
        module.load_scene_inputs(path, digest, root=root)

@pytest.mark.parametrize('damage', ['manifest', 'arrays', 'qualification', 'escape'])
def test_materialized_input_cannot_override_identity_or_qualification(tmp_path, damage):
    module = importlib.import_module('dh116.env.scene')
    (root, path, digest) = package(tmp_path)
    data = json.loads(path.read_text())
    if damage == 'manifest':
        path.write_text('{}')
    if damage == 'arrays':
        (path.parent / 'arrays.npz').write_bytes(b'broken')
    if damage == 'qualification':
        data['qualification']['grasp'] = True
        path.write_text(json.dumps(data))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if damage == 'escape':
        data['assets']['left'] = '../outside.usd'
        path.write_text(json.dumps(data))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        module.load_scene_inputs(path, digest, root=root)

def test_current_pinch_bundle_keeps_authored_and_measured_reset_distinct(monkeypatch):
    """The native author uses the proposal; solver Reset keeps its measured packet."""
    from types import SimpleNamespace
    from dh116.env.scene import load_task_scene
    from dh116.env.scene import prepared_scene
    from dh116.env.lifecycle import NativeOwners
    import dh116.hand.articulation as tabletop_regex_articulation
    import dh116.hand.articulation as tabletop_regex_task_setup
    root = Path(__file__).resolve().parents[1]
    (archive, measured) = load_task_scene('pinch', root=root)
    inputs = archive.scene_inputs
    assert inputs['grasp'] == archive.style == 'pinch' and inputs['qualification'] == {'native': False, 'teacher': False, 'grasp': False} and (measured is inputs['reset'])
    pair = archive.pair
    assert pair['reset_seed']['sha256'] == 'c19b0c971d76e4e613b7ca3ed12d06012514fd3e7a08a3823daee927e634ce81' and measured['seed']['sha256'] == '9ced76db3e4444107271296993f985b97dc034c40e5140f483288de5546b3bb5' and (pair['reset_seed']['reset_root_pose_w'] != measured['seed']['reset_root_pose_w'])
    proposed = np.asarray(pair['rod_seed']['proposed_nodes_m'], dtype=np.float64)
    observed = np.asarray(measured['rod_reset']['node_positions'])
    assert proposed.shape == (41, 3) and observed.shape == (1, 41, 3) and np.array_equal(proposed.astype(np.float32), observed[0]) and (not np.array_equal(proposed, observed[0]))

    class ReachedNativeAuthoring(Exception):
        pass

    def stop_before_sdk(*args, **kwargs):
        raise ReachedNativeAuthoring
    monkeypatch.setattr(tabletop_regex_articulation, 'spawn_regex_articulations', stop_before_sdk)
    lanes = [('/review/env_0', (0.0, 0.0, 0.0)), ('/review/env_1', (2.2, 0.0, 0.0))]
    with pytest.raises(ReachedNativeAuthoring):
        tabletop_regex_task_setup.prepare_regex_task(SimpleNamespace(current_time_step_index=0), NativeOwners(), None, root, prepared_scene(inputs), archive, lanes)

def test_pinch_required_sdf_pads_preserve_two_targets_and_three_non_target_roles():
    """Five reaction meshes remain; only the original Pinch pair requires SDF coverage."""
    from dh116.env.scene import load_task_scene
    from dh116.physics.geometry import TensorGeometry as TensorGeometry
    import dh116.physics.runtime as tabletop_tensor_physics_runtime
    root = Path(__file__).resolve().parents[1]
    (archive, _) = load_task_scene('pinch', root=root)
    materialized = archive.scene_inputs['geometry']
    inputs = tabletop_tensor_physics_runtime.pinch_geometry_inputs(materialized)
    required = ('finger14_link', 'finger23_link')
    assert inputs['required_pads'] == required
    assert all((len(inputs['grids'][side]) == 5 for side in ('left', 'right')))
    wrist = ('wrist_anchor', 'wrist_slide_x', 'wrist_slide_y', 'wrist_slide_z', 'wrist_rotate_x', 'wrist_rotate_y')
    names = tuple((tuple(inputs['geometry'][side]['source']) + wrist for side in ('left', 'right')))
    owner = TensorGeometry.from_materialized_geometry(inputs['geometry'], inputs['grids'], inputs['sources'], names, required_pads=inputs['required_pads'], device='cpu')
    assert len(owner.columns) == 44 and len(owner.pad_columns) == 4
    for (side_id, side) in enumerate(('left', 'right')):
        for (kind, entries) in inputs['geometry'][side].items():
            for link in entries:
                assert owner.non_target[owner.columns[side_id, kind + ':' + link]].item() is (link not in required)
