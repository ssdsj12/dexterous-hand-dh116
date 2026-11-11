"""Analytic sensor fixtures. These are software checks, not native identity evidence."""
import inspect
import math

import pytest
import torch

from go2_pvcnn.control.dh116 import tabletop_stereo_perception as perception


def extractor(**kwargs):
    # The baseline remains executable for the initial semantic RED run.
    return getattr(perception, 'TrackedWirePointExtractor', perception.WirePointExtractor)(**kwargs)


def scene(paths, size=64):
    rgb = torch.zeros(1, 3, size, size)
    for path in paths:
        for y, x in path:
            rgb[0, :, y, x] = torch.tensor([.7, .4, .1])
    depth = torch.full((1, size, size), .5)
    valid = torch.ones_like(depth, dtype=torch.bool)
    k = torch.tensor([[100., 0., size / 2], [0., 100., size / 2], [0., 0., 1.]])
    return rgb, depth, valid, k


def run(tracker, data, *, transform=None, **kwargs):
    accepted = inspect.signature(tracker.extract).parameters
    kwargs = {key: value for key, value in kwargs.items() if key in accepted}
    return tracker.extract(*data, torch.eye(4) if transform is None else transform, **kwargs)


def visible(result):
    return result.get('visibility', result['point_valid'])[0]


def staircase():
    path = [(8, 8)]
    for step in range(18):
        y, x = path[-1]
        path.extend([(y, x + 1), (y + 1, x + 1)])
    return path


def test_staircase_redundant_diagonals_do_not_create_false_branches():
    result = run(extractor(), scene([staircase()]))
    assert int(result['point_valid'].sum()) >= 30
    assert result.get('topology_valid', torch.tensor([False])).all()


def test_true_diagonals_are_kept_instead_of_losing_the_visible_path():
    result = run(extractor(), scene([[(j, j) for j in range(8, 40)]]))
    assert result['point_valid'].sum() == 32
    assert result.get('topology_valid', torch.tensor([False])).all()


def test_curved_visible_path_preserves_topological_order():
    path = [(12, x) for x in range(8, 38)] + [(y, 37) for y in range(13, 36)]
    result = run(extractor(), scene([path]))
    assert result['point_valid'].sum() >= 48
    pixel = result['point_pixels'][0, visible(result)]
    assert (pixel[1:] - pixel[:-1]).abs().amax(-1).max() <= 1


@pytest.mark.parametrize('kind', ['branch', 'cross', 'loop'])
def test_genuine_nonpath_topology_is_invalid(kind):
    horizontal = [(24, x) for x in range(8, 42)]
    vertical = [(y, 24) for y in range(8, 25 if kind == 'branch' else 42)]
    if kind == 'loop':
        paths = [[(12, x) for x in range(12, 38)], [(37, x) for x in range(12, 38)],
                 [(y, 12) for y in range(12, 38)], [(y, 37) for y in range(12, 38)]]
    else:
        paths = [horizontal, vertical]
    result = run(extractor(), scene(paths))
    assert not result['point_valid'].any()
    assert not result.get('topology_valid', torch.tensor([True])).any()


def test_one_pixel_genuine_branch_is_not_pruned_into_a_claimed_simple_path():
    data = scene([[(24, x) for x in range(8, 44)], [(23, 25)]])
    result = run(extractor(), data)
    assert not result['point_valid'].any()
    assert not result['topology_valid'].any()


def test_disconnected_visible_fragments_have_separate_component_ids():
    result = run(extractor(), scene([[(24, x) for x in range(5, 24)],
                                     [(24, x) for x in range(35, 56)]]))
    assert result['point_valid'].sum() == 40
    assert result['component_id'][0, visible(result)].unique().numel() == 2
    assert not result['full_cable_observed'].any()
    assert result['topology_valid'].all()


@pytest.mark.parametrize('kind', ['branch', 'loop', 'singleton'])
def test_ambiguous_component_cannot_discard_independent_observed_fragments(kind):
    fragments = [[(42, x) for x in range(6, 23)], [(53, x) for x in range(29, 53)]]
    if kind == 'branch':
        ambiguous = [(15, x) for x in range(8, 35)] + [(y, 21) for y in range(7, 15)]
    elif kind == 'loop':
        ambiguous = ([(y, x) for y in (10, 21) for x in range(10, 25)]
                     + [(y, x) for x in (10, 24) for y in range(11, 21)])
    else:
        ambiguous = [(16, 21)]
    result = run(extractor(), scene([*fragments, ambiguous]))
    assert result['point_valid'].sum() == sum(map(len, fragments))
    assert result['visible_component_count'].tolist() == [2]
    assert result['topology_valid'].tolist() == [True]
    assert result['all_visible_components_simple'].tolist() == [False]
    assert result['rejected_component_count'].tolist() == [1]
    rejected = result['rejected_components'][0][0]
    assert rejected['pixel_count'] == len(ambiguous)
    assert rejected['reason'] == 'ambiguous_or_branched_visible_skeleton'
    assert not result['full_cable_observed'].any()
    actual = {tuple(point) for point in result['point_pixels'][0, visible(result)].tolist()}
    assert actual == set(fragments[0]) | set(fragments[1])


def test_thinning_a_thick_fragment_preserves_a_separate_thin_staircase():
    data = scene([staircase()])
    data[0][0, :, 45:50, 8:53] = torch.tensor([.7, .4, .1])[:, None, None]
    result = run(extractor(), data)
    assert result['topology_valid'].all()
    assert result['component_id'][0, visible(result)].unique().numel() == 2
    assert int(result['point_valid'].sum()) > 70


def test_depth_hole_does_not_split_or_reorder_the_rgb_path():
    data = scene([[(24, x) for x in range(5, 56)]])
    data[2][0, 24, 25:35] = False
    data[1][0, 24, 25:35] = float('nan')
    result = run(extractor(), data)
    assert int(visible(result).sum()) == 51
    assert int(result['point_valid'].sum()) == 41
    assert result['component_id'][0, visible(result)].unique().numel() == 1
    assert result['depth_segment_id'][0, result['point_valid'][0]].unique().numel() == 2
    missing = visible(result) & ~result['point_valid'][0]
    assert torch.isnan(result['points_world'][0, missing]).all()
    assert torch.isnan(result['arc_length'][0, missing]).all()
    assert not result['metric_arc_complete'].any()
    assert result['visible_arc_length_m'].item() == pytest.approx(.195, abs=1e-6)


def test_missing_all_depth_keeps_only_observed_2d_path_and_zero_confidence():
    data = scene([[(24, x) for x in range(5, 56)]])
    data[1][:] = float('nan'); data[2][:] = False
    result = run(extractor(), data)
    assert int(visible(result).sum()) == 51
    assert not result['point_valid'].any()
    assert torch.isnan(result['points_world']).all()
    assert not result['confidence'].any()
    assert (result['track_id'] == -1).all()


def test_occlusion_never_adds_tokens_across_the_hidden_gap():
    tracker = extractor(identity_motion_limit_m=.03)
    whole = scene([[(24, x) for x in range(5, 56)]])
    run(tracker, whole, timestamp=0.)
    fragments = scene([[(24, x) for x in range(5, 23)], [(24, x) for x in range(37, 56)]])
    result = run(tracker, fragments, timestamp=.03)
    assert result['point_valid'].sum() == 37
    x = result['point_pixels'][0, visible(result), 1]
    assert not ((x >= 23) & (x < 37)).any()
    assert not result['temporal_identity_valid'].any()
    assert (result['track_id'][0, visible(result)] == -1).all()


def test_sampling_is_bounded_uniform_visible_arc_with_explicit_token_contract():
    result = run(extractor(), scene([[(20, x) for x in range(5, 190)]], size=200))
    assert result['point_valid'].sum() == 128
    assert result['points_world'] is result['xyz']
    assert result['point_valid'] is result['valid']
    mask = visible(result)
    arc = result['arc_length_px'][0, mask]
    assert arc[-1].item() == 184
    assert ((arc[1:] - arc[:-1]) <= 2).all()
    assert (result['tangent'][0, mask].norm(dim=-1) - 1).abs().max() < 1e-6
    for key in ('component_id', 'arc_length', 'confidence', 'visibility', 'track_id'):
        assert result[key].shape == (1, 128)


def test_complete_depth_resamples_metric_arc_under_perspective_without_interpolation():
    data = scene([[(20, x) for x in range(5, 190)]], size=200)
    x = torch.arange(200, dtype=data[1].dtype)
    data[1][0, 20] = .1 + 2 * (x / 200).pow(3)
    result = run(extractor(point_count=32), data)
    mask = result['point_valid'][0]
    arc = result['arc_length'][0, mask]
    steps = torch.diff(arc)
    # Pixel-uniform sampling concentrates tokens at the near end; uniform
    # metric targets select actual observed pixels throughout the visible arc.
    assert float(steps.std() / steps.mean()) < .35
    assert result['sampling_arc_space'] == [['metric_3d']]
    pixel = result['point_pixels'][0, mask]
    camera_z = result['points_world'][0, mask, 2]
    assert torch.equal(camera_z, data[1][0, pixel[:, 0], pixel[:, 1]])


def test_partial_depth_sampling_explicitly_reports_pixel_arc_fallback():
    data = scene([[(24, x) for x in range(5, 56)]])
    data[2][0, 24, 25:35] = False
    result = run(extractor(), data)
    assert result.get('sampling_arc_space') == [['image_2d_partial_depth']]


def rotation(angle):
    a = math.radians(angle)
    t = torch.eye(4)
    t[:2, :2] = torch.tensor([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return t


def test_temporal_orientation_survives_rotation_across_dominant_world_axis():
    tracker = extractor(identity_motion_limit_m=.08)
    data = scene([[(25, x) for x in range(16, 49)]])
    previous = None
    previous_id = None
    for index, angle in enumerate(range(70, 116, 5)):
        result = run(tracker, data, transform=rotation(angle), timestamp=index * .03)
        mask = result['point_valid'][0]
        direction = result['points_world'][0, mask][-1] - result['points_world'][0, mask][0]
        if previous is not None:
            assert (direction @ previous).item() > 0
            assert result['temporal_identity_valid'].all()
            assert result['track_id'][0, mask].unique().item() == previous_id
        previous = direction
        previous_id = result.get('track_id', torch.full((1, 128), -1))[0, mask].unique().item()


def test_two_pixel_component_has_finite_association_cost_and_correct_orientation():
    tracker = extractor(identity_motion_limit_m=.02)
    data = scene([[(32, 31), (32, 32)]])
    first = run(tracker, data, timestamp=0.)
    turn = rotation(180)
    turn[1, 3] = .005  # rotate about the observed component centre
    second = run(tracker, data, transform=turn, timestamp=.03)
    a = first['points_world'][0, first['point_valid'][0]]
    b = second['points_world'][0, second['point_valid'][0]]
    assert (torch.diff(a, dim=0)[0] @ torch.diff(b, dim=0)[0]).item() > 0
    assert second['temporal_identity_valid'].all()


def test_prediction_keeps_ids_when_two_components_exchange_nearest_position():
    tracker = extractor(identity_motion_limit_m=.06)
    def pair(a, b):
        return scene([[(a, x) for x in range(8, 40)], [(b, x) for x in range(8, 40)]])
    first = run(tracker, pair(10, 50), timestamp=0.)
    second = run(tracker, pair(20, 42), timestamp=.1)
    third = run(tracker, pair(31, 34), timestamp=.2)
    fourth = run(tracker, pair(42, 26), timestamp=.3)
    assert second['temporal_identity_valid'].all()
    assert third['temporal_identity_valid'].all()
    assert fourth['temporal_identity_valid'].all()
    initial = visible(first) & (first['point_pixels'][0, :, 0] == 10)
    final = visible(fourth) & (fourth['point_pixels'][0, :, 0] == 42)
    assert first['track_id'][0, initial].unique().item() == fourth['track_id'][0, final].unique().item()


def test_ambiguous_cross_frame_component_association_keeps_unknown_ids():
    tracker = extractor(identity_motion_limit_m=.2)
    run(tracker, scene([[(15, x) for x in range(12, 44)], [(45, x) for x in range(12, 44)]]), timestamp=0.)
    result = run(tracker, scene([[(29, x) for x in range(12, 44)], [(31, x) for x in range(12, 44)]]), timestamp=.03)
    assert not result['temporal_identity_valid'].any()
    assert (result['track_id'][0, visible(result)] == -1).all()


@pytest.mark.parametrize('event', ['reset', 'backwards', 'nonfinite', 'large_gap'])
def test_reset_and_timestamp_anomalies_clear_correspondence(event):
    tracker = extractor()
    data = scene([[(24, x) for x in range(8, 44)]])
    run(tracker, data, timestamp=1.)
    matched = run(tracker, data, timestamp=1.03)
    assert matched['temporal_identity_valid'].all()
    kwargs = dict(timestamp={'reset': 1.06, 'backwards': .9, 'nonfinite': float('nan'), 'large_gap': 5.}[event])
    if event == 'reset':
        kwargs['reset_mask'] = torch.tensor([True])
    result = run(tracker, data, **kwargs)
    assert not result['temporal_identity_valid'].any()


def test_point_confidence_is_observed_depth_confidence_and_padding_is_zero():
    data = scene([[(24, x) for x in range(8, 44)]])
    confidence = torch.full_like(data[1], .37)
    result = run(extractor(), data, point_confidence=confidence)
    mask = result['point_valid']
    assert torch.allclose(result['confidence'][mask], torch.full_like(result['confidence'][mask], .37))
    assert not result['confidence'][~mask].any()


def test_legacy_graph_and_extractor_remain_explicitly_available():
    assert perception.LegacyWirePointExtractor is perception.WirePointExtractor
    assert perception._legacy_visible_path is perception._visible_path
    legacy = run(perception.LegacyWirePointExtractor(), scene([staircase()]))
    assert not legacy['point_valid'].any()
