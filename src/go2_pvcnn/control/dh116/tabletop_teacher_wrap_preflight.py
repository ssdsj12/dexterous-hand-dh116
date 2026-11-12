"""Screen archived DH116 pregrasp geometry for a distinct whole-hand wrap."""
import math

from .tabletop_effective_pad import DISTAL_LINKS
from .tabletop_surface_contract import validate_contract


DIGITS = ('thumb', 'index', 'middle', 'ring', 'little')
STYLE = 'power_wrap'


def _number(value, name, low=None, high=None):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or low is not None and value < low or high is not None and value > high):
        raise ValueError(f'invalid finite {name}')
    return value


def _vector(value, size, name, low=None, high=None):
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f'{name} requires {size} entries')
    return [_number(x, name, low, high) for x in value]


def _unit_vectors(value, name):
    if not isinstance(value, list) or len(value) != len(DIGITS):
        raise ValueError(f'{name} requires one vector per named distal')
    result = [_vector(x, 3, name) for x in value]
    if any(abs(math.hypot(*x) - 1.) > 1e-5 for x in result):
        raise ValueError(f'{name} requires finite unit vectors')
    return result


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def screen_wrap_hand(hand, *, distance_m=.0008, normal_dot=-.5, radial_dot=-.75):
    """Derive geometry from all five ordered distal measurements, not labels."""
    if (hand.get('distal_order') != list(DIGITS) or hand.get('collision_units') != 'mm'
            or hand.get('distance_units') != 'm'):
        raise ValueError('ordered distal geometry with explicit SI/mm units required')
    distances = _vector(hand['patch_distance_m'], 5, 'patch distances', 0.)
    alignment = _vector(hand['patch_normal_dot'], 5, 'patch alignment cosines', -1.000001, 1.000001)
    radial = _unit_vectors(hand['radial_directions'], 'radial directions')
    normals = _unit_vectors(hand['hand_normals'], 'hand normals')
    eligible = [distance <= distance_m and dot <= normal_dot for distance, dot in zip(distances, alignment, strict=True)]
    thumb_partners = [DIGITS[i] for i in range(1, 5)
                      if eligible[0] and eligible[i] and _dot(radial[0], radial[i]) <= radial_dot
                      and _dot(normals[0], normals[i]) <= -.5]
    collision_ok = (all(_number(hand[k], k, 0.) <= .3 for k in ('penetration', 'object_in_hand', 'self'))
                    and _number(hand['minimum_clearance_m'], 'table clearance') >= .00029)
    return dict(aligned_digits=[digit for digit, valid in zip(DIGITS, eligible, strict=True) if valid],
                missing_aligned_digits=[digit for digit, valid in zip(DIGITS, eligible, strict=True) if not valid],
                opposed_thumb_partners=thumb_partners, patch_distance_m=dict(zip(DIGITS, distances, strict=True)),
                geometry_screen_passed=all(eligible) and len(thumb_partners) >= 3 and collision_ok,
                native_contact_verified=False)


def audit_wrap_candidates(source, native):
    """Report source geometry coverage without emitting a native reset or trial."""
    validate_contract(native)
    support = native['end_support_scene']
    if (support['version'] != 'dh116-static-end-supports-v1' or support['wire_axis'] != 'Y'
            or support['wire_length_m'] != .5 or support['wire_radius_m'] != .008):
        raise ValueError('the existing supported solid cylinder geometry is required')
    contract = source['contract']
    if (source.get('geometry_only') is not True or source.get('qualified') is not False
            or contract.get('version') != 'multifinger_bounded_cpu_geometry_v1'
            or source.get('source_hashes') != native['source_sha256']
            or contract.get('distal_order') != list(DIGITS)):
        raise ValueError('source-pinned DH116 geometry-only multifinger report required')
    pose = _vector(source['object_pose'], 7, 'source object pose')
    target_z = support['lift_reference_bottom_m'] + support['wire_radius_m']
    same_object_geometry = (source['rod_length_m'] == support['wire_length_m']
        and source['rod_radius_m'] == support['wire_radius_m'] and source['table_height_m'] == support['table_top_m']
        and pose[3:] == [1., 0., 0., 0.] and abs(pose[2] - target_z) <= 1e-9
        and all(pose[0] == x['center_m'][0] for x in support['supports']) and pose[1] == 0.)
    thresholds = dict(distance_m=min(.0008, _number(contract['contact_distance_m'], 'distance', 0.)),
        normal_dot=min(-.5, _number(contract['contact_hand_object_normal_dot_max'], 'normal alignment', -1., 1.)),
        radial_dot=min(-.75, _number(contract['contact_opposition_radial_dot_max'], 'radial opposition', -1., 1.)))
    result = []
    seen = set()
    for candidate in source['candidates']:
        index = candidate['candidate_index']
        if type(index) is not int or index in seen or candidate.get('qualified') is not False:
            raise ValueError('unique unqualified candidate index required')
        seen.add(index)
        geometry = candidate['geometry']
        hands = geometry['per_hand']
        if not isinstance(hands, list) or len(hands) != 2:
            raise ValueError('measured source geometry of both hands is required')
        screened = {side: screen_wrap_hand(hand, **thresholds) for side, hand in zip(('left','right'), hands, strict=True)}
        pair_ok = all(hand['geometry_screen_passed'] for hand in screened.values())
        separation_ok = _number(geometry['inter_hand_penetration_mm'], 'inter-hand penetration', 0.) == 0.
        scene_ok = same_object_geometry and source.get('support_collision_checked') is True
        result.append(dict(candidate_index=index, hands=screened, bilateral_distal_screen_passed=pair_ok,
                           source_geometry_candidate=(pair_ok and scene_ok and separation_ok
                                                       and candidate.get('geometry_feasible') is True)))
    return dict(version='dh116-teacher-wrap-geometry-preflight-v1', proposed_style=STYLE,
                interpretation='whole_hand_wrap_thumb_and_four_other_distals_per_hand',
                required_distals=dict(zip(DIGITS, DISTAL_LINKS, strict=True)),
                reference_object_pose_z_m=target_z, archived_object_pose_m=pose,
                archived_scene_matches_reference=same_object_geometry, thresholds=thresholds, candidates=result,
                geometry_candidates=sum(row['source_geometry_candidate'] for row in result),
                runtime_grasp_style_registered=False, native_grasp_verified=False, teacher_training_ready=False)
