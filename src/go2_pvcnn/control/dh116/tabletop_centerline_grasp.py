"""CPU source-CAD proposals for a frozen scene; never measured contact.

The object clearance proxy is the union of radius-thick line segments (capsules).
It conservatively contains a straight finite cylinder, including its flat caps.
Bent paths are a geometry proxy and cannot be sent to the rigid-cylinder native
experiment. Full source convex hulls are rechecked after the sampled optimizer.
An absent separating witness means UNCERTIFIED, not a proven collision.
"""
import copy
import hashlib
import itertools
import time
import numpy as np
import torch
from scipy.spatial import ConvexHull

from .clearance import hull_wire_lower_bound, separation_lower_bound
from .linear_coupling import active_limits, expand
from .tabletop_effective_pad import DIGITS, DISTAL_LINKS, _patch
from .tabletop_gravity_feasibility import gravity_feasibility

SIDES = ('left', 'right')
STYLES = {'pinch': (0, 1), 'tripod': (0, 1, 2)}
FROZEN_IDENTITY_TOLERANCES = dict(position_m=1e-6, quaternion_norm=1e-6, length_m=1e-6)
PROVENANCE_SCOPE = ('Author-supplied frozen inputs and local source file hashes only; '
                    'no independent native physical/reset/candidate identity verification.')
DEFAULT_THRESHOLDS = dict(contact_distance_m=.001, target_arc_tolerance_m=.008,
    normal_cosine_min=.8, opposition_radial_dot_max=-.7, opposition_normal_dot_max=-.5,
    allowed_object_overlap_m=.0003, noncontact_object_clearance_m=0.,
    self_clearance_m=.0003, interhand_clearance_m=.0003, obstacle_clearance_m=.0003,
    table_clearance_m=.00029, cable_sample_spacing_m=.002,
    end_margin_m=.01, minimum_hand_arc_separation_m=.02)


def _np(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def json_safe(value):
    """Convert arrays/scalars without silently serializing NaN or infinity."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return json_safe(_np(value).tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError('nonfinite report value')
    return value


def _finite(value, shape, name):
    a = _np(value)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f'{name}: finite {shape} required')
    return a


def _polyline(nodes_w):
    nodes = _np(nodes_w)
    if nodes.ndim != 2 or nodes.shape[1] != 3 or len(nodes) < 2 or not np.isfinite(nodes).all():
        raise ValueError('finite nondegenerate Nx3 centerline required')
    delta = np.diff(nodes, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    if np.any(lengths < 1e-9) or not np.isfinite(lengths.sum()):
        raise ValueError('zero/invalid centerline segment')
    tangents = delta/lengths[:, None]
    cumulative = np.r_[0., np.cumsum(lengths)]
    bent = bool(np.any(tangents @ tangents[0] < 1.-1e-9))
    return nodes, lengths, tangents, cumulative, bent


def sample_centerline_frames(nodes_w, arc_positions_m):
    """Interpolate finite arc lengths; vertex tangent is the outgoing segment.

    Terminal tangent is the last segment. u,v,t form a right-handed local basis;
    the least-aligned world axis fixes u deterministically for each segment.
    Vertex/end flags expose the non-smooth proxy and cap limitations.
    """
    nodes, lengths, tangents, cumulative, bent = _polyline(nodes_w)
    arcs = _np(arc_positions_m)
    if arcs.ndim != 1 or not np.isfinite(arcs).all() or np.any(arcs < 0) or np.any(arcs > cumulative[-1]):
        raise ValueError('arc positions must lie on the finite centerline')
    rows = []
    for s in arcs:
        k = min(int(np.searchsorted(cumulative, s, side='right')-1), len(lengths)-1)
        t = tangents[k]
        axis = np.eye(3)[np.argmin(np.abs(t))]
        u = np.cross(t, axis); u /= np.linalg.norm(u)
        v = np.cross(t, u)
        rows.append(dict(arc_position_m=float(s), point_w_m=(nodes[k]+(s-cumulative[k])*t).tolist(),
            tangent_w=t.tolist(), transverse_u_w=u.tolist(), transverse_v_w=v.tolist(),
            segment_index=k, at_vertex=bool(np.any(np.isclose(s, cumulative, atol=1e-10, rtol=0))),
            at_endpoint=bool(s == 0 or s == cumulative[-1]), bent_centerline_proxy=bent,
            centerline_length_m=float(cumulative[-1])))
    return rows


def _closest(points, nodes_w):
    nodes, lengths, tangent, cum, _ = _polyline(nodes_w)
    points = _np(points).reshape(-1, 3)
    delta = nodes[1:]-nodes[:-1]
    uv = np.clip(np.einsum('nsi,si->ns', points[:, None]-nodes[:-1], delta)/(lengths**2), 0, 1)
    nearest = nodes[:-1][None]+uv[..., None]*delta[None]
    distance = np.linalg.norm(points[:, None]-nearest, axis=-1)
    index = distance.argmin(1)
    selected = nearest[np.arange(len(points)), index]
    radial = points-selected
    norm = np.linalg.norm(radial, axis=-1)
    return selected, radial/np.maximum(norm[:, None], 1e-12), norm, cum[index]+uv[np.arange(len(points)), index]*lengths[index]


def _frozen_object_identity(scene, wire):
    """Bind a declared straight Y-cylinder to its frozen seed, never to native.

    Endpoints are unordered because a cylinder has an undirected axis. The
    uniform cylinder COM is the geometric midpoint; displaced COM models need
    a separate explicitly implemented object model. Bent paths must explicitly
    declare an independent proxy, whose wire_pose is provenance only.
    """
    nodes, _, _, cumulative, bent = _polyline(scene['centerline_w_m'])
    model = scene['native_object_model']
    if model not in ('rigid_cylinder', 'bent_centerline_proxy'):
        raise ValueError('object model must be rigid_cylinder or independent bent_centerline_proxy')
    if bent and model != 'bent_centerline_proxy':
        raise ValueError('bent centerline requires independent bent_centerline_proxy; it is not a native rigid cylinder')
    if model == 'bent_centerline_proxy':
        return dict(model=model, frozen_pose_binding='not_applicable_independent_geometry_proxy',
            independent_proxy=True, native_geometry_verified=False,
            tolerances=dict(FROZEN_IDENTITY_TOLERANCES))
    length = float(cumulative[-1])
    center = (nodes[0]+nodes[-1])/2
    tolerance = FROZEN_IDENTITY_TOLERANCES['position_m']
    if np.linalg.norm(center-wire[:3]) > tolerance:
        raise ValueError('wire_pose center does not match frozen cylinder centerline midpoint')
    if np.linalg.norm(_np(scene['center_of_mass_w_m'])-center) > tolerance:
        raise ValueError('center_of_mass_w_m does not match uniform frozen cylinder midpoint/wire_pose')
    w, x, y, z = wire[3:]
    axis = np.array([2*(x*y-w*z), 1-2*(x*x+z*z), 2*(y*z+w*x)])
    expected = wire[:3]+np.array([-.5, .5])[:, None]*length*axis
    endpoints = nodes[[0, -1]]
    endpoint_error = min(np.linalg.norm(endpoints-expected, axis=1).max(),
                         np.linalg.norm(endpoints-expected[::-1], axis=1).max())
    if endpoint_error > tolerance:
        raise ValueError('wire_pose local Y axis does not match frozen centerline endpoints')
    declarations = {}
    if 'native_length_m' in scene:
        declarations['native_length_m'] = scene['native_length_m']
    source_parameters = scene['source_provenance'].get('object_parameters', {})
    if not isinstance(source_parameters, dict):
        raise ValueError('source_provenance.object_parameters must be a mapping')
    if 'length_m' in source_parameters:
        declarations['source_provenance.object_parameters.length_m'] = source_parameters['length_m']
    if source_parameters.get('axis', 'Y') != 'Y':
        raise ValueError('wire_pose cylinder model requires local Y source axis')
    for name, declared in declarations.items():
        if (not np.isscalar(declared) or not np.isfinite(declared) or declared <= 0 or
                abs(declared-length) > FROZEN_IDENTITY_TOLERANCES['length_m']):
            raise ValueError(f'{name}: native length does not match frozen centerline length')
    return dict(model=model, frozen_pose_binding='straight centerline midpoint, endpoints/Y-axis, COM and declared length',
        independent_proxy=False, native_geometry_verified=False, derived_length_m=length,
        source_length_declarations=declarations, tolerances=dict(FROZEN_IDENTITY_TOLERANCES))


def validate_frozen_inputs(scene, seed):
    """Require explicit physics and obstacle completeness; never infer new physics."""
    required = ('schema_version', 'centerline_w_m', 'radius_m', 'mass_kg', 'friction',
        'drive_caps_nm', 'gravity_w_m_s2', 'gravity_torques_nm', 'center_of_mass_w_m',
        'table', 'obstacles', 'obstacles_complete', 'source_usd', 'source_provenance', 'native_object_model')
    for key in required:
        if key not in scene:
            raise ValueError(f'frozen scene missing {key}')
    if scene['schema_version'] != 'dh116_centerline_scene_v1':
        raise ValueError('unsupported frozen scene schema_version')
    _polyline(scene['centerline_w_m'])
    for key in ('radius_m', 'mass_kg', 'friction'):
        v = scene[key]
        if not np.isscalar(v) or not np.isfinite(v) or v <= 0:
            raise ValueError(f'{key}: positive finite explicit value required')
    if np.any(_finite(scene['drive_caps_nm'], (12,), 'drive_caps_nm') <= 0):
        raise ValueError('drive_caps_nm must be positive')
    _finite(scene['gravity_w_m_s2'], (3,), 'gravity_w_m_s2')
    _finite(scene['center_of_mass_w_m'], (3,), 'center_of_mass_w_m')
    if scene['gravity_torques_nm'] is not None:
        _finite(scene['gravity_torques_nm'], (12,), 'gravity_torques_nm')
    if not isinstance(scene['table'], dict) or not np.isfinite(scene['table'].get('top_z_m', np.nan)):
        raise ValueError('table.top_z_m must be explicit')
    if scene['obstacles_complete'] is not True:
        raise ValueError('obstacles_complete must explicitly include original supports/objects')
    if not isinstance(scene['obstacles'], list):
        raise ValueError('obstacles list required')
    ids = []
    for obstacle in scene['obstacles']:
        if not obstacle.get('id') or obstacle['id'] in ids:
            raise ValueError('unique obstacle id required')
        vertices = _np(obstacle.get('vertices_w_m'))
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4 or not np.isfinite(vertices).all():
            raise ValueError('obstacle vertices_w_m must be finite full convex vertices')
        ConvexHull(vertices)
        ids.append(obstacle['id'])
    if (set(scene['source_usd']) != set(SIDES) or not isinstance(scene['source_provenance'], dict)
            or not scene['source_provenance']):
        raise ValueError('both source_usd and nonempty source_provenance required')
    for key in ('roots', 'grasp_command', 'wire_pose', 'provenance'):
        if key not in seed:
            raise ValueError(f'frozen seed missing {key}')
    roots = _finite(seed['roots'], (2, 7), 'roots')
    wire = _finite(seed['wire_pose'], (7,), 'wire_pose')
    norms = np.r_[np.linalg.norm(roots[:, 3:], axis=1), np.linalg.norm(wire[3:])]
    if np.any(np.abs(norms-1.) > FROZEN_IDENTITY_TOLERANCES['quaternion_norm']):
        raise ValueError('unit quaternion required for each frozen root and wire_pose (wxyz)')
    _frozen_object_identity(scene, wire)
    q = _finite(seed['grasp_command'], (12,), 'grasp_command')
    expand(q[:6]); expand(q[6:])
    if not seed['provenance']:
        raise ValueError('explicit seed provenance required')
    return True


def build_source_pad_atlas(models):
    """Build all five *individual* audited CAD patches, never two-pad surrogates."""
    if len(models) != 2 or [m.side for m in models] != list(SIDES):
        raise ValueError('ordered left and right source models required')
    sides = {}
    for side, model in zip(SIDES, models):
        rows = {}
        for shape in model.spec['shapes']:
            body = shape['body'].split('/')[-1]
            if body not in DISTAL_LINKS:
                continue
            if body in rows:
                raise ValueError('one source distal hull per body required')
            rows[body] = _patch(shape['vertices'], side, body)
        if set(rows) != set(DISTAL_LINKS):
            raise ValueError('all five independent source distal hulls required')
        sides[side] = rows
    return dict(version='dh116_centerline_source_pad_v1', sides=sides,
                geometry_only=True, measured_contact=False,
                normal_convention='CAD outward ON ROD; native force ON HAND is opposite')


def _thresholds(config):
    result = dict(DEFAULT_THRESHOLDS)
    overrides = (config or {}).get('thresholds', {})
    if set(overrides)-set(result):
        raise ValueError('unknown threshold key')
    result.update(overrides)
    if any(not np.isscalar(v) or not np.isfinite(v) for v in result.values()):
        raise ValueError('finite screening thresholds required')
    for key, value in result.items():
        if key.endswith('_m') and value < 0:
            raise ValueError('distance thresholds cannot be negative')
    if result['cable_sample_spacing_m'] <= 0 or not 0 < result['normal_cosine_min'] <= 1:
        raise ValueError('invalid normal/spacing thresholds')
    for key in ('opposition_radial_dot_max', 'opposition_normal_dot_max'):
        if not -1 <= result[key] < 0:
            raise ValueError('opposition thresholds must be negative cosines')
    return result


def _hulls(model, poses):
    result = []
    for source, shape in zip(model.spec['shapes'], model.shapes):
        pose = _np(poses[shape['body']])[0]
        rotation, translation = pose[:3, :3], pose[:3, 3]
        vertices = _np(source['vertices'])@rotation.T+translation
        eq = _np(shape['planes'])
        normals = eq[:, :3]@rotation.T
        planes = np.c_[normals, eq[:, 3]-normals@translation]
        result.append(dict(body=shape['body'], vertices=vertices, planes=planes,
                           lower=vertices.min(0), upper=vertices.max(0)))
    return result


def _separation(a, b):
    # World-axis box gap is itself a valid separating witness and a cheap path.
    gaps = np.maximum(a['lower']-b['upper'], b['lower']-a['upper'])
    if gaps.max() > .002:
        return float(gaps.max())
    axes = np.r_[np.eye(3), a['planes'][:, :3], b['planes'][:, :3]]
    return separation_lower_bound(a['vertices'], b['vertices'], axes)


def _pad(model, poses, patch, body, scene, target, thresholds, digit):
    pose = _np(poses[body])[0]; rotation, translation = pose[:3, :3], pose[:3, 3]
    center = _finite(patch['center_local_m'], (3,), 'pad center')
    normal = _finite(patch['reference_outward_normal_local'], (3,), 'pad outward')
    if abs(np.linalg.norm(normal)-1.) > 1e-6:
        raise ValueError('unit CAD pad normal required')
    center_w = center@rotation.T+translation
    closest, radial, distance, arc = _closest([center_w], scene['centerline_w_m'])
    radial, distance, arc = radial[0], float(distance[0]), float(arc[0])
    object_surface = closest[0]+radial*scene['radius_m']
    local_object = (object_surface-translation)@rotation
    delta = local_object-center
    surface_offset = float(delta@normal)
    local_contact = local_object-surface_offset*normal
    offset = local_contact-center
    half = _finite(patch['half_widths_m'], (2,), 'pad ellipse half widths')
    if np.any(half <= 0):
        raise ValueError('positive pad half widths required')
    ellipse = float((offset@_np(patch['tangent_u'])/half[0])**2+(offset@_np(patch['tangent_v'])/half[1])**2)
    normal_w = normal@rotation.T
    cosine = float(-normal_w@radial)
    gap = distance-scene['radius_m']
    reasons = []
    if abs(gap) > thresholds['contact_distance_m']:
        reasons.append('PAD_DISTANCE')
    if abs(arc-target['arc_position_m']) > thresholds['target_arc_tolerance_m']:
        reasons.append('TARGET_ARC_REACH')
    if cosine < thresholds['normal_cosine_min']:
        reasons.append('WRONG_NORMAL')
    if ellipse > 1.+1e-9 or abs(surface_offset) > patch['surface_tolerance_m']:
        reasons.append('OUTSIDE_CENTRAL_PAD')
    # Source hull, rather than a fitted sphere or another finger's plane.
    shape = next(s for s in model.shapes if s['body'] == body)
    eq = _np(shape['planes'])
    source = next(s for s in model.spec['shapes'] if s['body'] == body)
    source_sha = hashlib.sha256(np.unique(_np(source['vertices']), axis=0).astype('<f8').tobytes()).hexdigest()
    if patch.get('hull_vertices_sha256') != source_sha or patch.get('geometry_only') is not True:
        reasons.append('SOURCE_PAD_IDENTITY')
    at_center = center@eq[:, :3].T+eq[:, 3]
    supporting = np.abs(at_center) <= 1e-7
    if np.max(at_center) > 1e-7 or not np.any(supporting & (eq[:, :3]@normal > .999)):
        reasons.append('OUTWARD_SOURCE_FRAME_MISMATCH')
    if np.max(local_contact@eq[:, :3].T+eq[:, 3]) > 1e-7:
        reasons.append('OUTSIDE_SOURCE_SURFACE')
    # Geometric, not measured, upward support capacity of the local cone.
    z = float(np.clip(normal_w[2], -1, 1))
    upward = z+scene['friction']*np.sqrt(max(0., 1-z*z))
    return dict(digit=digit, body=body, point_kind='geometric reference only',
        pad_center_w_m=center_w, source_pad_point_w_m=local_contact@rotation.T+translation,
        object_surface_point_w_m=object_surface, source_pad_point_local_m=local_contact,
        cad_outward_normal_w=normal_w, expected_native_force_on_hand_normal_w=-normal_w,
        radial_direction_w=radial, normal_cosine=cosine, distance_to_tube_m=gap,
        source_surface_offset_m=surface_offset, ellipse_squared_radius=ellipse,
        closest_arc_position_m=arc, target_arc_error_m=arc-target['arc_position_m'],
        upward_cone_capacity_per_normal_load=upward, rejection_reasons=reasons,
        normal_convention='CAD outward ON ROD; native force ON HAND is opposite')


def _static(models, q, roots, contacts, scene):
    if scene['gravity_torques_nm'] is None:
        return dict(static_feasible=False, status='gravity_compensation_unknown',
                    model='static_inscribed_friction_cone_lp', geometric_contact_proposals=True)
    if np.any(_np(scene['gravity_torques_nm']) != 0.):
        return dict(static_feasible=False, status='compensation_model_required',
            model='static_inscribed_friction_cone_lp', geometric_contact_proposals=True,
            message='A frozen nonzero compensation vector is not valid across optimized configurations; a configuration-dependent model is required.')
    points, normals, jacobians, hands, digits, faces = [], [], [], [], [], []
    for hi, model in enumerate(models):
        for digit_index, contact in enumerate(contacts[hi]):
            local = model.t(contact['source_pad_point_local_m'])
            root = model.t(roots[hi])[None]
            def point_function(active):
                pose = model.fk(active[None], root)[contact['body']][0]
                return pose[:3, :3]@local+pose[:3, 3]
            j = _np(torch.autograd.functional.jacobian(point_function, model.t(q[hi])))
            full = np.zeros((3, 12)); full[:, hi*6:(hi+1)*6] = j
            jacobians.append(full)
            points.append(contact['object_surface_point_w_m'])
            normals.append(contact['cad_outward_normal_w'])
            hands.append(hi); digits.append(digit_index); faces.append(0 if digit_index == 0 else 1)
    result = gravity_feasibility(points, normals, jacobians, scene['center_of_mass_w_m'],
        scene['mass_kg'], scene['friction'], scene['drive_caps_nm'],
        gravity=scene['gravity_w_m_s2'], gravity_torques=scene['gravity_torques_nm'],
        hand_ids=hands, digit_ids=digits, face_ids=faces)
    result.update(geometric_contact_proposals=True, normal_convention='force ON ROD',
        hand_model='six active Jacobian columns per hand; affine mimic FK',
        limitations='No measured forces, wrist/arm torque limits, cooked PhysX or dynamic stability certificate.')
    return result


def screen_candidate(models, atlas, centerline, candidate, obstacles=None, config=None):
    """Check all available hulls and required independent pads, retaining failures.

    `centerline` is the explicit frozen scene mapping, not merely a point array.
    geometry_feasible and static_feasible remain independent evidence fields.
    """
    scene = centerline
    thresholds = _thresholds(config)
    if obstacles is not None and json_safe(obstacles) != json_safe(scene['obstacles']):
        raise ValueError('obstacles must equal the frozen scene; support changes are forbidden')
    if len(models) != 2 or candidate.get('style') not in STYLES:
        raise ValueError('two source models and pinch/tripod style required')
    q = _finite(candidate['active_angles'], (12,), 'active_angles').reshape(2, 6)
    roots = _finite(candidate['wrist_pose_w'], (2, 7), 'wrist_pose_w')
    arcs = _finite(candidate['arc_positions_m'], (2,), 'arc_positions_m')
    frames = sample_centerline_frames(scene['centerline_w_m'], arcs)
    nodes, _, _, cumulative, bent = _polyline(scene['centerline_w_m'])
    reasons, hands, world_hulls, all_contacts = [], [], [], []
    required = STYLES[candidate['style']]
    if scene.get('obstacles_complete') is not True:
        reasons.append('ORIGINAL_OBSTACLES_NOT_VERIFIED')
    if abs(arcs[0]-arcs[1]) < thresholds['minimum_hand_arc_separation_m']:
        reasons.append('LEFT_RIGHT_ARC_SEPARATION')
    for side, arc in zip(SIDES, arcs):
        if min(arc, cumulative[-1]-arc) < thresholds['end_margin_m']:
            reasons.append(f'{side}:TARGET_ENDPOINT_MARGIN')
    clearance = dict(full_source_hulls_checked=True, full_cable_segments_checked=len(nodes)-1,
        per_hand=[], self_pairs=[], interhand_pairs=[], obstacle_pairs=[],
        semantics='lower bound certificate; a failed supplied-axis witness is UNCERTIFIED, not proven collision',
        self_exclusions='existing same fixed rigid cluster and direct neighboring clusters only')
    for hi, (side, model) in enumerate(zip(SIDES, models)):
        try:
            expand(q[hi])
        except ValueError:
            reasons.append(f'{side}:ACTIVE_COUPLED_LIMITS')
        poses = model.fk(model.t(q[hi])[None], model.t(roots[hi])[None])
        hulls = _hulls(model, poses); world_hulls.append(hulls)
        pads = []
        hand = dict(side=side, target=frames[hi], required_digits=[DIGITS[d] for d in required],
                    pads=pads, opposition_pairs=[], source_hull_count=len(hulls))
        side_atlas = atlas.get('sides', {}).get(side, {})
        body_map = {s['body'].split('/')[-1]: s['body'] for s in model.shapes}
        for digit in required:
            name = DISTAL_LINKS[digit]
            patch = side_atlas.get(name)
            if patch is None or name not in body_map:
                reasons.append(f'{side}/{DIGITS[digit]}:MISSING_SOURCE_PAD')
                continue
            pad = _pad(model, poses, patch, body_map[name], scene, frames[hi], thresholds, DIGITS[digit])
            pads.append(pad)
            reasons.extend(f'{side}/{DIGITS[digit]}:{r}' for r in pad['rejection_reasons'])
        for a, b in itertools.combinations(pads, 2):
            radial_dot = float(a['radial_direction_w']@b['radial_direction_w'])
            normal_dot = float(a['cad_outward_normal_w']@b['cad_outward_normal_w'])
            if radial_dot <= thresholds['opposition_radial_dot_max'] and normal_dot <= thresholds['opposition_normal_dot_max']:
                hand['opposition_pairs'].append(dict(digits=[a['digit'], b['digit']], radial_dot=radial_dot, normal_dot=normal_dot))
        if not hand['opposition_pairs']:
            reasons.append(f'{side}:NON_OPPOSED')
        if not pads or max(p['upward_cone_capacity_per_normal_load'] for p in pads) <= 0:
            reasons.append(f'{side}:NO_UPWARD_BEARING')
        hull_rows = []
        contact_bodies = {body_map[DISTAL_LINKS[d]] for d in required if DISTAL_LINKS[d] in body_map}
        for hull in hulls:
            allowed = -thresholds['allowed_object_overlap_m'] if hull['body'] in contact_bodies else thresholds['noncontact_object_clearance_m']
            bound = hull_wire_lower_bound(hull['planes'], nodes, scene['radius_m'],
                thresholds['cable_sample_spacing_m'], refine_below=allowed+.001)
            table = float(hull['lower'][2]-scene['table']['top_z_m'])
            hull_rows.append(dict(body=hull['body'], cable_clearance_lower_bound_m=bound,
                cable_threshold_m=allowed, table_clearance_m=table))
            if bound < allowed:
                reasons.append(f'{side}/{hull["body"].split("/")[-1]}:FULL_CABLE_CLEARANCE_UNCERTIFIED')
            if table < thresholds['table_clearance_m']:
                reasons.append(f'{side}/{hull["body"].split("/")[-1]}:TABLE_CLEARANCE')
        clearance['per_hand'].append(dict(side=side, hulls=hull_rows))
        for i, j in model.self_pairs:
            bound = _separation(hulls[i], hulls[j])
            clearance['self_pairs'].append(dict(side=side, bodies=[hulls[i]['body'], hulls[j]['body']], lower_bound_m=bound))
            if bound < thresholds['self_clearance_m']:
                reasons.append(f'{side}:SELF_CLEARANCE_UNCERTIFIED:{i}/{j}')
        hands.append(hand); all_contacts.append(pads)
    for i, a in enumerate(world_hulls[0]):
        for j, b in enumerate(world_hulls[1]):
            bound = _separation(a, b)
            clearance['interhand_pairs'].append(dict(shape_indices=[i, j], lower_bound_m=bound))
            if bound < thresholds['interhand_clearance_m']:
                reasons.append(f'INTERHAND_CLEARANCE_UNCERTIFIED:{i}/{j}')
    obstacle_rows = scene['obstacles'] if obstacles is None else obstacles
    for obstacle in obstacle_rows:
        vertices = _np(obstacle['vertices_w_m']); convex = ConvexHull(vertices)
        other = dict(vertices=vertices, planes=convex.equations, lower=vertices.min(0), upper=vertices.max(0))
        for side, hulls in zip(SIDES, world_hulls):
            for i, hull in enumerate(hulls):
                bound = _separation(hull, other)
                clearance['obstacle_pairs'].append(dict(side=side, shape_index=i, obstacle_id=obstacle['id'], lower_bound_m=bound))
                if bound < thresholds['obstacle_clearance_m']:
                    reasons.append(f'{side}/{hull["body"].split("/")[-1]}:OBSTACLE_CLEARANCE_UNCERTIFIED:{obstacle["id"]}')
    geometry_feasible = not reasons
    valid_contacts = all(len(p) == len(required) and all(not row['rejection_reasons'] for row in p) for p in all_contacts)
    if valid_contacts:
        static = _static(models, q, roots, all_contacts, scene)
    else:
        static = dict(static_feasible=False, status='required_geometry_contacts_missing_or_invalid',
                      model='static_inscribed_friction_cone_lp', geometric_contact_proposals=True)
    if not static['static_feasible']:
        if scene['gravity_torques_nm'] is None:
            reasons.append('STATIC_GRAVITY_COMPENSATION_UNKNOWN')
        elif np.any(_np(scene['gravity_torques_nm']) != 0.):
            reasons.append('STATIC_COMPENSATION_MODEL_REQUIRED')
        else:
            reasons.append('STATIC_UPWARD_SUPPORT_INFEASIBLE')
    accepted = geometry_feasible and static['static_feasible']
    score = sum(abs(p['distance_to_tube_m'])/.001+(1-p['normal_cosine']) for pads in all_contacts for p in pads)
    result = copy.deepcopy(candidate)
    result.update(geometry_feasible=geometry_feasible, static_feasible=bool(static['static_feasible']),
        accepted=bool(accepted), geometry_only=True, native_contact_verified=False,
        native_experiment_eligible=False, native_candidate_gate_implemented=False,
        static_proposal_ready_for_native_verification=bool(accepted),
        provenance_verification_scope=PROVENANCE_SCOPE,
        continuous_contact_verified=False, bent_centerline_proxy=bent,
        rejection_reasons=list(dict.fromkeys(reasons)), hand_diagnostics=hands,
        clearance=clearance, static_support=static, thresholds=thresholds, score=float(score),
        object_proxy='union of radius-thick finite segments; contains cylinder caps conservatively',
        limitations=['CAD proposals and ellipse reference points are not measured contact.',
            'Source convex hull separation is not PhysX cooking or force closure.',
            'Sampled optimization is followed by all available source hull checks.',
            'Bent polyline has no matching native deformable object in current rigid-cylinder scene.',
            'Static LP omits wrist/arm effort and dynamics.',
            'Native frozen-scene/reset/candidate closed-loop admission gate is not implemented; geometry results never admit native experiments.',
            PROVENANCE_SCOPE])
    return json_safe(result)


def _optimize(models, atlas, scene, seed, frames, style, azimuth, config, deadline):
    """Bounded proposal fit; only six original active angles and wrist pose vary."""
    q0 = _np(seed['grasp_command']).reshape(2, 6)
    roots = _np(seed['roots'])
    translation_bound = config.get('wrist_translation_bound_m', .02)
    rotation_bound = config.get('wrist_rotation_bound_rad', .3)
    x = torch.nn.Parameter(torch.tensor(np.c_[q0, np.zeros((2, 6))], dtype=torch.float64))
    optimizer = torch.optim.Adam([x], lr=.02)
    lower, upper = active_limits()
    actual_steps, best_cost, best = 0, None, x.detach().clone()

    def decode(hi, row):
        model = models[hi]
        angle = row[9:12]
        # Rotation.from_rotvec is used only for export; differentiable quaternion
        # exponential uses sinc with a finite zero-angle gradient.
        theta = angle.norm()
        delta_q = torch.cat((torch.cos(theta/2)[None], .5*torch.sinc(theta/(2*torch.pi))*angle))
        base = model.t(roots[hi, 3:]); base = base/base.norm()
        w, v = delta_q[0], delta_q[1:]; bw, bv = base[0], base[1:]
        quat = torch.cat(((w*bw-(v*bv).sum())[None], w*bv+bw*v+torch.cross(v, bv, dim=0)))
        root = torch.cat((model.t(roots[hi, :3])+row[6:9], quat))
        return row[:6], root

    def loss_value():
        loss = x.sum()*0
        for hi, model in enumerate(models):
            q, root = decode(hi, x[hi]); poses = model.fk(q[None], root[None])
            f = frames[hi]
            u, v = _np(f['transverse_u_w']), _np(f['transverse_v_w'])
            radial = np.cos(azimuth)*u+np.sin(azimuth)*v
            tangent = _np(f['tangent_w']); origin = _np(f['point_w_m'])
            for d in STYLES[style]:
                body_name = DISTAL_LINKS[d]
                patch = atlas['sides'][SIDES[hi]].get(body_name)
                if patch is None:
                    continue
                body = next(s['body'] for s in model.shapes if s['body'].split('/')[-1] == body_name)
                pose = poses[body][0]
                point = pose[:3, :3]@model.t(patch['center_local_m'])+pose[:3, 3]
                normal = pose[:3, :3]@model.t(patch['reference_outward_normal_local'])
                direction = radial if d == 0 else -radial
                offset = tangent*((d-1)*.004 if style == 'tripod' and d else 0)
                target = model.t(origin+scene['radius_m']*direction+offset)
                loss = loss+((point-target)/.008).square().sum()+2*((normal+model.t(direction))**2).sum()
            # Source vertices only: a differentiable table proxy; full certificates
            # for cable/self/interhand/obstacles remain mandatory below.
            for shape in model.shapes:
                p = poses[shape['body']][0]
                z = (shape['points']@p[:3, :3].T+p[:3, 3])[:, 2]
                loss = loss+10*((scene['table']['top_z_m']+.0003-z).clamp_min(0)/.005).square().mean()
        return loss

    for step in range(config.get('steps', 30)):
        if time.monotonic() >= deadline:
            break
        optimizer.zero_grad(set_to_none=True)
        loss = loss_value()
        value = float(loss.detach())
        if not np.isfinite(value):
            break
        if best_cost is None or value < best_cost:
            best_cost, best = value, x.detach().clone()
        loss.backward()
        if x.grad is None or not torch.isfinite(x.grad).all():
            break
        torch.nn.utils.clip_grad_norm_([x], 100.)
        optimizer.step()
        with torch.no_grad():
            x[:, :6].clamp_(torch.tensor(lower), torch.tensor(upper))
            x[:, 6:9].clamp_(-translation_bound, translation_bound)
            for hi in range(2):
                x[hi, 9:12] *= min(1., rotation_bound/max(float(x[hi, 9:12].norm()), 1e-12))
        actual_steps = step+1
    # Evaluate the final iterate as well as the pre-update states.
    if actual_steps:
        with torch.no_grad():
            value = float(loss_value())
            if np.isfinite(value) and (best_cost is None or value < best_cost):
                best_cost, best = value, x.detach().clone()
    qs, wrists = [], []
    with torch.no_grad():
        for hi in range(2):
            q, root = decode(hi, best[hi]); qs.extend(q.tolist()); wrists.append(root.tolist())
    return dict(style=style, active_angles=qs, wrist_pose_w=wrists,
        arc_positions_m=[f['arc_position_m'] for f in frames], azimuth_rad=float(azimuth),
        optimizer_steps=actual_steps, optimizer_cost=best_cost,
        wrist_translation_delta_m=best[:, 6:9].tolist(), wrist_rotation_vector_w_rad=best[:, 9:12].tolist())


def generate_candidates(models, atlas, centerline, seed, obstacles=None, config=None):
    """Enumerate distinct bilateral arc targets/styles; retain every failure.

    The wall budget stops admission of additional work. An already-started full
    certificate finishes; unstarted candidates are retained as budget rejections.
    This is a cooperative CPU bound, not a hard real-time/solver timeout.
    """
    scene, config = centerline, dict(config or {})
    validate_frozen_inputs(scene, seed)
    thresholds = _thresholds(config)
    _, _, _, cumulative, bent = _polyline(scene['centerline_w_m'])
    arcs = config.get('arc_positions_m', np.linspace(.2*cumulative[-1], .8*cumulative[-1], 3).tolist())
    frames = sample_centerline_frames(scene['centerline_w_m'], arcs)
    styles = config.get('styles', ['pinch', 'tripod'])
    azimuths = config.get('azimuths_rad', [0., np.pi/2, np.pi])
    if not styles or set(styles)-set(STYLES) or not azimuths or not np.isfinite(azimuths).all():
        raise ValueError('explicit valid styles and finite azimuths required')
    if len(arcs) < 2 or len(set(float(a) for a in arcs)) != len(arcs):
        raise ValueError('at least two distinct arc positions required')
    if type(config.get('steps', 30)) is not int or not 0 <= config.get('steps', 30) <= 200:
        raise ValueError('steps must be an integer in 0..200')
    for key, default in (('wall_seconds', 120.), ('wrist_translation_bound_m', .02), ('wrist_rotation_bound_rad', .3)):
        if not np.isfinite(config.get(key, default)) or config.get(key, default) <= 0:
            raise ValueError(f'positive finite {key} required')
    if obstacles is not None and json_safe(obstacles) != json_safe(scene['obstacles']):
        raise ValueError('obstacles must equal the frozen scene; support changes are forbidden')
    started = time.monotonic(); deadline = started+config.get('wall_seconds', 120.)
    accepted, rejected = [], []
    for left, right in itertools.permutations(frames, 2):
        for style, azimuth in itertools.product(styles, azimuths):
            index = len(accepted)+len(rejected)
            if time.monotonic() >= deadline:
                rejected.append(dict(candidate_index=index, style=style,
                    arc_positions_m=[left['arc_position_m'], right['arc_position_m']], azimuth_rad=float(azimuth),
                    geometry_feasible=False, static_feasible=False, accepted=False,
                    native_contact_verified=False, native_experiment_eligible=False,
                    static_proposal_ready_for_native_verification=False, native_candidate_gate_implemented=False,
                    screening_complete=False, rejection_reasons=['CPU_BUDGET_NOT_SCREENED']))
                continue
            proposal = _optimize(models, atlas, scene, seed, [left, right], style, azimuth, config, deadline)
            proposal['candidate_index'] = index
            row = screen_candidate(models, atlas, scene, proposal, scene['obstacles'], config)
            row['screening_complete'] = True
            (accepted if row['accepted'] else rejected).append(row)
    accepted.sort(key=lambda r: r['score'])
    return json_safe(dict(schema_version='dh116_centerline_candidates_v1', geometry_only=True,
        native_contact_verified=False, native_experiment_eligible=False, native_candidate_gate_implemented=False,
        static_proposal_ready_for_native_verification=bool(accepted),
        provenance_verification_scope=PROVENANCE_SCOPE,
        frozen_object_identity=_frozen_object_identity(scene, _np(seed['wire_pose'])),
        accepted=accepted, candidates=accepted, rejected=rejected,
        geometry_feasible_count=sum(r['geometry_feasible'] for r in accepted+rejected),
        static_feasible_count=sum(r['static_feasible'] for r in accepted+rejected),
        accepted_count=len(accepted), rejected_count=len(rejected),
        status='complete' if accepted else 'no_accepted_geometry_static_candidate',
        source_hulls_per_hand=[len(m.shapes) for m in models], thresholds=thresholds,
        search_config=config, frozen_scene=scene, frozen_seed=seed,
        seed_parameter_contract='six original active radians per hand, exact affine mimic; q unchanged physical bounds',
        bent_centerline_proxy=bent, elapsed_s=time.monotonic()-started,
        budget_semantics='cooperative admission/optimization bound; a started full check completes',
        limitations=['Only CPU geometry and static LP proposals; no native force or lift measurement.',
            'No external object force, attachment, support removal or scene reset is performed.',
            'All available source hulls are rechecked; optimization points are not continuous contact proof.',
            'Native frozen-scene/reset/candidate closed-loop admission gate is not implemented.',
            PROVENANCE_SCOPE]))
