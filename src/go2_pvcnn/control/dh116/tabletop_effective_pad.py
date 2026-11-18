"""Conservative inner-distal DH116 CAD atlas, in actual rigid LINK frames.

No geometric reference is a measured contact. Source bounds/inner directions
were audited from both shipped USD collision meshes (including instance proxy
transforms), following SurfaceHandModel.from_usd. Each body's own hull supplies
the planes; no other finger's plane or thumb frame is substituted.
"""
import hashlib
import numpy as np
from scipy.spatial import ConvexHull

DISTAL_LINKS = ('finger14_link', 'finger23_link', 'finger33_link', 'finger43_link', 'finger53_link')
DIGITS = ('thumb', 'index', 'middle', 'ring', 'little')
_BOUNDS = {
    'finger23_link': [[-.007311,-.016610,.007571],[.007310,.001821,.045482]],
    'finger33_link': [[-.007311,-.017393,.014610],[.007310,.001036,.052521]],
    'finger43_link': [[-.007311,-.016610,.007571],[.007311,.001821,.045482]],
    'finger53_link': [[-.007030,-.013983,.008133],[.007030,.001969,.035801]],
}
_THUMB_BOUNDS = {
    'left': [[-.009993,-.013214,-.037453],[.009993,.001033,-.012507]],
    'right': [[-.009993,.010435,-.017053],[.009993,.035909,-.001901]],
}
# native_scene._PAD_LOCAL_POINTS is only an orientation/region seed for thumb.
# Four-finger seeds use the individual audited CAD bounds, not thumb/index FK.
_THUMB_SEEDS = {'left': [0.,-.010453,-.033942], 'right': [0.,.032193,-.015561]}
_THUMB_OUTWARD = {'left': [0.,-.84,-.54], 'right': [0.,.42,-.90]}


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _patch(vertices, side, body):
    vertices = np.unique(_array(vertices), axis=0)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError('finite source CAD vertices required')
    thumb = body == DISTAL_LINKS[0]
    expected = np.asarray(_THUMB_BOUNDS[side] if thumb else _BOUNDS[body])
    bounds = np.array([vertices.min(0), vertices.max(0)])
    if not np.allclose(bounds, expected, atol=5e-5, rtol=0):
        raise ValueError(f'unsupported DH116 CAD/link frame: {side}/{body}')
    hull = ConvexHull(vertices)
    triangles = vertices[hull.simplices]
    centroids = triangles.mean(1)
    seed = np.asarray(_THUMB_SEEDS[side] if thumb else [0., bounds[0,1]+.00015, bounds[1,2]-.006])
    hint = np.asarray(_THUMB_OUTWARD[side] if thumb else [0.,-1.,.15])
    hint /= np.linalg.norm(hint)
    eligible = hull.equations[:,:3] @ hint > .9
    if not eligible.any():
        raise ValueError(f'unverified inner surface: {side}/{body}')
    ids = np.flatnonzero(eligible)
    nearest = ids[np.argmin(np.linalg.norm(centroids[ids]-seed, axis=-1))]
    center = centroids[nearest]
    outward = hull.equations[nearest,:3]
    u = np.array([1.,0.,0.]) - outward[0]*outward
    u /= np.linalg.norm(u)
    v = np.cross(outward,u)
    # Actual supporting planes form the local convex surface envelope. The
    # elliptical window cuts away lateral edges, tip cap, proximal hinge.
    planes = np.unique(np.round(hull.equations[eligible], 10), axis=0)
    return dict(center_local_m=center.tolist(), reference_outward_normal_local=outward.tolist(),
                tangent_u=u.tolist(), tangent_v=v.tolist(), half_widths_m=[.004,.0035],
                planes_local=planes.tolist(), surface_tolerance_m=.001,
                normal_cosine_min=.8, geometry_only=True,
                source='TensorWorld.shapes: USD collisions mesh-to-rigid-link transform + ConvexHull',
                source_asset=('assets/fixed_dual_panda_dh116_v1/dependencies/' +
                              ('d134680ab619_DH116_left.usd' if side=='left' else '3046ad2ed855_DH116_right.usd')),
                hull_vertices_sha256=hashlib.sha256(vertices.astype('<f8').tobytes()).hexdigest(),
                audited_bounds_m=bounds.tolist(), seed_local_m=seed.tolist(),
                region='central inner distal ellipse; actual source supporting planes only')


def build_pad_atlas(world):
    """Build once on CPU from TensorWorld's actual local source collision hulls.

    Unknown body names, frames or missing/duplicate distal hulls fail at init.
    CUDA-to-CPU transfers occur only here, never per environment in sample().
    """
    sides = {side: {} for side in ('left','right')}
    for hi, body_id, vertices in world.shapes:
        if hi not in (0,1):
            raise ValueError('two DH116 hands required')
        name = world.hands[hi].body_names[body_id]
        if name not in DISTAL_LINKS:
            continue
        side = ('left','right')[hi]
        if name in sides[side]:
            raise ValueError('one audited distal hull per body required')
        sides[side][name] = _patch(vertices, side, name)
    if any(set(rows) != set(DISTAL_LINKS) for rows in sides.values()):
        raise ValueError('all five source distal hulls per hand required')
    return dict(version='dh116_effective_pad_v1', sides=sides,
                reference_points_kind='geometry_only; actual body_link_pose_w FK; never native contact',
                normal_convention='native normal is force ON HAND; opposite CAD outward',
                limitations=[
                    'CAD convex collision envelope, not hardware pad material/calibration or PhysX cooked mesh.',
                    'Only central inner distal ellipse (4mm x 3.5mm half axes); side edges, tip caps, hinge and other links excluded.',
                    'Unverified remaining distal surface is rejected, including other possible usable pad surfaces.',
                    '1mm source-surface tolerance and cosine>=0.8 are conservative geometric gates, not contact manufacture.',
                    'Contact buffer capacity/truncation must be configured/monitored by the native world.',
                ])
