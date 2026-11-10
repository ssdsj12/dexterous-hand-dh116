"""CPU source-convex geometry for opt-in right-hand tripod recovery.

No Isaac import, stage mutation, contact manufacture or physics prediction.
Measured link frames are rebased at each snapshot. Only joint-local increments
use the affine mimic derivative; actual follower offsets are never projected.
All collision results are conservative source certificates, not cooked PhysX.
"""
from dataclasses import dataclass
import copy
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from .linear_coupling import ACTIVE, JOINTS, MIMIC, coupling_matrix
from .clearance import hull_wire_lower_bound, separation_lower_bound
from .tabletop_effective_pad import DIGITS, DISTAL_LINKS

_REGION_CACHE = {}


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _finite(value, shape, name):
    result = _array(value)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(name+' requires finite '+str(shape))
    return result


def _rigid(value):
    t = _finite(value, (4, 4), 'body frame')
    if (not np.allclose(t[3], [0, 0, 0, 1], atol=1e-9, rtol=0)
            or not np.allclose(t[:3, :3].T@t[:3, :3], np.eye(3), atol=1e-8, rtol=0)
            or abs(np.linalg.det(t[:3, :3])-1) > 1e-8):
        raise ValueError('body frame must be rigid')
    return t.copy()


def pose_matrix(pose):
    p = _finite(pose, (7,), 'wxyz pose')
    if abs(np.linalg.norm(p[3:])-1) > 1e-6:
        raise ValueError('unit wxyz quaternion required')
    result = np.eye(4); result[:3, 3] = p[:3]
    result[:3, :3] = Rotation.from_quat(p[[4, 5, 6, 3]]).as_matrix()
    return result


def world_hull(vertices, pose, body):
    vertices = _array(vertices); pose = _rigid(pose)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError('complete finite source hull vertices required')
    hull = ConvexHull(vertices)
    world = vertices@pose[:3, :3].T+pose[:3, 3]
    normals = hull.equations[:, :3]@pose[:3, :3].T
    return dict(body=body, vertices=world,
        planes=np.c_[normals, hull.equations[:, 3]-normals@pose[:3, 3]],
        lower=world.min(0), upper=world.max(0), source_vertex_count=len(vertices))


def _world_cached(shape, pose):
    r, p = pose[:3, :3], pose[:3, 3]
    vertices = shape['vertices']@r.T+p; normals = shape['planes'][:, :3]@r.T
    return dict(body=shape['body'], vertices=vertices,
        planes=np.c_[normals, shape['planes'][:, 3]-normals@p],
        lower=vertices.min(0), upper=vertices.max(0), source_vertex_count=len(vertices))


class SourceHandGeometry:
    """An independent source spec; legacy SurfaceHandModel stays untouched."""
    def __init__(self, spec, pad_atlas_side):
        self.spec = copy.deepcopy(spec); self.side = spec['side']; self.root = spec['root']
        self.source_provenance_verified=False
        if self.side not in ('left', 'right'):
            raise ValueError('left/right side required')
        pending = list(spec['edges']); known = {self.root}; self.edges = []; self.paths = {self.root: []}
        moving = [e['name'] for e in pending if e['axis'] is not None]
        if len(moving) != 11 or set(moving) != set(JOINTS):
            raise ValueError('all eleven named finger joints required')
        while pending:
            ready = [e for e in pending if e['parent'] in known]
            if not ready:
                raise ValueError('disconnected source joint tree')
            for e in ready:
                if e['child'] in known:
                    raise ValueError('duplicate source child body')
                axis = None if e['axis'] is None else _finite(e['axis'], (3,), 'joint axis')
                if axis is not None and abs(np.linalg.norm(axis)-1) > 1e-9:
                    raise ValueError('unit joint axis required')
                row = dict(e, t0=_rigid(e['t0']), t1=_rigid(e['t1']), axis=axis)
                self.paths[e['child']] = self.paths[e['parent']]+[len(self.edges)]
                self.edges.append(row); known.add(e['child']); pending.remove(e)
        self.bodies = known; self.shapes = []
        for s in spec['shapes']:
            if s['body'] not in known:
                raise ValueError('unknown collision body')
            v = _array(s['vertices']); hull = ConvexHull(v)
            self.shapes.append(dict(body=s['body'], vertices=v.copy(), planes=hull.equations.copy()))
        self.patches = {}; self.patch_planes = {}
        for key, patch in pad_atlas_side.items():
            digit = DIGITS[DISTAL_LINKS.index(key)] if key in DISTAL_LINKS else key
            body = patch.get('body')
            if body is None:
                body = next((s['body'] for s in self.shapes
                    if s['body'].split('/')[-1] == DISTAL_LINKS[DIGITS.index(digit)]), None)
            if body not in known or patch.get('geometry_only') is not True:
                raise ValueError('independent source pad body and geometry_only identity required')
            shape = next(s for s in self.shapes if s['body'] == body)
            digest = hashlib.sha256(np.unique(shape['vertices'], axis=0).astype('<f8').tobytes()).hexdigest()
            if patch.get('hull_vertices_sha256', digest) != digest:
                raise ValueError('source pad hash mismatch')
            _, _, eligible = _patch_parameterization(shape['planes'], patch)
            self.patch_planes[digit] = eligible.copy()
            self.patches[digit] = dict(copy.deepcopy(patch), body=body)
        # Identical exclusions to SurfaceHandModel: fixed rigid cluster/direct neighbor.
        cluster = {b: b for b in known}
        def root(b):
            while cluster[b] != b:
                b = cluster[b]
            return b
        for e in self.edges:
            if e['axis'] is None:
                cluster[root(e['child'])] = root(e['parent'])
        adjacent = set()
        for e in self.edges:
            a, b = root(e['parent']), root(e['child'])
            if a != b:
                adjacent.update(((a, b), (b, a)))
        self.self_pairs = [(i, j) for i, a in enumerate(self.shapes)
            for j, b in enumerate(self.shapes[:i])
            if root(a['body']) != root(b['body']) and (root(a['body']), root(b['body'])) not in adjacent]
        self.source_hash = hashlib.sha256(json.dumps(self.spec, sort_keys=True,
            separators=(',', ':')).encode()).hexdigest()

    @classmethod
    def from_usd(cls, source, side):
        """Read/pin every source collision mesh without loading Isaac/PhysX."""
        from pxr import Usd,UsdGeom
        from .hand_point_kinematics import HandPointFK
        from .usd_surface import read_surface,layer_pins
        from .tabletop_effective_pad import _patch
        source=Path(source).resolve(strict=True);stage=Usd.Stage.Open(str(source))
        if stage is None or UsdGeom.GetStageMetersPerUnit(stage)!=1.:
            raise ValueError('readable metre source USD required')
        pins=layer_pins(stage);fk=HandPointFK(source,side,[0.,0.,0.,1.,0.,0.,0.])
        bodies={fk.root}|{e[2] for e in fk.edges};records=[];shapes=[];atlas={}
        meshes=[p for p in Usd.PrimRange(stage.GetPseudoRoot(),Usd.TraverseInstanceProxies())
                if '/collisions/' in str(p.GetPath()) and p.IsA(UsdGeom.Mesh)]
        for body in sorted(bodies):
            under=[p for p in meshes if str(p.GetPath()).startswith(body+'/collisions/')]
            if not under:continue
            row=read_surface(stage,body,'collisions');points=_array(row['points_body'])
            vertices=np.unique(points,axis=0);vertices=vertices[ConvexHull(vertices).vertices]
            shapes.append(dict(body=body,vertices=vertices.tolist()))
            name=body.split('/')[-1]
            if name in DISTAL_LINKS:atlas[name]=_patch(vertices,side,name)
            records.append({k:row[k] for k in ('body_path','mesh_path','mesh_to_body','collision_owner',
                'collision_approximation','orientation','geometry_sha256')})
        if len(shapes)!=17 or len(meshes)!=17 or len(atlas)!=5:
            raise ValueError('all seventeen source collision meshes/five independent pads required')
        if pins!=layer_pins(stage) or any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=digest for p,digest in pins.items()):
            raise ValueError('source layers changed during extraction')
        manifest=dict(verified=True,scope='independent source USD only; never native scene identity',
            source_path=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            layer_sha256=pins,meters_per_unit=1.,collision_meshes=records,source_collision_count=17,
            mimic_mapping=[dict(follower=f,master=m,ratio=r,offset_rad=o) for f,m,r,o in MIMIC],
            mimic_mapping_source='linear_coupling.py affine simulation increment; runtime coefficients require separate native audit')
        spec=dict(side=side,root=fk.root,source=str(source),source_manifest=manifest,
            edges=[dict(name=n,parent=a,child=b,t0=t0.tolist(),t1=t1.tolist(),axis=None if axis is None else axis.tolist())
                   for n,a,b,t0,t1,axis in fk.edges],shapes=shapes)
        result=cls(spec,atlas);result.source_provenance_verified=True
        return result

    def bind(self, full_q, poses):
        full = _finite(full_q, (11,), 'actual named full q')
        if not self.bodies.issubset(poses):
            raise ValueError('missing SDK body pose')
        actual = {b: _rigid(poses[b]) for b in self.bodies}
        relative = [np.linalg.inv(actual[e['parent']])@actual[e['child']] for e in self.edges]
        return dict(full_q=full.copy(), poses=actual, relative=relative,
            source_hash=self.source_hash, side=self.side,source_provenance_verified=self.source_provenance_verified)

    def predict(self, snapshot, delta_active):
        delta = _finite(delta_active, (6,), 'active increment')
        result = self.predict_full_delta(snapshot, coupling_matrix()@delta)
        result['delta_active'] = delta.copy()
        return result

    def predict_full_delta(self, snapshot, delta_full, root_pose_next=None):
        """Independent replay using actual next full-q and optional SDK root.

        This validation path imposes no mimic relation on measured motion.
        root_pose_next is a rigid 4x4 matrix or a measured wxyz seven-vector.
        """
        full_delta = _finite(delta_full, (11,), 'full joint increment')
        if snapshot['source_hash'] != self.source_hash or snapshot['side'] != self.side:
            raise ValueError('snapshot/source identity mismatch')
        if root_pose_next is None:
            root = snapshot['poses'][self.root].copy()
        else:
            root = pose_matrix(root_pose_next) if _array(root_pose_next).shape == (7,) else _rigid(root_pose_next)
        poses = {self.root: root}
        for e, relative in zip(self.edges, snapshot['relative']):
            local = np.eye(4)
            if e['axis'] is not None:
                local[:3, :3] = Rotation.from_rotvec(e['axis']*full_delta[JOINTS.index(e['name'])]).as_matrix()
            poses[e['child']] = poses[e['parent']]@e['t0']@local@np.linalg.inv(e['t0'])@relative
        hulls = [_world_cached(s, poses[s['body']]) for s in self.shapes]
        facets = {}
        for digit, patch in self.patches.items():
            planes = self.patch_planes[digit]
            facets[digit] = [dict(normal_w=poses[patch['body']][:3, :3]@p[:3], plane_local=p.copy()) for p in planes]
        return dict(full_q=snapshot['full_q']+full_delta, poses=poses, hulls=hulls,
            pad_facets=facets, delta_full=full_delta.copy())

    def motion_bounds(self, snapshot, delta_from, delta_to):
        """Hausdorff bound for the whole linearly interpolated joint interval.

        Conservative chain lengths, not a Jacobian or an endpoint-only sample.
        Each bound is from the interval midpoint to every intermediate pose.
        """
        half = np.abs(coupling_matrix()@(_array(delta_to)-_array(delta_from)))*.5
        extents = [2*np.linalg.norm(e['t0'][:3, 3])+np.linalg.norm(b[:3, 3])
                   for e, b in zip(self.edges, snapshot['relative'])]
        result = []
        for s in self.shapes:
            path = self.paths[s['body']]; bound = 0.
            for k, edge_index in enumerate(path):
                e = self.edges[edge_index]
                if e['axis'] is None:
                    continue
                radius = (np.linalg.norm(e['t0'][:3, 3])+sum(extents[j] for j in path[k:])
                          +np.linalg.norm(s['vertices'], axis=-1).max())
                a = half[JOINTS.index(e['name'])]
                bound += 2*radius*np.sin(min(a, np.pi)*.5)
            result.append(float(bound))
        return result

    def normal_motion_bounds(self, delta_from, delta_to):
        half=np.abs(coupling_matrix()@(_array(delta_to)-_array(delta_from)))*.5
        result=[]
        for shape in self.shapes:
            angle=sum(half[JOINTS.index(self.edges[i]['name'])] for i in self.paths[shape['body']]
                      if self.edges[i]['axis'] is not None)
            result.append(float(2*np.sin(min(angle,np.pi)*.5)))
        return result


def _clip_polygon(polygon, g, h):
    out = []
    if not len(polygon):
        return np.empty((0, 2))
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        da, db = float(g@a+h), float(g@b+h)
        ina, inb = da <= 0., db <= 0.
        if ina:
            out.append(a)
        if ina != inb:
            out.append(a+(b-a)*(da/(da-db)))
    return np.array(out).reshape(-1, 2)


def quadratic_disk_polygon_extrema(hessian, linear, inequalities, offsets):
    """Global extrema of z'H z+2g'z on |z|<=1 and Gz+h<=0.

    Enumerates interior stationary point, every clipped polygon edge and every
    stationary point of the circle (quartic tan-half-angle equation). No random
    surface samples or local nonlinear optimizer grant region qualification.
    """
    H = _finite(hessian, (2, 2), 'quadratic Hessian'); g = _finite(linear, (2,), 'quadratic linear')
    G, h = _array(inequalities), _array(offsets)
    if G.ndim != 2 or G.shape[1] != 2 or h.shape != (len(G),) or not np.isfinite([*G.ravel(), *h]).all():
        raise ValueError('finite planar halfspaces required')
    norms = np.linalg.norm(G, axis=1); zero = norms < 1e-15
    if np.any(h[zero] > 1e-11):
        return dict(complete=True, empty=True)
    if np.any(h > norms+1e-11):
        return dict(complete=True, empty=True)
    G, h = G[~zero]/norms[~zero, None], h[~zero]/norms[~zero]
    polygon = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
    for n, offset in zip(G, h):
        polygon = _clip_polygon(polygon, n, offset)
        if not len(polygon):
            return dict(complete=True, empty=True)
    candidates = []
    def add(p):
        if np.isfinite(p).all() and np.dot(p, p) <= 1.+2e-10 and np.all(G@p+h <= 2e-10):
            p=np.asarray(p); p=p/max(1.,np.linalg.norm(p))
            candidates.append(p)
    stationary = np.linalg.lstsq(H, -g, rcond=None)[0]
    if np.linalg.norm(H@stationary+g) <= 1e-10*max(1., np.linalg.norm(H), np.linalg.norm(g)):
        add(stationary)
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        d = b-a; aa = float(d@d)
        if aa < 1e-28:
            add(a); continue
        bb, cc = 2*float(a@d), float(a@a)-1
        disc = bb*bb-4*aa*cc
        if disc < -1e-13:
            continue
        root = np.sqrt(max(0., disc)); lo = max(0., (-bb-root)/(2*aa)); hi = min(1., (-bb+root)/(2*aa))
        if lo > hi+1e-12:
            continue
        add(a+lo*d); add(a+hi*d)
        den = float(d@H@d)
        if abs(den) > 1e-25:
            t = -float(d@(H@a+g))/den
            if lo <= t <= hi:
                add(a+t*d)
    d = H[1, 1]-H[0, 0]; cross = H[0, 1]
    coeff = np.array([cross+g[1], 2*(d-g[0]), -6*cross, -2*(d+g[0]), cross-g[1]])
    scale = np.max(np.abs(coeff))
    if scale > 1e-28:
        roots = np.roots(np.trim_zeros((coeff/scale)[::-1], 'f'))
        for t in roots:
            if abs(t.imag) <= 1e-8*(1+abs(t.real)):
                angle = 2*np.arctan(t.real); add(np.array([np.cos(angle), np.sin(angle)]))
    for angle in (0., np.pi/2, np.pi, 3*np.pi/2):
        add(np.array([np.cos(angle), np.sin(angle)]))
    if not candidates:
        # Polygon is either outside disk or numerically unresolved. Prove the
        # former with its exact edge-distance minimum; never turn failure clear.
        closest = []
        for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
            d = b-a; t = np.clip(-a@d/max(d@d, 1e-30), 0, 1); closest.append(np.linalg.norm(a+t*d))
        origin_inside = np.all(G@np.zeros(2)+h <= 0)
        if not origin_inside and min(closest) > 1.+1e-10:
            return dict(complete=True, empty=True)
        return dict(complete=False, empty=False, reason='REGION_NUMERICALLY_UNCERTIFIED')
    pts = np.array(candidates); values = np.einsum('ni,ij,nj->n', pts, H, pts)+2*pts@g
    lo, hi = int(values.argmin()), int(values.argmax())
    return dict(complete=True, empty=False, minimum_point=pts[lo], maximum_point=pts[hi],
        minimum_value=float(values[lo]), maximum_value=float(values[hi]), candidates=len(pts))


def _patch_parameterization(planes, patch):
    eq = _array(planes); c = _finite(patch['center_local_m'], (3,), 'patch center')
    n = _finite(patch['reference_outward_normal_local'], (3,), 'patch reference')
    u, v = (_finite(patch[k], (3,), k) for k in ('tangent_u', 'tangent_v'))
    half = _finite(patch['half_widths_m'], (2,), 'ellipse half axes')
    if (np.any(half <= 0) or not np.allclose(np.array([u, v, n])@np.array([u, v, n]).T,
                                         np.eye(3), atol=1e-7, rtol=0)):
        raise ValueError('orthonormal patch frame and positive ellipse required')
    eligible = np.unique(np.round(_array(patch['planes_local']), 12), axis=0)
    if eligible.ndim != 2 or eligible.shape[1] != 4 or not len(eligible):
        raise ValueError('source facet planes required')
    rows = []
    for p in eligible:
        distance = np.max(np.abs(eq-p), axis=1)
        if distance.min() > 2e-9 or p[:3]@n <= .5:
            raise ValueError('patch facet does not match full source hull')
        p = eq[int(distance.argmin())]
        B = np.column_stack((half[0]*u, half[1]*v)); denom = p[:3]@n
        base = c-n*((p[:3]@c+p[3])/denom)
        A = B-n[:, None]*(p[:3]@B)[None]/denom
        rows.append((base, A, p))
    return c, rows, eligible


def _compiled_region(planes, patch):
    """Cache a mathematically identical 2D polygon for each fixed source facet."""
    names=('center_local_m','reference_outward_normal_local','tangent_u','tangent_v','half_widths_m','planes_local')
    signature=json.dumps({k:_array(patch[k]).tolist() for k in names},sort_keys=True,separators=(',',':'))
    key=hashlib.sha256(_array(planes).tobytes()+signature.encode()).hexdigest()
    if key in _REGION_CACHE:return _REGION_CACHE[key]
    center, source_rows, eligible=_patch_parameterization(planes,patch);eq=_array(planes);rows=[]
    for base,A,face in source_rows:
        G,h=eq[:,:3]@A,eq[:,:3]@base+eq[:,3];norms=np.linalg.norm(G,axis=1)
        zero=norms<1e-15
        if np.any(h[zero]>1e-11) or np.any(h>norms+1e-11):continue
        gg,hh=G[~zero]/norms[~zero,None],h[~zero]/norms[~zero]
        polygon=np.array([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]])
        for n,offset in zip(gg,hh):
            polygon=_clip_polygon(polygon,n,offset)
            if not len(polygon):break
        if not len(polygon):continue
        # Degenerate face/ellipse intersections keep the original inequalities.
        area=abs(np.cross(polygon,np.roll(polygon,-1,axis=0)).sum())*.5
        if area>1e-14:
            reduced=[];offsets=[]
            for a,b in zip(polygon,np.roll(polygon,-1,axis=0)):
                d=b-a;n=np.array([d[1],-d[0]]);length=np.linalg.norm(n)
                if length>1e-14:
                    n=n/length;reduced.append(n);offsets.append(-n@a)
            G,h=np.array(reduced),np.array(offsets)
        rows.append((base,A,face,G,h))
    result=(center,rows,eligible,len(source_rows))
    if len(_REGION_CACHE)>=32:_REGION_CACHE.pop(next(iter(_REGION_CACHE)))
    _REGION_CACHE[key]=result
    return result


def _rod(rod):
    t = pose_matrix(rod['pose_w']); radius, length = float(rod['radius_m']), float(rod['length_m'])
    if not np.isfinite([radius, length]).all() or min(radius, length) <= 0:
        raise ValueError('explicit positive finite cylinder dimensions required')
    return t[:3, 3], t[:3, 1], radius, length/2


def _facet_closest(base, A, G, h, origin, axis, radius):
    P = np.eye(3)-np.outer(axis, axis); b = P@(base-origin); B = P@A
    result = quadratic_disk_polygon_extrema(B.T@B, B.T@b, G, h)
    if not result['complete'] or result['empty']:
        return result
    zlo, zhi = result['minimum_point'], result['maximum_point']
    flo, fhi = np.linalg.norm(b+B@zlo)**2, np.linalg.norm(b+B@zhi)**2
    if flo <= radius**2 <= fhi:
        lo, hi = 0., 1.
        for _ in range(55):
            mid = (lo+hi)/2; z = zlo+(zhi-zlo)*mid
            if np.linalg.norm(b+B@z)**2 < radius**2:
                lo = mid
            else:
                hi = mid
        z = zlo+(zhi-zlo)*(lo+hi)/2
    else:
        z = zlo if flo > radius**2 else zhi
    result.update(point=base+A@z, coordinates=z, minimum_radial_m=float(np.sqrt(max(0., flo))))
    return result


def audit_pad_region(planes, patch, pose, rod, *, contact_distance_m, normal_cosine_min):
    """All facets, same contact witness; distance and direction stay separate."""
    pose = _rigid(pose); eq = _array(planes); center, rows, _, source_facet_count = _compiled_region(eq, patch)
    origin, axis, radius, half = _rod(rod); witnesses = []; facing_witnesses = []; complete = True
    for base, A, face, local_G, local_h in rows:
        world_base, world_A = pose[:3, :3]@base+pose[:3, 3], pose[:3, :3]@A
        normal = pose[:3, :3]@face[:3]
        G, h = local_G, local_h
        axial_base = axis@(world_base-origin); axial_A = axis@world_A
        G = np.r_[G, axial_A[None], -axial_A[None]]
        h = np.r_[h, axial_base-half, -axial_base-half]
        normal_radial = normal-axis*(normal@axis); magnitude = np.linalg.norm(normal_radial)
        domains = [(G, h, witnesses)]
        if magnitude >= normal_cosine_min:
            direction = -normal_radial/magnitude; transverse = np.cross(axis, direction)
            tangent = np.sqrt(max(0., (magnitude/normal_cosine_min)**2-1))
            n1, n2 = transverse-tangent*direction, -transverse-tangent*direction
            domains.append((np.r_[G, (n1@world_A)[None], (n2@world_A)[None],(-direction@world_A)[None]],
                np.r_[h, n1@(world_base-origin), n2@(world_base-origin),-direction@(world_base-origin)], facing_witnesses))
        for gg, hh, destination in domains:
            fit = _facet_closest(world_base, world_A, gg, hh, origin, axis, radius)
            complete &= fit['complete']
            if not fit['complete'] or fit['empty']:
                continue
            point = fit['point']; axial = axis@(point-origin); radial = point-origin-axial*axis
            length = np.linalg.norm(radial); local = (point-pose[:3, 3])@pose[:3, :3]
            signed = eq[:, :3]@local+eq[:, 3]
            cosine = float(-normal@radial/max(length, 1e-30)); gap = float(length-radius)
            destination.append(dict(point_w_m=point, point_local_m=local,
                object_surface_point_w_m=origin+axis*axial+radial*(radius/max(length, 1e-30)),
                normal_w=normal, distance_m=gap, facing_cosine=cosine,
                axial_m=float(axial), sidewall=bool(abs(axial) <= half),
                ellipse_squared_radius=float(fit['coordinates']@fit['coordinates']),
                source_surface_residual_m=float(abs(signed.max())),
                minimum_region_signed_radial_gap_m=fit['minimum_radial_m']-radius))
    closest = min(witnesses, key=lambda w: abs(w['distance_m'])) if witnesses else None
    same = min(facing_witnesses, key=lambda w: abs(w['distance_m'])) if facing_witnesses else None
    chosen = same if same is not None else closest
    distance_pass = complete and closest is not None and abs(closest['distance_m']) <= contact_distance_m
    direction_pass = (complete and same is not None and abs(same['distance_m']) <= contact_distance_m
                      and same['facing_cosine'] >= normal_cosine_min and same['sidewall']
                      and same['ellipse_squared_radius'] <= 1.)
    return dict(complete=bool(complete), distance_pass=bool(distance_pass), direction_pass=bool(direction_pass),
        contact_geometry_qualified=bool(direction_pass and distance_pass), witness=chosen,
        nearest_distance_witness=closest, facet_count=source_facet_count, intersecting_facet_domains=len(rows),
        method='all_source_facets_full_halfspaces_ellipse_global_quadratic_extrema',
        center_is_only_region_seed=True, native_contact_verified=False)


@dataclass(frozen=True)
class GeometryThresholds:
    contact_distance_m: float
    normal_cosine_min: float
    object_overlap_m: float
    noncontact_object_clearance_m: float
    table_clearance_m: float
    self_clearance_m: float
    interhand_clearance_m: float
    obstacle_clearance_m: float
    max_command_step_rad: float
    command_lead_rad: float
    sources: dict

    def __post_init__(self):
        names = set(self.__dataclass_fields__)-{'sources'}
        if not self.sources or any(not self.sources.get(k) for k in names):
            raise ValueError('explicit threshold source required for every field')
        if any(not np.isfinite(getattr(self, k)) or getattr(self, k) < 0 for k in names):
            raise ValueError('finite nonnegative thresholds required')
        if not 0 < self.normal_cosine_min <= 1:
            raise ValueError('positive normal cosine required')


def convex_clearance_certificate(a, b, threshold, *, motion_bound=0.):
    gaps = np.maximum(a['lower']-b['upper'], b['lower']-a['upper'])
    bound = float(gaps.max())
    if bound < threshold+motion_bound:
        axes = np.r_[np.eye(3), a['planes'][:, :3], b['planes'][:, :3]]
        bound = separation_lower_bound(a['vertices'], b['vertices'], axes)
    bound -= motion_bound
    passed = bool(bound >= threshold)
    return dict(passed=passed, status='CERTIFIED' if passed else 'UNCERTIFIED',
        lower_bound_m=float(bound), threshold_m=float(threshold), sampled_fallback=False,
        meaning='failed supplied-axis certificate is not proof of collision')


def hull_rod_certificate(hull, rod, threshold, *, motion_bound=0.):
    center, axis, radius, half = _rod(rod)
    nodes = np.array([center-half*axis, center+half*axis])
    bound = hull_wire_lower_bound(hull['planes'], nodes, radius, .002,
        refine_below=float(threshold+motion_bound+.001))-motion_bound
    passed = bool(bound >= threshold)
    return dict(passed=passed, status='CERTIFIED' if passed else 'UNCERTIFIED',
        lower_bound_m=float(bound), threshold_m=float(threshold), sampled_fallback=False,
        whole_rod_checked=True, object_proxy='conservative_capsule_contains_finite_cylinder')


def _minimum_segment_distance(base, A, G, h, origin, axis, half):
    """Exact piecewise quadratic distance from convex face domain to segment."""
    axial_A = axis@A; axial = axis@(base-origin); best = np.inf; complete = True
    P = np.eye(3)-np.outer(axis, axis)
    pieces = [(P, origin, np.r_[G, axial_A[None], -axial_A[None]],
               np.r_[h, axial-half, -axial-half]),
              (np.eye(3), origin+half*axis, np.r_[G, -axial_A[None]], np.r_[h, half-axial]),
              (np.eye(3), origin-half*axis, np.r_[G, axial_A[None]], np.r_[h, half+axial])]
    for projection, target, gg, hh in pieces:
        B, b = projection@A, projection@(base-target)
        result = quadratic_disk_polygon_extrema(B.T@B, B.T@b, gg, hh)
        complete &= result['complete']
        if result['complete'] and not result['empty']:
            best = min(best, float(np.linalg.norm(B@result['minimum_point']+b)))
    return best, complete


def nonpad_contact_certificate(vertices, planes, patch, pose, rod, *,
                               normal_cosine_min, motion_bound=0., normal_motion_bound=0.,
                               object_overlap_m=0., polygon_sides=16):
    """Certify that possible capsule contact lies only on correct-facing pad.

    An inscribed polygon narrows the existing ellipse; it never enlarges the
    allowed region. Every source facet outside that polygon, facing cone or
    finite sidewall is checked, including non-pad distal faces. A failed bound
    is UNCERTIFIED, not a claim of collision. Source faces are never sampled.
    """
    vertices, eq, pose = _array(vertices), _array(planes), _rigid(pose)
    origin, axis, radius, half = _rod(rod); c, _, eligible = _patch_parameterization(eq, patch)
    denominator=radius-object_overlap_m-motion_bound
    radial_change=(2*motion_bound/denominator) if denominator>0 else np.inf
    required_cosine=normal_cosine_min+normal_motion_bound+radial_change
    permitted_half=half-motion_bound
    nref = _array(patch['reference_outward_normal_local']); u = _array(patch['tangent_u']); v = _array(patch['tangent_v'])
    widths = _array(patch['half_widths_m']); best = np.inf; complete = True; domains = 0
    unique = np.unique(np.round(eq, 12), axis=0)
    for face in unique:
        n = face[:3]; ids = np.abs(vertices@n+face[3]) <= 2e-9
        if not ids.any():
            return dict(passed=False, complete=False, status='UNCERTIFIED', reason='SOURCE_FACE_IDENTITY')
        base = vertices[ids].mean(0)
        helper = np.eye(3)[np.argmin(np.abs(n))]; fu = np.cross(n, helper); fu /= np.linalg.norm(fu); fv = np.cross(n, fu)
        extent = float(np.linalg.norm(vertices[ids]-base, axis=-1).max())+1e-8
        A = np.column_stack((fu, fv))*extent
        G, h = eq[:, :3]@A, eq[:, :3]@base+eq[:, 3]
        world_base = pose[:3, :3]@base+pose[:3, 3]; world_A = pose[:3, :3]@A
        normal = pose[:3, :3]@n
        permitted = np.max(np.abs(eligible-face), axis=1).min() <= 2e-9
        normal_radial = normal-axis*(normal@axis); magnitude = np.linalg.norm(normal_radial)
        exclusions = []
        if not permitted or magnitude < required_cosine or permitted_half <= 0:
            exclusions.append((None, None))
        else:
            for angle in np.arange(polygon_sides)*2*np.pi/polygon_sides:
                d = np.cos(angle)*u/widths[0]+np.sin(angle)*v/widths[1]
                exclusions.append((-d@A, np.cos(np.pi/polygon_sides)-d@(base-c)))
            direction = -normal_radial/magnitude; transverse = np.cross(axis, direction)
            tangent = np.sqrt(max(0., (magnitude/required_cosine)**2-1))
            for d in (transverse-tangent*direction, -transverse-tangent*direction):
                exclusions.append((-d@world_A, -d@(world_base-origin)))
            exclusions.append((direction@world_A,direction@(world_base-origin)))
            axial = axis@(world_base-origin); axial_A = axis@world_A
            exclusions.extend(((-axial_A, permitted_half-axial), (axial_A, permitted_half+axial)))
        for extra_g, extra_h in exclusions:
            gg = G if extra_g is None else np.r_[G, extra_g[None]]
            hh = h if extra_h is None else np.r_[h, extra_h]
            distance, done = _minimum_segment_distance(world_base, world_A, gg, hh, origin, axis, half)
            best = min(best, distance-radius-motion_bound); complete &= done; domains += 1
    # If all excluded domains were empty, there is no non-pad surface.
    passed = complete and best >= 0.
    return dict(passed=bool(passed), complete=bool(complete), status='CERTIFIED' if passed else 'UNCERTIFIED',
        lower_bound_m=float(best) if np.isfinite(best) else None,
        source_facets_checked=len(unique), excluded_domains_checked=domains,
        swept_required_facing_cosine=float(required_cosine) if np.isfinite(required_cosine) else None,
        normal_motion_bound=float(normal_motion_bound),radial_unit_motion_bound=float(radial_change) if np.isfinite(radial_change) else None,
        permitted_sidewall_half_length_m=float(permitted_half),
        allowed_region='inscribed_16gon_subset_of_original_ellipse_and_facing_cone_and_sidewall',
        sampled_fallback=False)


def validate_substep_coverage(right_deltas, encoded_first_delta):
    x = _array(right_deltas); first = _array(encoded_first_delta); reasons = []
    if x.ndim != 3 or x.shape[1:] != (8, 6) or not len(x) or not np.isfinite(x).all():
        reasons.append('EIGHT_SUBSTEPS_REQUIRED')
    elif first.shape != (6,) or not np.isfinite(first).all() or not np.allclose(first, x[0, -1], atol=1e-12, rtol=0):
        reasons.append('ENCODED_FIRST_MISMATCH')
    return dict(passed=not reasons, reasons=reasons, required_substeps_per_action=8,
        geometry_interpolation='joint-local rotations with interval displacement certificate')


def build_command_trajectory(snapshot, commands, *, response, substeps=8):
    """Explicit response approximation starts from actual, tracks absolute q.

    response is per control action, not per physics tick. This helper does not
    claim native drive/mimic tracking; native full-q motion validation is
    independently available through predict_full_delta().
    """
    commands=_array(commands)
    if commands.ndim != 2 or commands.shape[1] != 6 or not len(commands) or not np.isfinite(commands).all():
        raise ValueError('finite H x 6 absolute commands required')
    if not np.isfinite(response) or not 0 < response <= 1 or substeps != 8:
        raise ValueError('explicit response in (0,1] and eight substeps required')
    initial=_finite(snapshot['full_q'],(11,),'actual full q')[[JOINTS.index(n) for n in ACTIVE]]
    current=initial.copy();alpha=1-(1-float(response))**(1/substeps);result=[]
    for command in commands:
        row=[]
        for _ in range(substeps):
            current=current+alpha*(command-current);row.append(current-initial)
        result.append(row)
    return np.asarray(result)


class TripodNonlinearGeometry:
    """Fixed-wrist state audit with explicit increments for either hand.

    audit_trajectory is the right-only pure helper. A native caller holding a
    left command must predict left tracking too and call audit_state with that
    left_delta at every state and swept midpoint, plus both motion bounds.

    scene: rod mapping, table_top_z_m, obstacles[{id,vertices_w_m}],
    obstacles_complete=True and nonempty provenance. Limits are separate arrays
    supplied to audit_trajectory, never inferred from proposal defaults.
    """
    def __init__(self, left, right, scene, thresholds):
        if (left.side, right.side) != ('left', 'right'):
            raise ValueError('ordered left/right source models required')
        if scene.get('obstacles_complete') is not True or not scene.get('provenance'):
            raise ValueError('verified complete original obstacles/provenance required')
        _rod(scene['rod'])
        if not np.isfinite(scene['table_top_z_m']):
            raise ValueError('explicit finite table top required')
        self.models = [left, right]; self.scene = copy.deepcopy(scene); self.thresholds = thresholds
        self.obstacles = [world_hull(o['vertices_w_m'], np.eye(4), o['id']) for o in scene['obstacles']]

    def audit_state(self, snapshots, right_delta, *, left_delta=None,motion_bounds=None, normal_motion_bounds=None,
                    bearing_mask=None,bearing_provenance=None):
        started = time.perf_counter(); t = self.thresholds; reasons = []; pads = []; predictions = []
        if bearing_mask is None:bearing_mask=np.zeros((2,5),dtype=bool)
        else:
            if hasattr(bearing_mask,'detach'):bearing_mask=bearing_mask.detach().cpu().numpy()
            bearing_mask=np.asarray(bearing_mask)
            if bearing_mask.shape!=(2,5) or bearing_mask.dtype!=np.bool_:
                raise ValueError('native bearing mask must be bool 2 x 5')
        if bearing_mask.any():
            p=bearing_provenance
            if (not isinstance(p,dict) or not p.get('source') or not p.get('snapshot_identity')
                    or p.get('native_accepted_contacts') is not True
                    or not isinstance(p.get('physics_step'),int) or p['physics_step']<0):
                raise ValueError('same-snapshot native bearing provenance required')
        if bearing_mask[:,3:].any():reasons.append('NONDESIGNATED_BEARING')
        for hi, model in enumerate(self.models):
            prediction = model.predict(snapshots[hi], (np.zeros(6) if left_delta is None else left_delta) if hi == 0 else right_delta)
            predictions.append(prediction); rows = {}
            for digit, patch in model.patches.items():
                s = next(s for s in model.shapes if s['body'] == patch['body'])
                rows[digit] = audit_pad_region(s['planes'], patch, prediction['poses'][patch['body']],
                    self.scene['rod'], contact_distance_m=t.contact_distance_m,
                    normal_cosine_min=t.normal_cosine_min)
            pads.append(rows)
        count_ok = all(len(p['hulls']) == 17 for p in predictions)
        if not count_ok:
            reasons.append('ALL_34_SOURCE_HULLS_REQUIRED')
        if motion_bounds is None:
            motion_bounds = [np.zeros(len(p['hulls'])) for p in predictions]
        if normal_motion_bounds is None:
            if any(np.any(_array(b)>0) for b in motion_bounds):
                reasons.append('SWEPT_NORMAL_COVERAGE_MISSING')
            normal_motion_bounds=[np.zeros(len(p['hulls'])) for p in predictions]
        collision = dict(object=[], table=[], self=[], interhand=[], obstacles=[])
        contact_formation_safe = True
        for hi, (model, p) in enumerate(zip(self.models, predictions)):
            for i, hull in enumerate(p['hulls']):
                patch_digit = next((d for d, row in model.patches.items() if row['body'] == hull['body']), None)
                qualifies = patch_digit is not None and pads[hi][patch_digit]['contact_geometry_qualified']
                # No blanket distal overlap exemption for a far/wrong-facing pad.
                native_bearing=(patch_digit is not None and bool(bearing_mask[hi,DIGITS.index(patch_digit)])
                                and DIGITS.index(patch_digit)<3)
                threshold = -t.object_overlap_m if qualifies and native_bearing else t.noncontact_object_clearance_m
                check = hull_rod_certificate(hull, self.scene['rod'], threshold, motion_bound=motion_bounds[hi][i])
                check.update(side=model.side, body=hull['body'],native_bearing_overlap_permitted=bool(qualifies and native_bearing))
                collision['object'].append(check)
                table = float(hull['lower'][2]-self.scene['table_top_z_m']-motion_bounds[hi][i])
                collision['table'].append(dict(side=model.side, body=hull['body'], clearance_m=table,
                    threshold_m=t.table_clearance_m, passed=table >= t.table_clearance_m))
                if not check['passed']:
                    reasons.append(model.side+'/'+hull['body']+':OBJECT_UNCERTIFIED')
                if table < t.table_clearance_m:
                    reasons.append(model.side+'/'+hull['body']+':TABLE_CLEARANCE')
                if check['lower_bound_m'] < 0:
                    nonpad = (nonpad_contact_certificate(model.shapes[i]['vertices'], model.shapes[i]['planes'],
                        model.patches[patch_digit], p['poses'][hull['body']], self.scene['rod'],
                        normal_cosine_min=t.normal_cosine_min, motion_bound=motion_bounds[hi][i],
                        normal_motion_bound=normal_motion_bounds[hi][i],object_overlap_m=t.object_overlap_m)
                        if qualifies and native_bearing else dict(passed=False, complete=True, status='UNCERTIFIED'))
                    check['nonpad_contact'] = nonpad
                    if not nonpad['passed']:
                        contact_formation_safe = False
                        reasons.append(model.side+'/'+hull['body']+':NONPAD_CONTACT_NOT_CERTIFIED')
                for obstacle in self.obstacles:
                    row = convex_clearance_certificate(hull, obstacle, t.obstacle_clearance_m,
                        motion_bound=motion_bounds[hi][i])
                    row.update(side=model.side, body=hull['body'], obstacle_id=obstacle['body'])
                    collision['obstacles'].append(row)
                    if not row['passed']:
                        reasons.append(model.side+'/'+hull['body']+':ORIGINAL_OBSTACLE_UNCERTIFIED')
            for i, j in model.self_pairs:
                row = convex_clearance_certificate(p['hulls'][i], p['hulls'][j], t.self_clearance_m,
                    motion_bound=motion_bounds[hi][i]+motion_bounds[hi][j])
                row.update(side=model.side, indices=[i, j]); collision['self'].append(row)
                if not row['passed']:
                    reasons.append(model.side+':SELF_UNCERTIFIED')
        for i, a in enumerate(predictions[0]['hulls']):
            for j, b in enumerate(predictions[1]['hulls']):
                row = convex_clearance_certificate(a, b, t.interhand_clearance_m,
                    motion_bound=motion_bounds[0][i]+motion_bounds[1][j])
                row.update(indices=[i, j]); collision['interhand'].append(row)
                if not row['passed']:
                    reasons.append('INTERHAND_UNCERTIFIED')
        motion_safe = not reasons
        qualified = all(all(pads[hi].get(d, {}).get('contact_geometry_qualified', False)
            for d in ('thumb', 'index', 'middle')) for hi in (0, 1))
        return dict(motion_safe=motion_safe, contact_geometry_qualified=qualified,
            contact_formation_safe=contact_formation_safe, accepted=motion_safe,
            audit_complete=count_ok and all(r['complete'] for hand in pads for r in hand.values()),
            collision=collision, pads=pads, predictions=predictions,
            rejection_reasons=list(dict.fromkeys(reasons)), native_contact_verified=False,
            source_provenance_verified=all(m.source_provenance_verified for m in self.models),native_candidate_eligible=False,
            bearing_mask=bearing_mask.copy(),bearing_provenance=copy.deepcopy(bearing_provenance),
            left_prediction_scope='fixed_actual_geometry_only' if left_delta is None else 'explicit_left_joint_increment',
            source_hull_counts=[len(p['hulls']) for p in predictions], elapsed_s=time.perf_counter()-started)

    def audit_trajectory(self, snapshots, right_deltas, encoded_first_delta, *,
                         command_trajectory, initial_command, active_lower, active_upper,
                         full_lower, full_upper, limit_sources, response, encoded_first_command,
                         bearing_mask=None,bearing_provenance=None):
        coverage = validate_substep_coverage(right_deltas, encoded_first_delta)
        if not coverage['passed']:
            return dict(accepted=False, motion_safe=False, audit_complete=False,
                coverage=coverage, rejection_reasons=coverage['reasons'])
        if not limit_sources or any(not limit_sources.get(k) for k in ('active', 'full', 'command', 'lead', 'response')):
            raise ValueError('explicit limit sources required')
        x = _array(right_deltas); H = len(x); commands = _finite(command_trajectory, (H, 6), 'command trajectory')
        initial = _finite(initial_command, (6,), 'initial command')
        alo, ahi = (_finite(v, (6,), n) for v, n in ((active_lower, 'active lower'), (active_upper, 'active upper')))
        flo, fhi = (_finite(v, (2, 11), n) for v, n in ((full_lower, 'full lower'), (full_upper, 'full upper')))
        increments = np.diff(np.r_[initial[None], commands], axis=0); t = self.thresholds
        active_ids = [JOINTS.index(n) for n in ACTIVE]; q0 = snapshots[1]['full_q'][active_ids]
        expected=build_command_trajectory(snapshots[1],commands,response=response)
        encoded=_finite(encoded_first_command,(6,),'encoded first command')
        joint_checks = dict(command_limits=bool(((commands >= alo)&(commands <= ahi)).all()),
            canonical_slew=bool((abs(increments) <= t.max_command_step_rad).all()),
            command_lead=bool((abs(commands-(q0+np.r_[np.zeros((1,6)),x[:-1,-1]])) <= t.command_lead_rad).all()),
            actual_active_limits=all(bool(((s['full_q'][active_ids]>=alo)&(s['full_q'][active_ids]<=ahi)).all()) for s in snapshots),
            predicted_active_limits=bool(((q0+x>=alo)&(q0+x<=ahi)).all()),
            command_state_coherence=bool(np.allclose(x,expected,atol=1e-12,rtol=0)),
            encoded_first_matches_command=bool(np.array_equal(encoded,commands[0])),
            actual_full_limits=all(bool(((s['full_q'] >= flo[h])&(s['full_q'] <= fhi[h])).all()) for h, s in enumerate(snapshots)),
            predicted_full_limits=True, sources=copy.deepcopy(limit_sources))
        if not joint_checks['command_state_coherence'] or not joint_checks['encoded_first_matches_command']:
            reasons=[]
            if not joint_checks['command_state_coherence']:reasons.append('COMMAND_STATE_MISMATCH')
            if not joint_checks['encoded_first_matches_command']:reasons.append('ENCODED_COMMAND_MISMATCH')
            return dict(accepted=False,motion_safe=False,audit_complete=False,coverage=coverage,
                joint_audit=joint_checks,rejection_reasons=reasons,response_model='actual_q_tracks_absolute_command',
                response=float(response),threshold_sources=copy.deepcopy(t.sources),native_contact_verified=False)
        if not all(v for k,v in joint_checks.items() if k!='sources'):
            return dict(accepted=False,motion_safe=False,audit_complete=False,coverage=coverage,
                joint_audit=joint_checks,rejection_reasons=[k.upper() for k,v in joint_checks.items() if k!='sources' and not v],
                response_model='actual_q_tracks_absolute_command',response=float(response),
                threshold_sources=copy.deepcopy(t.sources),native_contact_verified=False)
        states = []; swept = []; previous = np.zeros(6)
        for h in range(H):
            for substep in range(8):
                delta = x[h, substep]
                full = snapshots[1]['full_q']+coupling_matrix()@delta
                joint_checks['predicted_full_limits'] &= bool(((full >= flo[1])&(full <= fhi[1])).all())
                states.append(self.audit_state(snapshots, delta,bearing_mask=bearing_mask,bearing_provenance=bearing_provenance))
                bounds = [np.zeros(len(self.models[0].shapes)),
                    self.models[1].motion_bounds(snapshots[1], previous, delta)]
                normal_bounds=[np.zeros(len(self.models[0].shapes)),self.models[1].normal_motion_bounds(previous,delta)]
                swept.append(self.audit_state(snapshots, (previous+delta)/2, motion_bounds=bounds,normal_motion_bounds=normal_bounds,
                    bearing_mask=bearing_mask,bearing_provenance=bearing_provenance))
                previous = delta
        safe = all(v for k, v in joint_checks.items() if k != 'sources') and all(r['motion_safe'] for r in states+swept)
        complete = all(r['audit_complete'] for r in states+swept)
        reasons = [k.upper() for k, v in joint_checks.items() if k != 'sources' and not v]
        reasons += [reason for r in states+swept for reason in r['rejection_reasons']]
        return dict(accepted=bool(safe and complete), motion_safe=bool(safe and complete),
            contact_geometry_qualified=states[-1]['contact_geometry_qualified'],
            contact_formation_safe=all(r['contact_formation_safe'] for r in states+swept),
            audit_complete=complete, coverage=dict(coverage, horizon=H, states=len(states), swept_intervals=len(swept)),
            joint_audit=joint_checks, states=states, swept=swept,
            threshold_sources=copy.deepcopy(t.sources), limit_sources=copy.deepcopy(limit_sources),
            source_provenance_verified=all(m.source_provenance_verified for m in self.models),native_candidate_eligible=False,
            bearing_provenance=copy.deepcopy(bearing_provenance),response=float(response),response_model='actual_q_tracks_absolute_command',
            left_prediction_scope='fixed_actual_geometry_only; not held-command tracking proof',
            rejection_reasons=list(dict.fromkeys(reasons)), native_contact_verified=False,
            substeps_are='explicit candidate joint states, not manufactured native measurements')
