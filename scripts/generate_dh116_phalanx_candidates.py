"""Bounded offline USD phalanx search. Geometry proposals, never a teacher.

No simulator import/instance, measured force synthesis, or repository mutation.
Contact points retain source hull triangle, barycentric coordinates, and body.
Optimization proxies are used only for search; acceptance uses the unchanged
production model's full collision samples/planes and self exclusion graph.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import torch

from go2_pvcnn.control.dh116 import tabletop_surface_kinematics as kinematics
from go2_pvcnn.control.dh116.tabletop_surface_candidates import object_surface_points, rank_diverse
from go2_pvcnn.control.dh116.tabletop_surface_objectives import finite_cylinder_sdf, _geometry
from go2_pvcnn.control.dh116.tabletop_surface_teacher import implementation_identity


RADIUS = .008
TABLE_Z = .30
CLEARANCE = .0003
DEPTH = .00025
ProductionSurfaceHandModel = kinematics.SurfaceHandModel


def emit(event, **values):
    print(event, json.dumps(values, allow_nan=False), flush=True)


def origin(model, n):
    pose = model.t([[0., 0., 0., 1., 0., 0., 0.]]).repeat(n, 1)
    return pose


def surface_catalog(model, count=20):
    """Area samples and facet centroids, all on actual source hull triangles."""
    rng = np.random.default_rng(918)
    result = []
    for leaves in (("finger11_link", "finger12_link", "finger13_link", "finger14_link"),
                   ("finger21_link", "finger22_link", "finger23_link")):
        points, normals, triangles, barys, bodies = [], [], [], [], []
        for leaf in leaves:
            rows = [r for r in model.spec['shapes'] if r['body'].split('/')[-1] == leaf]
            if not rows:
                raise ValueError('missing source phalanx: ' + leaf)
            for shape in rows:
                v = np.asarray(shape['vertices'])
                hull = ConvexHull(v)
                tri = v[hull.simplices]
                area = np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)
                choices = rng.choice(len(tri), 512, p=area/area.sum())
                uv = rng.random((512, 2))
                s = np.sqrt(uv[:, 0])
                weights = np.column_stack((1-s, s*(1-uv[:, 1]), s*uv[:, 1]))
                # Include facet centroids so small faces have an opportunity.
                choices = np.r_[choices, np.arange(len(tri))]
                weights = np.r_[weights, np.full((len(tri), 3), 1/3)]
                p = (tri[choices]*weights[:, :, None]).sum(1)
                normal = hull.equations[choices, :3]
                selected = [int(area[choices].argmax())]
                distance = np.full(len(p), np.inf)
                for _ in range(count-1):
                    last = selected[-1]
                    # Include different outward normal directions, not just tips.
                    d = ((p-p[last])**2).sum(1) + .0001*(1-normal@normal[last])
                    distance = np.minimum(distance, d)
                    selected.append(int(distance.argmax()))
                for i in selected:
                    points.append(p[i]); normals.append(normal[i])
                    triangles.append(tri[choices[i]]); barys.append(weights[i])
                    bodies.append(shape['body'])
        result.append(dict(points=model.t(np.array(points)), normals=model.t(np.array(normals)),
                           triangles=model.t(np.array(triangles)), bary=model.t(np.array(barys)),
                           bodies=bodies))
    return result


def catalog_world(model, poses, catalog):
    ps, ns = [], []
    for row in catalog:
        transforms = torch.stack([poses[b] for b in row['bodies']], 1)
        ps.append(torch.einsum('npij,pj->npi', transforms[:, :, :3, :3], row['points'])
                  + transforms[:, :, :3, 3])
        ns.append(torch.einsum('npij,pj->npi', transforms[:, :, :3, :3], row['normals']))
    return ps, ns


def basis_from_contacts(p, normal):
    a, b = normal[:, 0], normal[:, 1]
    axis = torch.linalg.cross(a, b)
    axis = axis/axis.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    bisector = a+b
    bisector = bisector/bisector.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    third = torch.linalg.cross(axis, bisector)
    basis = torch.stack((axis, bisector, third), -1)
    center = (p+RADIUS*normal).mean(1)
    return basis, center


def choose_pair(model, q, target_dot, catalog):
    poses = model.fk(q, origin(model, len(q)))
    p, norm = catalog_world(model, poses, catalog)
    a, b = p[0]+RADIUS*norm[0], p[1]+RADIUS*norm[1]
    delta = b[:, None]-a[:, :, None]
    first = (norm[0][:, :, None]*delta).sum(-1)/RADIUS
    second = (norm[1][:, None]*delta).sum(-1)/RADIUS
    dot = (norm[0][:, :, None]*norm[1][:, None]).sum(-1)
    cost = first.square()+second.square()+.5*(dot-target_dot[:, None, None]).square()
    loss, indices = cost.flatten(1).min(-1)
    return loss, indices


def selected_contacts(model, poses, selected, logits):
    weights = logits.softmax(-1)
    local = (selected['triangles']*weights[:, :, :, None]).sum(2)
    p, normals = [], []
    for digit in range(2):
        transforms = torch.stack([poses[b][i] for i, b in enumerate(selected['bodies'][digit])])
        p.append(torch.einsum('nij,nj->ni', transforms[:, :3, :3], local[:, digit]) + transforms[:, :3, 3])
        normals.append(torch.einsum('nij,nj->ni', transforms[:, :3, :3], selected['normals'][:, digit]))
    return torch.stack(p, 1), torch.stack(normals, 1), local


def sparse_vertices(model, poses, count=24):
    values = []
    for shape, row in zip(model.shapes, model.spec['shapes']):
        vertices = np.asarray(row['vertices'])
        v = kinematics._inscribed_proxy(vertices, count=count)
        values.append(kinematics.transform(poses[shape['body']], model.t(v)))
    return torch.cat(values, 1)


def optimize_hand(model, catalog, hi, started, deadline, seeds, steps):
    generator = torch.Generator(device=model.device).manual_seed(731+hi)
    n = seeds
    x = torch.nn.Parameter(2*torch.rand(n, 6, device=model.device, dtype=model.dtype, generator=generator)-1)
    target = model.t(np.resize(np.array([-.99, -.95, -.875, -.7, -.4, -.1, .2, .6]), n))
    low, high = model.lower, model.upper
    best = model.t(np.full(n, np.inf)); best_x = x.detach().clone()
    optimizer = torch.optim.Adam([x], lr=.022)
    phase1 = min(180, steps//2)
    done1 = 0
    for tick in range(phase1):
        optimizer.zero_grad(set_to_none=True)
        q = low+(x.clamp(-1, 1)+1)/2*(high-low)
        loss, _ = choose_pair(model, q, target, catalog)
        better = loss.detach()<best
        best_x = torch.where(better[:, None], x.detach(), best_x)
        best = torch.minimum(best, loss.detach())
        loss.sum().backward(); optimizer.step()
        with torch.no_grad(): x.clamp_(-1, 1)
        done1 = tick+1
        if tick % 60 == 0:
            emit('PAIR_SEARCH', hand=model.side, iteration=done1, minimum=float(best.min()), elapsed_s=time.monotonic()-started)
        if time.monotonic()>deadline-35:
            break
    with torch.no_grad():
        q = low+(best_x+1)/2*(high-low)
        _, indices = choose_pair(model, q, target, catalog)
        ia, ib = indices//len(catalog[1]['points']), indices % len(catalog[1]['points'])
        picks = [ia.cpu().tolist(), ib.cpu().tolist()]
        selected = dict(triangles=torch.stack([catalog[d]['triangles'][idx] for d, idx in enumerate((ia, ib))], 1),
                        normals=torch.stack([catalog[d]['normals'][idx] for d, idx in enumerate((ia, ib))], 1),
                        bodies=[[catalog[d]['bodies'][i] for i in picks[d]] for d in range(2)])
        initial_bary = torch.stack([catalog[d]['bary'][idx] for d, idx in enumerate((ia, ib))], 1)
    x = torch.nn.Parameter(best_x.clone())
    logits = torch.nn.Parameter(initial_bary.clamp_min(1e-8).log())
    roll = torch.nn.Parameter(model.t(np.resize(np.linspace(-math.pi, math.pi, 32, endpoint=False), n)))
    optimizer = torch.optim.Adam([x, logits, roll], lr=.018)
    best = model.t(np.full(n, np.inf))
    best_state = (x.detach().clone(), logits.detach().clone(), roll.detach().clone())
    proxy = [(s['body'], model.t(kinematics._inscribed_proxy(np.array(r['vertices']), count=24)))
             for s, r in zip(model.shapes, model.spec['shapes'])]
    done2 = 0
    for tick in range(steps-done1):
        optimizer.zero_grad(set_to_none=True)
        q = low+(x.clamp(-1, 1)+1)/2*(high-low)
        poses = model.fk(q, origin(model, n))
        p, normal, _ = selected_contacts(model, poses, selected, logits)
        centerline = p+RADIUS*normal
        delta = centerline[:, 1]-centerline[:, 0]
        tangent = ((normal*delta[:, None]).sum(-1)/RADIUS).square().sum(-1)
        dot = (normal[:, 0]*normal[:, 1]).sum(-1)
        basis, center = basis_from_contacts(p, normal)
        points = torch.cat([kinematics.transform(poses[b], v) for b, v in proxy], 1)
        relative = points-center[:, None]
        radial = relative-(relative*basis[:, None, :, 0]).sum(-1, keepdim=True)*basis[:, None, :, 0]
        depth = (RADIUS-radial.norm(dim=-1)).clamp_min(0).amax(-1)
        up = roll.cos()[:, None]*basis[:, :, 1]+roll.sin()[:, None]*basis[:, :, 2]
        min_z = (relative*up[:, None]).sum(-1).amin(-1)+.308
        table = (.3003-min_z).clamp_min(0)
        loss = tangent+.1*(dot-target).square()+3*(depth/.008).square()+6*(table/.008).square()
        better = loss.detach()<best
        best_state = tuple(torch.where(better.reshape((-1,)+(1,)*(v.ndim-1)), v.detach(), old)
                           for v, old in zip((x, logits, roll), best_state))
        best = torch.minimum(best, loss.detach())
        loss.sum().backward()
        torch.nn.utils.clip_grad_norm_([x, logits, roll], 20.)
        optimizer.param_groups[0]['lr'] = .018*(.05+.95*(1-tick/max(1, steps-done1))**2)
        optimizer.step()
        with torch.no_grad(): x.clamp_(-1, 1)
        done2 = tick+1
        if tick % 60 == 0:
            emit('TABLE_REFINEMENT', hand=model.side, iteration=done1+done2, minimum=float(best.min()),
                 table_safe=int((min_z>=.3003).sum()), proxy_rod_safe=int((depth<=DEPTH).sum()), elapsed_s=time.monotonic()-started)
        if time.monotonic()>deadline:
            break
    with torch.no_grad():
        q = low+(best_state[0]+1)/2*(high-low)
        poses = model.fk(q, origin(model, n))
        p, normal, local = selected_contacts(model, poses, selected, best_state[1])
        basis, center = basis_from_contacts(p, normal)
    emit('OPTIMIZED', hand=model.side, iterations=done1+done2, seeds=n, elapsed_s=time.monotonic()-started)
    return dict(q=q.detach().cpu().numpy(), p=p.cpu().numpy(), normal=normal.cpu().numpy(),
                local=local.cpu().numpy(), basis=basis.cpu().numpy(), center=center.cpu().numpy(),
                bodies=selected['bodies'], local_normals=selected['normals'].cpu().numpy(), triangles=selected['triangles'].cpu().numpy(),
                bary=best_state[1].softmax(-1).cpu().numpy(), objective=best.cpu().numpy(),
                iterations=done1+done2)


def self_diagnostic(model, poses):
    worst = dict(depth_m=0., pair=None)
    for i, j in model.self_pairs:
        a, b = model.shapes[i], model.shapes[j]
        for source, target in ((a, b), (b, a)):
            points = kinematics.transform(poses[source['body']], source['probe_points'])
            depth = float(model._inside(points, poses[target['body']], target, fast=False)[0])
            if depth>worst['depth_m']:
                worst = dict(depth_m=depth, pair=[source['body'].split('/')[-1], target['body'].split('/')[-1]])
    return worst


def filter_hand(model, optimized, seed, hi, deadline, max_final):
    """Analytic full-vertex table prefilter, then unchanged full-model culls."""
    raw = []
    counts = dict(table=0, hand_in_object=0, object_in_hand=0, self_penetration=0, contact=0,
                  final_evaluated=0, final_cap_skipped=0, time_budget_skipped=0)
    inclusive = dict(table=0, hand_in_object=0, object_in_hand=0, self_penetration=0, contact=0)
    best_table = None
    obj_np = np.asarray(seed['wire_pose'])
    order = np.argsort(optimized['objective'])
    # Table clearance is independent of axial position. Use ALL source hull
    # vertices; no sampled/proxy table test may qualify a pose.
    for begin in range(0, len(order), 16):
        ids = order[begin:begin+16]
        with torch.no_grad():
            q = model.t(optimized['q'][ids])
            poses = model.fk(q, origin(model, len(ids)))
            vertices = torch.cat([kinematics.transform(poses[r['body']], model.t(r['vertices']))
                                  for r in model.spec['shapes']], 1).cpu().numpy()
        for ii, i in enumerate(ids):
            basis = optimized['basis'][i]
            if not np.isfinite(basis).all() or abs(np.linalg.det(basis)-1)>.001:
                counts['contact'] += 1; continue
            relative = vertices[ii]-optimized['center'][i]
            rolls = np.linspace(-math.pi, math.pi, 64, endpoint=False)
            # z @ R = cos(roll)*bisector + sin(roll)*third.
            ups = np.cos(rolls)[:, None]*basis[:, 1]+np.sin(rolls)[:, None]*basis[:, 2]
            clear = (relative@ups.T).min(0)+obj_np[2]-TABLE_Z
            best_idx = int(clear.argmax())
            if best_table is None or clear[best_idx]>best_table['maximum_clearance_m']:
                best_table = dict(seed_index=int(i), maximum_clearance_m=float(clear[best_idx]),
                                  contact_body_names=[optimized['bodies'][d][i].split('/')[-1] for d in range(2)])
            counts['table'] += int((clear<CLEARANCE).sum())*2*3
            valid = np.flatnonzero(clear>=CLEARANCE)
            # At most four well separated table-safe rolls per q.
            selected = []
            for ri in valid[np.argsort(clear[valid])[::-1]]:
                if all(min(abs(int(ri)-j), 64-abs(int(ri)-j))>=4 for j in selected):
                    selected.append(int(ri))
                if len(selected)==4: break
            for ri in selected:
                up = ups[ri]
                for flip in (1., -1.):
                    axis = basis[:, 0]*flip
                    # Rows are the old-frame directions of new world X/Y/Z.
                    rotation = np.stack((np.cross(axis, up), axis, up), 0)
                    quat = Rotation.from_matrix(rotation).as_quat()
                    for segment in ((.17, .21, .23) if hi==0 else (-.17, -.21, -.23)):
                        position = obj_np[:3]+[0., segment, 0.]-rotation@optimized['center'][i]
                        pose = np.r_[position, quat[3], quat[:3]]
                        # Untouched q preserves exact affine follower relation.
                        raw.append(dict(seed_index=int(i), q=optimized['q'][i].tolist(), pose=pose.tolist(),
                                        roll=float(rolls[ri]), segment_m=segment, prior_clearance_m=float(clear[ri]),
                                        contact_local_points=optimized['local'][i].tolist(),
                                        contact_body_names=[optimized['bodies'][d][i].split('/')[-1] for d in range(2)],
                                        contact_bodies=[optimized['bodies'][d][i] for d in range(2)],
                                        contact_local_normals=optimized['local_normals'][i].tolist(),
                                        source_triangles=optimized['triangles'][i].tolist(),
                                        source_barycentric=optimized['bary'][i].tolist(),
                                        proposal_loss=float(optimized['objective'][i])))
    # Round-robin over seeds so a candidate cap cannot erase body-pair coverage.
    raw.sort(key=lambda r: (r['roll'], r['segment_m'], r['proposal_loss']))
    buckets = {}
    for row in raw: buckets.setdefault(row['seed_index'], []).append(row)
    ordered = []
    for k in range(max([len(v) for v in buckets.values()] or [0])):
        ordered.extend(v[k] for v in buckets.values() if k<len(v))
    counts['final_cap_skipped'] = max(0, len(ordered)-max_final)
    ordered = ordered[:max_final]
    good, rejected_best = [], []
    for begin in range(0, len(ordered), 16):
        if time.monotonic()>deadline:
            counts['time_budget_skipped'] += len(ordered)-begin; break
        group = ordered[begin:begin+16]
        with torch.no_grad():
            q = model.t([r['q'] for r in group]); root = model.t([r['pose'] for r in group])
            poses = model.fk(q, root); sampled = model.sample_poses(poses)
            obj = model.t(seed['wire_pose'])[None].expand(len(group), -1)
            objects = object_surface_points(obj[:, :3], obj[:, 3:])
            clearance = sampled['collision_points'][:, :, 2].amin(-1)-TABLE_Z
            sdf = finite_cylinder_sdf(sampled['collision_points'], obj[:, :3], obj[:, 3:])
            hand_depth = (-sdf).clamp_min(0).amax(-1)
            inside = model.object_penetration_poses(poses, objects, fast=False)
            own = model.self_penetration_poses(poses, fast=False)
            points, normals = [], []
            for j, row in enumerate(group):
                points.append(torch.stack([kinematics.transform(poses[b][j:j+1], model.t([p]))[0, 0]
                                           for b, p in zip(row['contact_bodies'], row['contact_local_points'])]))
                # Outward normals from the selected source hull halfspace.
                ns = []
                for d, b in enumerate(row['contact_bodies']):
                    normal = row['contact_local_normals'][d]
                    ns.append(model.t(normal)@poses[b][j, :3, :3].T)
                normals.append(torch.stack(ns))
            points = torch.stack(points); normals = torch.stack(normals)
            contact_sdf, outward = _geometry(points, obj[:, :3], obj[:, 3:], RADIUS, .5, normals=True)
            distance = contact_sdf.abs()
            alignment = (1+(normals*outward).sum(-1)).clamp(0, 2).mean(-1)
            dots = (normals[:, 0]*normals[:, 1]).sum(-1)
        for j, row in enumerate(group):
            counts['final_evaluated'] += 1
            metrics = dict(minimum_clearance_m=float(clearance[j]), hand_in_object_m=float(hand_depth[j]),
                           object_in_hand_m=float(inside[j]), self_penetration_m=float(own[j]),
                           contact_distance_max_m=float(distance[j].max()), normal_alignment=float(alignment[j]))
            fail = dict(table=metrics['minimum_clearance_m']<CLEARANCE,
                        hand_in_object=metrics['hand_in_object_m']>DEPTH,
                        object_in_hand=metrics['object_in_hand_m']>DEPTH,
                        self_penetration=metrics['self_penetration_m']>DEPTH,
                        contact=metrics['contact_distance_max_m']>.0008 or metrics['normal_alignment']>.2)
            for key, value in fail.items(): inclusive[key] += int(value)
            badness = (max(0., CLEARANCE-metrics['minimum_clearance_m'])
                       + sum(max(0., metrics[key]-DEPTH) for key in ('hand_in_object_m', 'object_in_hand_m', 'self_penetration_m'))
                       + max(0., metrics['contact_distance_max_m']-.0008))
            if any(fail.values()):
                counts[next(key for key in fail if fail[key])] += 1
                rejected_best.append(dict(row, metrics=metrics, failed=[k for k in fail if fail[k]], badness_m=badness))
                rejected_best.sort(key=lambda r:r['badness_m']); rejected_best = rejected_best[:12]
                continue
            geom = dict(reach=float(distance[j].mean()/.008), penetration=float(hand_depth[j]/.001),
                        object_in_hand=float(inside[j]/.001), self=float(own[j]/.001), table=0.,
                        normal_alignment=float(alignment[j]), opposition=float((dots[j]+.8).clamp_min(0)),
                        normal_dot=float(dots[j]), patch_distance_m=distance[j].cpu().tolist(),
                        minimum_clearance_m=float(clearance[j]))
            good.append(dict(row, geometry=geom, geometry_feasible=True, qualified=False,
                             score=10*geom['reach']+geom['normal_alignment']+2*geom['opposition']))
        if begin % 128 == 0:
            emit('FILTER_PROGRESS', hand=model.side, evaluated=counts['final_evaluated'], safe=len(good),
                 rejected=counts, elapsed_in_filter_s=None)
        if len(good)>=48: break
    good.sort(key=lambda r:r['score'])
    # Retain differing q, body pairs, roll, and wrist positions.
    chosen = []
    for row in good:
        if any(np.max(np.abs(np.asarray(row['q'])-r['q']))<.035 and
               np.linalg.norm(np.asarray(row['pose'][:3])-r['pose'][:3])<.002 and
               abs(np.dot(row['pose'][3:], r['pose'][3:]))>.999 for r in chosen): continue
        chosen.append(row)
        if len(chosen)>=16: break
    for row in rejected_best[:4]:
        with torch.no_grad():
            poses = model.fk(model.t([row['q']]), model.t([row['pose']]))
            row['worst_self_pair'] = self_diagnostic(model, poses)
            objects = object_surface_points(model.t([seed['wire_pose'][:3]]), model.t([seed['wire_pose'][3:]]))
            per_shape = [(float(model._inside(objects, poses[s['body']], s, fast=False)[0]), s['body'].split('/')[-1])
                         for s in model.shapes]
            depth, name = max(per_shape)
            row['worst_object_hand_body'] = dict(body=name, depth_m=depth)
    diagnostic = dict(rejected_counts=counts, inclusive_rejected_counts=inclusive, best_table_case=best_table,
                      near_misses=rejected_best, aligned_table_safe_poses=len(raw),
                      contact_sampling='true hull triangles; outward halfspace normals; barycentric in simplex')
    emit('HAND_RESULT', hand=model.side, safe=len(good), retained=len(chosen), rejections=counts,
         best_table=best_table, near_misses=[dict(metrics=r['metrics'], bodies=r['contact_body_names'],
         worst_self_pair=r.get('worst_self_pair')) for r in rejected_best[:2]])
    return chosen, diagnostic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True,
                        help='New candidate JSON; existing evidence is never overwritten')
    parser.add_argument('--wall-seconds', type=float, default=235.)
    parser.add_argument('--seeds', type=int, default=256)
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--max-final', type=int, default=2048)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if not 1<=args.seeds<=256 or not 2<=args.steps<=400 or not 1<=args.max_final<=4096:
        raise ValueError('bounded seeds/steps/final candidates exceeded')
    if not 30<=args.wall_seconds<=240: raise ValueError('wall budget must be 30..240 seconds')
    torch.set_num_threads(6)
    start = time.monotonic(); root = args.root.resolve()
    out = args.output.resolve()
    if out.exists(): raise FileExistsError(out)
    seed = json.loads((root/'configs/dh116_training/grasp_initialization_below_axis_2mm_v1.json').read_text())
    if not np.allclose(seed['wire_pose'], [.43, 0., .308, 1., 0., 0., 0.]):
        raise ValueError('this bounded alignment assumes the configured world Y cylinder')
    paths = [root/'assets/fixed_dual_panda_dh116_v1/dependencies'/name for name in
             ('d134680ab619_DH116_left.usd', '3046ad2ed855_DH116_right.usd')]
    identity = implementation_identity(root)
    source = Path(__file__).resolve()
    source_sha = hashlib.sha256(source.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    identity[str(source)] = source_sha
    report = dict(qualified=False, geometry_only=True, surface_model_contract=dict(kinematics.CONTRACT),
                  implementation_hashes=identity, aux_source_lf_sha256=source_sha, aux_source_path=str(source),
                  source_hashes={s:hashlib.sha256(p.read_bytes()).hexdigest() for s,p in zip(('left', 'right'), paths)},
                  object_pose=seed['wire_pose'], seed_roots=seed['roots'],
                  mechanical_counts=dict(active=12, passive=10, finger_total=22),
                  initial_placement='free_global_palm_pose_seed; no hardware arm reach assertion',
                  ordinary_policy_bounds_unchanged=True, virtual_rail_m=[-.1, .1],
                  acceptance_thresholds_m=dict(table_minimum=CLEARANCE, hand_in_cylinder_maximum=DEPTH,
                                               object_in_hand_maximum=DEPTH, self_maximum=DEPTH),
                  geometry_feasible_count=0, candidates=[], hand_candidates=[], diagnostics=[],
                  requested_budget=dict(wall_seconds=args.wall_seconds, seeds_per_hand=args.seeds,
                                        max_steps_per_hand=args.steps, max_final_per_hand=args.max_final))
    out.parent.mkdir(parents=True, exist_ok=True)
    for hi, side in enumerate(('left', 'right')):
        model = ProductionSurfaceHandModel.from_usd(paths[hi], side, device=args.device)
        if len(model.shapes)!=17: raise ValueError('all seventeen source hulls required')
        catalog = surface_catalog(model)
        emit('MODEL', hand=side, shapes=len(model.shapes), self_pairs=len(model.self_pairs),
             contact_points=[len(c['bodies']) for c in catalog], source_sha256=report['source_hashes'][side])
        # Each hand gets an equal wall slot. Leave time for source hashing/output.
        hand_start = time.monotonic()
        hand_budget = max(15., (start+args.wall_seconds-5-hand_start)/(2-hi))
        optimized = optimize_hand(model, catalog, hi, start, hand_start+.57*hand_budget, args.seeds, args.steps)
        candidates, diagnostic = filter_hand(model, optimized, seed, hi, hand_start+hand_budget, args.max_final)
        diagnostic.update(iterations=optimized['iterations'], elapsed_s=time.monotonic()-hand_start)
        report['hand_candidates'].append(candidates); report['diagnostics'].append(diagnostic)
        report['elapsed_s'] = time.monotonic()-start
        out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        del model, catalog, optimized
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    combined = []
    for li, left in enumerate(report['hand_candidates'][0]):
        for ri, right in enumerate(report['hand_candidates'][1]):
            a, b = left['geometry'], right['geometry']
            geo = {k:max(a[k], b[k]) for k in ('penetration', 'object_in_hand', 'self', 'table')}
            geo.update({k:(a[k]+b[k])/2 for k in ('reach', 'normal_alignment', 'opposition')})
            geo.update(patch_distance_m=[a['patch_distance_m'], b['patch_distance_m']],
                       minimum_clearance_m=min(a['minimum_clearance_m'], b['minimum_clearance_m']))
            combined.append(dict(candidate_index=len(combined), qualified=False, geometry_feasible=True,
                                 geometry=geo, active_angles=left['q']+right['q'],
                                 wrist_pose_w=[left['pose'], right['pose']], score=left['score']+right['score'],
                                 seed_contact_points_local=[left['contact_local_points'], right['contact_local_points']],
                                 seed_contact_body_names=[left['contact_body_names'], right['contact_body_names']],
                                 intrinsic_hand_indices=[li, ri]))
    # Hands are at opposite rod ends. No untested cross-hand overlap is accepted:
    # demand disjoint source hull AABBs with a 0.3mm separation in world Y.
    if combined:
        models = [ProductionSurfaceHandModel.from_usd(p,s,device='cpu') for p,s in zip(paths,('left','right'))]
        safe = []
        with torch.no_grad():
            for row in combined:
                bounds = []
                for hi, m in enumerate(models):
                    sample = m.sample(m.t([row['active_angles'][6*hi:6*hi+6]]), m.t([row['wrist_pose_w'][hi]]))
                    ys = sample['collision_points'][0, :, 1]
                    bounds.append([float(ys.min()), float(ys.max())])
                if max(bounds[0][0]-bounds[1][1], bounds[1][0]-bounds[0][1])>=.0003:
                    row['cross_hand_y_bounds_m'] = bounds; safe.append(row)
        report['cross_hand_aabb_rejected'] = len(combined)-len(safe)
        combined = safe
    # Verify immutable implementations did not change during a concurrent run.
    actual_identity = implementation_identity(root)
    report['implementation_identity_stable'] = all(identity[k]==v for k,v in actual_identity.items())
    if not report['implementation_identity_stable']:
        report['identity_changed_during_search'] = actual_identity
        report['unqualified_geometry_proposals'] = combined
        combined = []
    report.update(candidates=rank_diverse(combined, maximum=12), geometry_feasible_count=len(combined),
                  hand_candidate_counts=[len(g) for g in report['hand_candidates']], elapsed_s=time.monotonic()-start,
                  gpu_job_finished=True, measured_contact=False, physical_validation_required=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    emit('PHALANX_RESULT', output=str(out), hand_candidate_counts=report['hand_candidate_counts'],
         geometry_feasible_count=len(combined), candidates=len(report['candidates']),
         elapsed_s=report['elapsed_s'], aux_source_lf_sha256=source_sha, gpu_job_finished=True)


if __name__=='__main__':
    main()
