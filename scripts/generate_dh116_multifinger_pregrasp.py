"""CPU-only, bounded, geometry-only multifinger pregrasp proposal (auxiliary).

Run via /root/hand/scripts/python.sh. Does not import a simulator or edit the
source repository/configs. Units: poses/distances meters, penetration fields mm.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[_key] = '4'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
if hasattr(os, 'sched_getaffinity'):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import signal
import time

START = time.monotonic()
import numpy as np
import torch
from go2_pvcnn.control.dh116.tabletop_surface_kinematics import SurfaceHandModel, transform, _surface
from go2_pvcnn.control.dh116.tabletop_free_wrist_grasp import compose_rotation

DIGITS = ('thumb', 'index', 'middle', 'ring', 'little')
DISTALS = ('finger14_link', 'finger23_link', 'finger33_link', 'finger43_link', 'finger53_link')
CENTER = (.43, 0., .308)
RADIUS, HALF_LENGTH, TABLE = .008, .25, .30
CONTACT_M, NORMAL_COS, OPPOSITION_COS = .0008, -.5, -.75
DEPTH_MM, CLEARANCE_M = .3, .00029


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cylinder(points):
    """Exact finite Y cylinder SDF plus analytic normal (identity object pose)."""
    rel = points - points.new_tensor(CENTER)
    radial = rel * points.new_tensor((1., 0., 1.))
    rho = torch.linalg.vector_norm(radial, dim=-1)
    side = rho - RADIUS
    cap = rel[..., 1].abs() - HALF_LENGTH
    d = torch.stack((side, cap), -1)
    pos = d.clamp_min(0.)
    outside = torch.linalg.vector_norm(pos, dim=-1)
    sdf = outside + d.amax(-1).clamp_max(0.)
    side_n = radial / rho.clamp_min(1e-12)[..., None]
    cap_n = torch.stack((rel[..., 1]*0, rel[..., 1].sign(), rel[..., 1]*0), -1)
    normal = torch.where((side >= cap)[..., None], side_n, cap_n)
    outer_n = (pos[..., :1]*side_n + pos[..., 1:]*cap_n) / outside.clamp_min(1e-12)[..., None]
    return sdf, torch.where((outside > 0)[..., None], outer_n, normal)


def rod_points(model, full):
    na, ny = (96, 513) if full else (24, 129)
    a = torch.arange(na, dtype=model.dtype) * (2*math.pi/na)
    y = torch.linspace(-HALF_LENGTH, HALF_LENGTH, ny, dtype=model.dtype)
    aa, yy = torch.meshgrid(a, y, indexing='ij')
    side = torch.stack((RADIUS*aa.cos(), yy, RADIUS*aa.sin()), -1).flatten(0, 1)
    # Include cap disks and centerline; no missing rod-end/interior samples.
    rr, tt = torch.meshgrid(torch.linspace(0, RADIUS, 9, dtype=model.dtype), a, indexing='ij')
    cap = torch.stack((rr*tt.cos(), torch.zeros_like(rr), rr*tt.sin()), -1).flatten(0, 1)
    caps = [cap + cap.new_tensor((0, s*HALF_LENGTH, 0)) for s in (-1, 1)]
    axis = torch.stack((y*0, y, y*0), -1)
    return torch.cat((side, *caps, axis), 0) + model.t(CENTER)


def sample_model(model, full):
    """Change only an in-memory auxiliary sampling density; retain source planes."""
    for i, (s, spec) in enumerate(zip(model.shapes, model.spec['shapes'])):
        v = np.asarray(spec['vertices'])
        p, _, _ = _surface(v, count=512 if full else 32, seed=911+i)
        s['points'] = model.t(np.concatenate((v, p)))
        s['probe_points'] = s['points']


def distal_samples(model, full):
    result = []
    for suffix in DISTALS:
        rows = [s for s in model.spec['shapes'] if s['body'].endswith('/'+suffix)]
        if not rows:
            raise ValueError('missing source distal collision body: '+suffix)
        ps, ns = [], []
        for i, row in enumerate(rows):
            p, n, _ = _surface(np.asarray(row['vertices']), count=1024 if full else 128, seed=307+i)
            ps.append(p); ns.append(n)
        result.append(dict(body=rows[0]['body'], points=model.t(np.concatenate(ps)),
                           normals=model.t(np.concatenate(ns))))
    return result


def decode(model, x, roots):
    z = x.clamp(-1, 1)
    q = model.lower + (z[:, :6]+1)/2*(model.upper-model.lower)
    offsets = .020*z[:, 6:9]
    # Rotation vector norm bounded by .5rad as well as per-coordinate .5rad.
    angles = .5*z[:, 9:]
    angles = angles / torch.maximum(angles.norm(dim=-1, keepdim=True)/.5, torch.ones_like(angles[:, :1]))
    pose = torch.cat((roots[:, :3]+offsets, compose_rotation(roots[:, 3:], angles)), -1)
    return q, pose, offsets, angles


def evaluate(model, q, root, patches, objects, full=False):
    poses = model.fk(q, root)
    all_points = torch.cat([transform(poses[s['body']], s['points']) for s in model.shapes], 1)
    sdf, _ = cylinder(all_points)
    penetration = (-sdf).clamp_min(0).amax(-1)*1000
    clearance = all_points[..., 2].amin(-1)-TABLE
    table = ((TABLE+.0003-all_points[..., 2]).clamp_min(0)*1000).amax(-1)
    obj = objects[None].expand(len(q), -1, -1)
    inside = model.object_penetration_poses(poses, obj, fast=not full)*1000
    self_depth = model.self_penetration_poses(poses, fast=not full)*1000
    points = torch.stack([transform(poses[p['body']], p['points']) for p in patches], 1)
    normals = torch.stack([p['normals']@poses[p['body']][:, :3, :3].transpose(-1, -2) for p in patches], 1)
    dist, outward = cylinder(points)
    dot = (normals*outward).sum(-1).clamp(-1, 1)
    # Closest *aligned* surface region on each of five distal collision hulls.
    sample_cost = ((dist-.0004)/.008).square() + .12*(1+dot).square()
    _, ids = sample_cost.topk(3, dim=-1, largest=False)
    near = dist.gather(-1, ids).abs().mean(-1)
    ndot = dot.gather(-1, ids).mean(-1)
    ix = ids[..., None].expand(-1, -1, -1, 3)
    contact = points.gather(2, ix).mean(2)
    hand_normal = normals.gather(2, ix).mean(2)
    hand_normal = hand_normal/hand_normal.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    radial = (contact-contact.new_tensor(CENTER))*contact.new_tensor((1., 0., 1.))
    direction = radial/radial.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    # Enumerate ALL ten three-digit subsets, without preferring thumb or index.
    choices = []
    for tri in itertools.combinations(range(5), 3):
        chosen = list(tri)
        d = near[:, chosen]
        n = ndot[:, chosen]
        radial_pairs = torch.stack([(direction[:, a]*direction[:, b]).sum(-1) for a,b in itertools.combinations(tri, 2)], -1)
        normal_pairs = torch.stack([(hand_normal[:, a]*hand_normal[:, b]).sum(-1) for a,b in itertools.combinations(tri, 2)], -1)
        pair_cost = ((radial_pairs+.85).clamp_min(0).square() + (normal_pairs+.65).clamp_min(0).square()).amin(-1)
        # Balanced gaps + distributed axial contacts are geometry proxies only.
        cost = 10*(d/.008).square().mean(-1) + 2*(d/.008).square().amax(-1)
        cost = cost + .6*(1+n).square().mean(-1) + 2*pair_cost + .5*(d.std(-1)/.008).square()
        choices.append(cost)
    grasp, subset = torch.stack(choices, -1).min(-1)
    # Margins target stricter than export gates, including full-rod-inside-hand.
    collision = 35*((penetration-.08).clamp_min(0).square() +
                    (inside-.08).clamp_min(0).square() + (self_depth-.08).clamp_min(0).square())
    score = grasp + collision + 60*table.square()
    return dict(score=score, grasp_score=grasp, penetration=penetration, object_in_hand=inside,
                self=self_depth, table=table, minimum_clearance_m=clearance,
                patch_distance_m=near, patch_normal_dot=ndot, contact_points=contact,
                hand_normals=hand_normal, radial_directions=direction, selected_subset=subset,
                poses=poses, collision_points=all_points)


def contact_check(g):
    """Per-hand hard gate: >=3 distinct aligned distals plus a near-opposed pair."""
    eligible = [i for i in range(5) if g['patch_distance_m'][i] <= CONTACT_M and g['patch_normal_dot'][i] <= NORMAL_COS]
    pairs = []
    for a,b in itertools.combinations(eligible, 2):
        radial_dot = float(np.dot(g['radial_directions'][a], g['radial_directions'][b]))
        normal_dot = float(np.dot(g['hand_normals'][a], g['hand_normals'][b]))
        if radial_dot <= OPPOSITION_COS and normal_dot <= -.5:
            pairs.append(dict(digits=[DIGITS[a], DIGITS[b]], radial_dot=radial_dot, normal_dot=normal_dot))
    reasons = []
    for k in ('penetration', 'object_in_hand', 'self'):
        if g[k] > DEPTH_MM: reasons.append(k+'_exceeds_0.3mm')
    if g['minimum_clearance_m'] < CLEARANCE_M: reasons.append('table_clearance_below_0.29mm')
    if len(eligible) < 3: reasons.append('fewer_than_3_aligned_close_distals')
    if not pairs: reasons.append('no_near_opposed_contact_pair')
    return not reasons, [DIGITS[i] for i in eligible], pairs, reasons


def as_hand_row(g, j, q, root, x, seed_index, offsets, angles):
    names = ('penetration', 'object_in_hand', 'self', 'table', 'minimum_clearance_m',
             'patch_distance_m', 'patch_normal_dot', 'contact_points', 'hand_normals', 'radial_directions')
    geo = {k:g[k][j].detach().cpu().tolist() for k in names}
    feasible, contacts, pairs, reasons = contact_check(geo)
    geo.update(distal_order=list(DIGITS), contact_digits=contacts, opposition_pairs=pairs,
               collision_units='mm', distance_units='m')
    return dict(start_index=j, seed_index=int(seed_index), score=float(g['score'][j]), geometry=geo,
                geometry_feasible=feasible, rejection_reasons=reasons, qualified=False,
                active_angles=q[j].tolist(), wrist_pose_w=root[j].tolist(), normalized_variables=x[j].tolist(),
                wrist_translation_delta_m=offsets[j].tolist(), wrist_local_XYZ_angles_rad=angles[j].tolist())


def combine(a, b, index):
    ga, gb = a['geometry'], b['geometry']
    geo = {k:max(ga[k], gb[k]) for k in ('penetration', 'object_in_hand', 'self', 'table')}
    geo.update(minimum_clearance_m=min(ga['minimum_clearance_m'], gb['minimum_clearance_m']),
               patch_distance_m=[ga['patch_distance_m'], gb['patch_distance_m']],
               normal_alignment=float(np.mean([1+x for g in (ga, gb) for x in g['patch_normal_dot']])),
               reach=float(np.mean([x for g in (ga, gb) for x in g['patch_distance_m']]))/.008,
               per_hand=[ga, gb], collision_units='mm', distance_units='m')
    return dict(candidate_index=index, active_angles=a['active_angles']+b['active_angles'],
                wrist_pose_w=[a['wrist_pose_w'], b['wrist_pose_w']], geometry=geo,
                score=a['score']+b['score'], geometry_feasible=a['geometry_feasible'] and b['geometry_feasible'],
                qualified=False, hand_diagnostics=[a, b])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/root/hand'))
    p.add_argument('--output', type=Path, default=Path('/root/hand/runs/multifinger_pregrasp_20261030_a/candidates.json'))
    p.add_argument('--starts', type=int, default=16)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--wall-seconds', type=float, default=240.)
    p.add_argument('--self-test', action='store_true')
    o = p.parse_args()
    if o.self_test:
        return self_test()
    if not 8 <= o.starts <= 16 or not 1 <= o.steps <= 200 or not 10 <= o.wall_seconds <= 240:
        raise ValueError('requires 8..16starts, <=200steps, <=240sec')
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    root, out = o.root, o.output
    if out.exists(): raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = dict(contract=dict(version='multifinger_bounded_cpu_geometry_v1', geometry_only=True,
                     qualified=False, active_per_hand=6, mimic='existing_exact_affine_mimic',
                     distal_order=list(DIGITS), any_three_of_five=True,
                     wrist_translation_per_axis_m=.020, wrist_local_rotation_vector_norm_rad=.5,
                     contact_distance_m=CONTACT_M, contact_hand_object_normal_dot_max=NORMAL_COS,
                     contact_opposition_radial_dot_max=OPPOSITION_COS,
                     penetration_limit_mm=DEPTH_MM, minimum_clearance_m=CLEARANCE_M,
                     geometry_is_not_measured_contact=True, anti_slip_or_load_sharing_not_established=True),
                  candidates=[], geometry_feasible_count=0, qualified=False, geometry_only=True,
                  status='initializing', object_pose=[*CENTER, 1., 0., 0., 0.], rod_radius_m=RADIUS,
                  rod_length_m=2*HALF_LENGTH, table_height_m=TABLE, threads=4,
                  cpu_affinity=sorted(os.sched_getaffinity(0)), starts_per_hand=o.starts,
                  maximum_steps=o.steps, wall_budget_s=o.wall_seconds, iterations=0,
                  script_sha256=sha(__file__))

    def save():
        report['elapsed_s'] = time.monotonic()-START
        temp = out.with_suffix('.tmp')
        temp.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temp.replace(out)

    class Deadline(Exception): pass
    def alarm(*_): raise Deadline('hard wall budget reached')
    signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, max(1., o.wall_seconds-(time.monotonic()-START)-2.))
    try:
        sources = [root/'assets/fixed_dual_panda_dh116_v1/dependencies'/f for f in
                   ('d134680ab619_DH116_left.usd', '3046ad2ed855_DH116_right.usd')]
        configs = [root/'configs/dh116_training'/f for f in ('right_pinch_pregrasp_v1.json', 'grip_lift_pregrasp_v1.json')]
        protected = list((root/'src/go2_pvcnn/control/dh116').glob('*.py')) + configs + sources
        before = {str(f.relative_to(root)):sha(f) for f in protected}
        report.update(source_hashes=dict(zip(('left', 'right'), map(sha, sources))),
                      source_before_hashes=before, seed_configs=[str(x) for x in configs])
        save()
        seeds = []
        for config in configs:
            seeds += json.loads(config.read_text())['candidates']
        models = [SurfaceHandModel.from_usd(f, side, device='cpu') for f,side in zip(sources, ('left', 'right'))]
        if any(len(m.shapes)!=17 for m in models): raise ValueError('all17 source collision hulls required')
        report['source_hulls_per_hand'] = [len(m.shapes) for m in models]
        # Half the starts from each named config, varying all six active joints.
        indices = [0 if j%2==0 else 1+(j//2)%(len(seeds)-1) for j in range(o.starts)]
        roots = torch.tensor([seeds[j]['wrist_pose_w'] for j in indices], dtype=torch.float64)
        seed_q = torch.tensor([seeds[j]['active_angles'] for j in indices], dtype=torch.float64).reshape(o.starts,2,6)
        x0 = torch.zeros(o.starts,2,12,dtype=torch.float64)
        for h,m in enumerate(models): x0[:,h,:6] = 2*(seed_q[:,h]-m.lower)/(m.upper-m.lower)-1
        rng = torch.Generator().manual_seed(301030)
        noise = torch.randn(x0.shape, dtype=x0.dtype, generator=rng)
        x0[:,:,:6] += .24*noise[:,:,:6]
        x0[:,:,6:9] = .12*noise[:,:,6:9]
        x0[:,:,9:] = .22*noise[:,:,9:]
        # Preserve both original seeds; other digits are deliberately explored.
        for j in (0,1):
            for h,m in enumerate(models): x0[j,h,:6] = 2*(seed_q[j,h]-m.lower)/(m.upper-m.lower)-1
            x0[j,:,6:] = 0
        x = torch.nn.Parameter(x0.clamp(-1,1))
        best_x = x.detach().clone()
        best = torch.full((o.starts,2), float('inf'), dtype=x.dtype)
        optimizer = torch.optim.Adam([x], lr=.020)
        for m in models: sample_model(m,False)
        patches = [distal_samples(m,False) for m in models]
        objects = [rod_points(m,False) for m in models]
        # Reserve at least40seconds for full samples/original convex plane checks.
        search_deadline = START + o.wall_seconds-45
        print('MULTIFINGER_START',json.dumps(dict(starts=o.starts,steps=o.steps,threads=4,
              setup_s=time.monotonic()-START, self_pairs=[len(m.self_pairs) for m in models])),flush=True)
        for step in range(o.steps):
            if time.monotonic() >= search_deadline: break
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for h,m in enumerate(models):
                q, pose, _, _ = decode(m,x[:,h],roots[:,h])
                g = evaluate(m,q,pose,patches[h],objects[h],False)
                loss = g['score']
                better = loss.detach()<best[:,h]
                best[:,h] = torch.minimum(best[:,h],loss.detach())
                best_x[:,h] = torch.where(better[:,None],x[:,h].detach(),best_x[:,h])
                losses.append(loss)
            torch.stack(losses).sum().backward()
            if not torch.isfinite(x.grad).all(): raise ValueError('nonfinite optimization gradient')
            torch.nn.utils.clip_grad_norm_([x],100.)
            optimizer.param_groups[0]['lr'] = .020*(.1+.9*(1-step/o.steps)**2)
            optimizer.step()
            with torch.no_grad(): x.clamp_(-1,1)
            report['iterations'] = step+1
            if step%20==0:
                print('MULTIFINGER_SEARCH',json.dumps(dict(step=step+1,best=best.amin(0).tolist(),
                      elapsed_s=time.monotonic()-START)),flush=True)
        report['status']='fullsample_convexplane_verification'
        report['search_elapsed_s']=time.monotonic()-START
        save()
        hand_rows = []
        for h,m in enumerate(models):
            sample_model(m,True)
            patch = distal_samples(m,True)
            obj = rod_points(m,True)
            with torch.no_grad():
                q,pose,offset,angles = decode(m,best_x[:,h],roots[:,h])
                g = evaluate(m,q,pose,patch,obj,True)
            rows = [as_hand_row(g,j,q,pose,best_x[:,h],indices[j],offset,angles) for j in range(o.starts)]
            hand_rows.append(rows)
            report['hand_diagnostics']=hand_rows
            print('MULTIFINGER_VERIFIED',json.dumps(dict(hand=h,feasible=sum(r['geometry_feasible'] for r in rows),
                  best=min(rows,key=lambda r:r['score']),elapsed_s=time.monotonic()-START)),flush=True)
            save()
        combined = []
        for i,a in enumerate(hand_rows[0]):
            for j,b in enumerate(hand_rows[1]):
                if a['geometry_feasible'] and b['geometry_feasible']:
                    combined.append(combine(a,b,i*o.starts+j))
        # Full source inter-hand collision check for retained combinations.
        verified = []
        for row in sorted(combined,key=lambda r:r['score']):
            if len(verified)>=4: break
            points, poses = [], []
            with torch.no_grad():
                for h,m in enumerate(models):
                    s=m.sample(m.t(row['active_angles'][h*6:(h+1)*6])[None],m.t(row['wrist_pose_w'][h])[None])
                    points.append(s['collision_points']);poses.append(s['poses'])
                inter = max(float(models[h].object_penetration_poses(poses[h],points[1-h],fast=False)[0])*1000 for h in range(2))
            row['geometry']['inter_hand_penetration_mm']=inter
            row['geometry']['self']=max(row['geometry']['self'],inter)
            if inter>DEPTH_MM: continue
            duplicate=any(np.max(np.abs(np.array(row['active_angles'])-v['active_angles']))<.035 and
                          np.max(np.linalg.norm(np.array(row['wrist_pose_w'])[:,:3]-np.array(v['wrist_pose_w'])[:,:3],axis=1))<.001 for v in verified)
            if not duplicate: verified.append(row)
        report.update(candidates=verified, geometry_feasible_count=len(verified),
                      feasible_hand_counts=[sum(r['geometry_feasible'] for r in rows) for rows in hand_rows],
                      feasible_pair_count_before_interhand_check=len(combined),
                      verification=dict(full_original_convex_planes=True, hulls_per_hand=17,
                          hand_samples='all_source_vertices_plus512_area_samples_per_hull',
                          distal_samples_per_finger=1024, rod_surface_grid=[96,513],
                          rod_cap_rings=9, rod_centerline_included=True,
                          self_samples='all_source_vertices_plus512_per_hull_bidirectional',
                          self_exclusions='same_fixed_cluster_and_direct_neighbor_clusters_only',
                          continuous_collision_proof=False), status='complete')
        if not verified:
            a,b=[min(rows,key=lambda r:(len(r['rejection_reasons']),r['score'])) for rows in hand_rows]
            report['best_infeasible']=combine(a,b,0)
            report['best_infeasible']['geometry_feasible']=False
            report['best_infeasible']['diagnostic_only']=True
            report['status']='no_feasible_multifinger_candidate'
        after={str(f.relative_to(root)):sha(f) for f in protected}
        report['protected_source_hashes_unchanged']=before==after
        report['changed_protected_paths']=[k for k in before if before[k]!=after[k]]
        if before!=after:
            report['candidates']=[]
            report['status']='source_identity_changed_during_run'
        save()
    except Deadline as e:
        report.update(status='wall_budget_exhausted',error=str(e))
        # Never export an incompletely verified pose.
        report['candidates']=[];report['geometry_feasible_count']=0
        save()
    except Exception as e:
        report.update(status='error',error=repr(e),candidates=[],geometry_feasible_count=0)
        save()
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
    print('MULTIFINGER_RESULT',json.dumps(dict(output=str(out),status=report['status'],
          candidates=len(report['candidates']),iterations=report['iterations'],elapsed_s=report['elapsed_s'])),flush=True)


def self_test():
    # Geometry gate contract, especially the forbidden thumb/index-only fallback.
    g=dict(penetration=.3,object_in_hand=.3,self=.3,minimum_clearance_m=.00029,
           patch_distance_m=[.0001,.0001,.1,.1,.1],patch_normal_dot=[-1.]*5,
           radial_directions=[[1,0,0],[-1,0,0],[1,0,0],[1,0,0],[1,0,0]],
           hand_normals=[[-1,0,0],[1,0,0],[-1,0,0],[-1,0,0],[-1,0,0]])
    assert not contact_check(g)[0], 'two fingers must be rejected'
    g['patch_distance_m']=[.1,.1,.0001,.0001,.0001]
    g['radial_directions'][3]=[-1,0,0];g['hand_normals'][3]=[1,0,0]
    assert contact_check(g)[0], 'middle/ring/little alone must be allowed'
    g['penetration']=.300001
    assert not contact_check(g)[0], '0.3mm is a hard limit'
    g['penetration']=.3;g['minimum_clearance_m']=.00028999
    assert not contact_check(g)[0], 'clearance is a hard limit'
    p=torch.tensor([[[.438,0,.308],[.43,0,.308],[.43,.26,.308],[.43,0,.3]]],dtype=torch.float64)
    d,_=cylinder(p)
    assert torch.allclose(d,torch.tensor([[0.,-.008,.01,0.]],dtype=torch.float64),atol=1e-12)
    print('SELF_TEST_PASS: arbitrary3digits, no2digitfallback, strictmm/mgates, finiteYcylinder')


if __name__=='__main__':
    main()
