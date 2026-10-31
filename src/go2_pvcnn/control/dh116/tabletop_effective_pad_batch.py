"""Optional read-only native sampler, equivalent to PadContactSampler.sample().

Call only for contacts=True; a None result requests the original implementation.
Uniform buffer capacities are required. No native buffer or pose is modified.
"""
import torch
from torch.nn.functional import pad

from .tabletop_effective_pad_runtime import DISTAL_LINKS, _rotation

_CONTACT_CHUNK = 4096


def _device_tensor(values, device, dtype=torch.long):
    # torch.tensor(..., device='cuda') waits for each tiny metadata copy.
    # Keep the copy ordered on the current stream without a host-side wait.
    return torch.tensor(values, dtype=dtype).to(device, non_blocking=True)


def _owners_batched(count, start, capacity):
    """The runtime interval algorithm with a leading sensor dimension."""
    invalid = (count < 0) | (start < 0) | (start > capacity) | (count > capacity-start)
    eligible = ~invalid & (count > 0)
    ordered, perm = torch.sort(torch.where(eligible, start, capacity+1), dim=1)
    ends = ordered + torch.where(eligible.gather(1, perm), count.gather(1, perm), 0)
    running_end, furthest = torch.cummax(ends, dim=1)
    overlap = (ordered[:, 1:] < running_end[:, :-1]) & eligible.gather(1, perm[:, 1:])
    conflicts = torch.zeros_like(count)
    conflicts.scatter_add_(1, perm[:, 1:], overlap.long())
    conflicts.scatter_add_(1, perm.gather(1, furthest[:, :-1]), overlap.long())
    invalid = invalid | (conflicts > 0)
    slots = torch.arange(capacity, device=count.device).expand(len(count), -1).contiguous()
    index = (torch.searchsorted(ordered, slots, right=True)-1).clamp_min(0)
    env = perm.gather(1, index)
    starts = start.gather(1, env)
    active = (eligible.gather(1, env) & ~invalid.gather(1, env) &
              (slots >= starts) & (slots < starts + count.gather(1, env)))
    return env, active, invalid


def _patch_batched(point, normal, centers, tangent_u, tangent_v, half_widths,
                   planes, plane_counts, tolerances, cosines):
    """Ten ragged CAD envelopes; retain first-plane tie behavior of _on_patch."""
    rel = point-centers[:, None]
    uv = torch.stack(((rel*tangent_u[:, None]).sum(-1),
                      (rel*tangent_v[:, None]).sum(-1)), -1)
    region = ((uv/half_widths[:, None])**2).sum(-1) <= 1.
    best = torch.full(point.shape[:-1], -torch.inf, device=point.device, dtype=point.dtype)
    direction = torch.zeros_like(point)
    for begin in range(0, planes.shape[1], 32):
        group = planes[:, begin:begin+32]
        signed = point @ group[..., :3].transpose(1, 2) + group[:, None, :, 3]
        ids = torch.arange(begin, begin+group.shape[1], device=point.device)
        signed = signed.masked_fill(ids[None, None] >= plane_counts[:, None, None], -torch.inf)
        distance, index = signed.max(-1)
        normals = group[..., :3].gather(1, index[..., None].expand(-1, -1, 3))
        direction = torch.where((distance > best)[..., None], normals, direction)
        best = torch.maximum(best, distance)
    cosine = (normal*direction).sum(-1)/normal.norm(dim=-1).clamp_min(1e-30)
    return region & (best.abs() <= tolerances[:, None]) & (cosine <= -cosines[:, None])


@torch.no_grad()
def sample_native_batched(sampler):
    """Return the runtime output dict and set last_signed_force_w, or return None.

    Layout checks inspect metadata only (no per-env loops or CUDA scalar reads).
    Contacts=False remains the caller's responsibility in the original sampler.
    Work involving CAD planes/rotations is bounded to 4096 contacts per sensor.
    """
    s = sampler
    w, n = s.world, s.n
    if n < 1 or len(s.bindings) != 34 or len(w.hands) != 2:
        return None
    raw = getattr(s,'native_contact_data',None)
    if raw is None:
        raw = [sensor.contact_physx_view.get_contact_data(dt=w.dt) for *_, sensor in s.bindings]
    if len(raw)!=len(s.bindings):
        return None
    if any(len(values) != 6 for values in raw):
        return None
    capacity = raw[0][0].numel()
    for values in raw:
        f, p, normal, sep, count, start = values
        if (f.numel() != capacity or sep.numel() != capacity or
                p.numel() != capacity*3 or normal.numel() != capacity*3 or
                count.numel() != n or start.numel() != n or
                any(v.device != f.device for v in values) or
                f.device != raw[0][0].device):
            return None
    distal = [i for i, (_, _, d, _) in enumerate(s.bindings) if d is not None]
    if (len(distal) != 10 or
            {(h, d) for h, _, d, _ in s.bindings if d is not None} !=
            {(h, d) for h in range(2) for d in range(5)} or
            any(h not in (0, 1) for h, _, _, _ in s.bindings)):
        return None
    poses_raw = [hand.data.body_link_pose_w for hand in w.hands]
    if (poses_raw[0].shape != poses_raw[1].shape or poses_raw[0].ndim != 3 or
            poses_raw[0].shape[0] != n or poses_raw[0].shape[-1] != 7):
        return None
    device = raw[0][0].device
    dtype = torch.float64
    poses = torch.stack(poses_raw).double()
    bad = ((~torch.isfinite(poses).all((-1, -2))) |
           ((poses[..., 3:].norm(dim=-1)-1).abs() > 1e-3).any(-1)).any(0)
    rotations = _rotation(poses)
    positions = torch.nan_to_num(poses[..., :3], nan=0., posinf=0., neginf=0.)
    # References always use the actual named distal body, independently of contact slots.
    reference_bodies = _device_tensor([[hand.body_names.index(link) for link in DISTAL_LINKS]
                                      for hand in w.hands], device)
    hands = torch.arange(2, device=device)[:, None]
    ref_r = rotations[hands, :, reference_bodies].permute(2, 0, 1, 3, 4)
    ref_p = positions[hands, :, reference_bodies].permute(2, 0, 1, 3)
    reference_centers = torch.stack([s.rows[h, d]['center_local_m']
                                     for h in range(2) for d in range(5)]).reshape(2, 5, 3)
    references = (ref_r @ reference_centers[..., None]).squeeze(-1) + ref_p

    count = torch.stack([v[4].reshape(n) for v in raw]).long()
    start = torch.stack([v[5].reshape(n) for v in raw]).long()
    env, active, invalid = _owners_batched(count, start, capacity)
    bad |= invalid.any(0)
    f = torch.stack([v[0].reshape(-1) for v in raw]).double()
    p = torch.stack([v[1].reshape(-1, 3) for v in raw]).double()
    normal = torch.stack([v[2].reshape(-1, 3) for v in raw]).double()
    sep = torch.stack([v[3].reshape(-1) for v in raw]).double()
    finite = (torch.isfinite(f) & torch.isfinite(p).all(-1) &
              torch.isfinite(normal).all(-1) & torch.isfinite(sep))
    sane = finite & ((normal.norm(dim=-1)-1).abs() <= .05)
    faults = torch.zeros(n, device=device, dtype=torch.long)
    faults.scatter_add_(0, env.reshape(-1), (active & ~sane).long().reshape(-1))
    bad |= faults > 0
    hi = _device_tensor([h for h, _, _, _ in s.bindings], device)
    owner_hand = (env*2 + hi[:, None]).reshape(-1)
    safe_f = torch.where(active & finite, f, 0.)
    total = torch.zeros(n*2, device=device, dtype=dtype)
    signed_force = torch.zeros(n*2, 3, device=device, dtype=dtype)
    total.scatter_add_(0, owner_hand, safe_f.abs().reshape(-1))
    signed_force.scatter_add_(0, owner_hand[:, None].expand(-1, 3),
                              (torch.nan_to_num(normal)*safe_f[..., None]).reshape(-1, 3))
    total = total.reshape(n, 2)
    signed_force = signed_force.reshape(n, 2, 3)

    loads = torch.zeros(n*10, device=device, dtype=dtype)
    vectors = torch.zeros(n*10, 3, device=device, dtype=dtype)
    points = torch.zeros_like(vectors)
    depth = torch.zeros_like(loads)
    body_ids = torch.full((n, 2, 5), -1, device=device, dtype=torch.long)
    if capacity:
        ds = _device_tensor(distal, device)
        dh = hi[ds]
        db = _device_tensor([s.bindings[i][1] for i in distal], device)
        dd = _device_tensor([s.bindings[i][2] for i in distal], device)
        rows = [s.rows[s.bindings[i][0], s.bindings[i][2]] for i in distal]
        centers, u, v, widths = (torch.stack([row[key] for row in rows]) for key in
                                 ('center_local_m', 'tangent_u', 'tangent_v', 'half_widths_m'))
        counts = [len(row['planes_local']) for row in rows]
        planes = torch.stack([pad(row['planes_local'], (0, 0, 0, max(counts)-len(row['planes_local'])))
                              for row in rows])
        plane_counts = _device_tensor(counts, device)
        tolerances = _device_tensor([row['surface_tolerance_m'] for row in rows], device, dtype)
        cosines = _device_tensor([row['normal_cosine_min'] for row in rows], device, dtype)
        de, df, dp, dn, dsep, da, dsane = (value[ds] for value in (env, f, p, normal, sep, active, sane))
        for begin in range(0, capacity, _CONTACT_CHUNK):
            end = begin+_CONTACT_CHUNK
            e, cf = de[:, begin:end], df[:, begin:end]
            cp = torch.nan_to_num(dp[:, begin:end], nan=0., posinf=0., neginf=0.)
            cn = torch.nan_to_num(dn[:, begin:end], nan=0., posinf=0., neginf=0.)
            r = rotations[dh[:, None], e, db[:, None]]
            pos = positions[dh[:, None], e, db[:, None]]
            local_p = torch.einsum('dki,dkij->dkj', cp-pos, r)
            local_n = torch.einsum('dki,dkij->dkj', cn, r)
            accept = (da[:, begin:end] & dsane[:, begin:end] & (cf > 0) &
                      _patch_batched(local_p, local_n, centers, u, v, widths,
                                     planes, plane_counts, tolerances, cosines))
            accepted_f = torch.where(accept, cf, 0.)
            owner = (e*10 + dh[:, None]*5 + dd[:, None]).reshape(-1)
            loads.scatter_add_(0, owner, accepted_f.reshape(-1))
            expanded = owner[:, None].expand(-1, 3)
            vectors.scatter_add_(0, expanded, (cn*accepted_f[..., None]).reshape(-1, 3))
            points.scatter_add_(0, expanded, (cp*accepted_f[..., None]).reshape(-1, 3))
            accepted_depth = torch.where(accept, (-dsep[:, begin:end]).clamp_min(0), 0.)
            depth.scatter_reduce_(0, owner, accepted_depth.reshape(-1), reduce='amax', include_self=True)
        body_ids[:, dh, dd] = db
    loads = loads.reshape(n, 2, 5)
    vectors = vectors.reshape(n, 2, 5, 3)
    points = points.reshape(n, 2, 5, 3)
    depth = depth.reshape(n, 2, 5)
    bad |= (~torch.isfinite(signed_force).all((1, 2)) | ~torch.isfinite(loads).all((1, 2)) |
            ~torch.isfinite(vectors).all((1, 2, 3)) | ~torch.isfinite(points).all((1, 2, 3)) |
            ~torch.isfinite(references).all((1, 2, 3)))
    valid = (loads > 0) & ~bad[:, None, None]
    points = torch.where(valid[..., None], points/loads.clamp_min(1e-30)[..., None], 0.)
    loads = torch.where(valid, loads, 0.)
    vectors = torch.where(valid[..., None], vectors, 0.)
    depth = torch.where(valid, depth, 0.)
    references = torch.where(bad[:, None, None, None], 0., references)
    body_ids = torch.where(valid, body_ids, -1)
    rejected = torch.nan_to_num((total-loads.sum(-1)).clamp_min(0), nan=0., posinf=0., neginf=0.)
    s.last_signed_force_w = signed_force
    return dict(effective_pad_loads_n=loads, effective_pad_vectors_w=vectors,
                effective_pad_points_w=points, effective_pad_valid=valid, effective_pad_body_ids=body_ids,
                rejected_contact_loads_n=rejected, effective_pad_depth_m=depth,
                pad_reference_points_w=references, effective_pad_bad_env=bad)
