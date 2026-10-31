"""Optional native-contact geometry recovery; never an effective-force source.

No sampler, mask, load, reward, controller or physics configuration is changed.
Only an explicit PadCentering.adjust() call changes a caller's velocity request.
The wider near-field search is for inner-face recovery geometry ONLY.
"""
import math

import torch

from .tabletop_effective_pad_runtime import DISTAL_LINKS, _on_patch, _owners, _rotation


def _distal_poses(sampler):
    return torch.stack([hand.data.body_link_pose_w[:, [hand.body_names.index(link)
                        for link in DISTAL_LINKS]] for hand in sampler.world.hands], dim=1).double()


@torch.no_grad()
def _read_native(sampler):
    """Return force-weighted LOCAL recovery points, validity and bad-env bits.

    Read only the ten distal object-filtered sensors, using the physics dt for
    native force units. Original rows/buffers and sampler measurement fields
    remain untouched. Finite negative/release scalars are not invalid, but do
    not supply recovery geometry. Any bad owned data quarantines that env.
    """
    s = sampler
    pose = _distal_poses(s)
    device = pose.device
    bad = ((~torch.isfinite(pose).all((-1, -2, -3)))
           | ((pose[..., 3:].norm(dim=-1)-1).abs() > 1e-3).any((-1, -2)))
    rotations = _rotation(pose)
    positions = torch.nan_to_num(pose[..., :3], nan=0., posinf=0., neginf=0.)
    weights = torch.zeros(s.n, 2, 5, dtype=torch.float64, device=device)
    weighted = torch.zeros(s.n, 2, 5, 3, dtype=torch.float64, device=device)
    for h, _, digit, sensor in s.bindings:
        if digit is None:
            continue
        raw = sensor.contact_physx_view.get_contact_data(dt=s.world.dt)
        if (len(raw) != 6 or not all(isinstance(x, torch.Tensor) for x in raw)
                or any(x.device != device for x in raw)):
            bad.fill_(True)
            continue
        force, point, normal, separation, count, start = raw
        capacity = force.numel()
        if (point.numel() != capacity*3 or normal.numel() != capacity*3
                or separation.numel() != capacity or count.numel() != s.n or start.numel() != s.n):
            bad.fill_(True)
            continue
        f = force.reshape(-1).double()
        point, normal = point.reshape(-1, 3).double(), normal.reshape(-1, 3).double()
        sep = separation.reshape(-1).double()
        count, start = count.reshape(s.n), start.reshape(s.n)
        integer = (torch.isfinite(count) & torch.isfinite(start)
                   & (count == count.trunc()) & (start == start.trunc()))
        bad |= ~integer
        count = torch.where(integer, count, -1).long()
        start = torch.where(integer, start, -1).long()
        env, owned, invalid = _owners(count, start, capacity)
        bad |= invalid
        if not capacity:
            continue
        finite = (torch.isfinite(f) & torch.isfinite(point).all(-1)
                  & torch.isfinite(normal).all(-1) & torch.isfinite(sep))
        sane = finite & ((normal.norm(dim=-1)-1).abs() <= .05)
        faults = torch.zeros(s.n, device=device, dtype=torch.long)
        faults.scatter_add_(0, env, (owned & ~sane).long())
        bad |= faults > 0
        r = rotations[env, h, digit]
        safe_point = torch.nan_to_num(point, nan=0., posinf=0., neginf=0.)
        safe_normal = torch.nan_to_num(normal, nan=0., posinf=0., neginf=0.)
        local_point = torch.einsum('ki,kij->kj', safe_point-positions[env, h, digit], r)
        local_normal = torch.einsum('ki,kij->kj', safe_normal, r)
        row = s.rows[h, digit]
        recovery_row = dict(row)  # Shallow copy, replace only this copy's width tensor.
        recovery_row['half_widths_m'] = torch.full_like(row['half_widths_m'], torch.inf)
        rel = local_point-row['center_local_m']
        uv = torch.stack(((rel*row['tangent_u']).sum(-1), (rel*row['tangent_v']).sum(-1)), -1)
        ellipse = ((uv/row['half_widths_m'])**2).sum(-1)
        accepted = (owned & sane & (f > 0) & (ellipse <= 2.25)
                    & _on_patch(local_point, local_normal, recovery_row))
        recovery_f = torch.where(accepted, f, 0.)
        weights[:, h, digit].scatter_add_(0, env, recovery_f)
        safe_local = torch.where(accepted[:, None], local_point, 0.)
        weighted[:, h, digit].scatter_add_(0, env[:, None].expand(-1, 3), safe_local*recovery_f[:, None])
    bad |= ~torch.isfinite(weights).all((-1, -2)) | ~torch.isfinite(weighted).all((-1, -2, -3))
    valid = (weights > 0) & ~bad[:, None, None]
    local = weighted/weights.clamp_min(torch.finfo(torch.float64).tiny)[..., None]
    local = torch.where(valid[..., None], local, 0.)
    return local, valid, bad


class PadCentering:
    """Teacher-only tangential centering using nearby native contact geometry.

    adjust accepts float64 world velocity/inward arrays (N,2,5,3), object pose
    (N,7; world xyz+wxyz) and velocity (N,6; world linear xyz+angular xyz).
    Object point velocity is V+omega cross (contact_world-object_center).
    The correction moves the FK pad center TOWARDS the native contact point.

    Cache stores local geometry, never force. A fresh read resets age; a missed
    read may retain it for <=0.1 s using the CURRENT real link pose. Call reset
    for each environment reset. Invalid owned data clears that environment.

    The original normal scalar is preserved, except a known recovery point at
    ellipse>=1 suppresses positive inward motion. Unloading remains unchanged.
    Recovery tangent is capped at 3mm/s, then blended into the old tangent with
    weight clamp((sqrt(ellipse)-.45)/.4,0,1). Diagnostics are geometry-only and
    MUST NOT enter effective loads/rewards or physical measurement admission.

    Explicit transport_velocity_w preserves the planned rigid translation and
    applies centering to the relative velocity. Recovery cannot oppose planned
    tangential transport; outside-patch suppression affects only extra inward
    correction, never the normal component of the transport itself.
    """
    def __init__(self, sampler, *, kp=4.):
        if not math.isfinite(kp) or kp < 0:
            raise ValueError('nonnegative finite centering gain required')
        distal = [(h, b, d, sensor) for h, b, d, sensor in sampler.bindings if d is not None]
        if (sampler.n < 1 or len(sampler.world.hands) != 2 or len(distal) != 10
                or {(h, d) for h, _, d, _ in distal} != {(h, d) for h in range(2) for d in range(5)}):
            raise ValueError('exactly ten named distal object-contact bindings required')
        if any(sampler.world.hands[h].body_names[b] != DISTAL_LINKS[d]
               or getattr(sensor.contact_physx_view, 'filter_count', 1) != 1
               for h, b, d, sensor in distal):
            raise ValueError('distal link identity and one object-only contact filter required')
        self.sampler, self.kp = sampler, kp
        device = sampler.world.hands[0].data.body_link_pose_w.device
        self._local = torch.zeros(sampler.n, 2, 5, 3, device=device, dtype=torch.float64)
        self._valid = torch.zeros(sampler.n, 2, 5, device=device, dtype=torch.bool)
        self._age = torch.zeros(sampler.n, 2, 5, device=device, dtype=torch.float64)

    @torch.no_grad()
    def reset(self, ids):
        ids = torch.as_tensor(ids, device=self._local.device, dtype=torch.long)
        self._local[ids] = 0.
        self._valid[ids] = False
        self._age[ids] = 0.

    @torch.no_grad()
    def adjust(self, velocity, actual_inward, object_pose, object_velocity, dt,*,transport_velocity_w=None):
        nenv = self.sampler.n
        shape = (nenv, 2, 5, 3)
        args = (velocity, actual_inward, object_pose, object_velocity)
        if (not all(isinstance(x, torch.Tensor) for x in args)
                or velocity.shape != shape or actual_inward.shape != shape
                or object_pose.shape != (nenv, 7) or object_velocity.shape != (nenv, 6)):
            raise ValueError('matching pad velocities, inward axes and world object state required')
        if any(x.dtype != torch.float64 or x.device != self._local.device for x in args):
            raise ValueError('float64 input tensors on the sampler device required')
        if not all(bool(torch.isfinite(x).all()) for x in args) or not math.isfinite(dt) or dt <= 0:
            raise ValueError('finite inputs and positive finite control dt required')
        transport=torch.zeros_like(velocity)
        if transport_velocity_w is not None:
            if (transport_velocity_w.shape!=(nenv,3) or transport_velocity_w.dtype!=torch.float64
                    or transport_velocity_w.device!=velocity.device or not torch.isfinite(transport_velocity_w).all()):
                raise ValueError('finite float64 N x 3 planned transport velocity required')
            transport=transport_velocity_w[:,None,None,:].expand_as(velocity)

        local, fresh, bad = _read_native(self.sampler)
        self._age += dt
        self._local = torch.where(fresh[..., None], local, self._local)
        self._age = torch.where(fresh, 0., self._age)
        self._valid = (self._valid | fresh) & (self._age <= .1+1e-12) & ~bad[:, None, None]
        poses = _distal_poses(self.sampler)
        r = _rotation(poses)
        pos = torch.nan_to_num(poses[..., :3], nan=0., posinf=0., neginf=0.)
        rows = [self.sampler.rows[h, d] for h in range(2) for d in range(5)]
        center, u, v, widths = (torch.stack([row[key] for row in rows]).reshape(2, 5, -1)
            for key in ('center_local_m', 'tangent_u', 'tangent_v', 'half_widths_m'))
        rel = self._local-center
        du, dv = (rel*u).sum(-1), (rel*v).sum(-1)
        ellipse = (du/widths[..., 0])**2+(dv/widths[..., 1])**2
        geometry_valid = self._valid & (ellipse <= 2.25)
        length = actual_inward.norm(dim=-1, keepdim=True)
        axis = actual_inward/length.clamp_min(1e-30)
        active = geometry_valid & (length[..., 0] > 1e-12)
        weight = ((ellipse.sqrt()-.45)/.4).clamp(0., 1.)
        weight = torch.where(active, weight, 0.)
        contact = (r @ self._local[..., None]).squeeze(-1)+pos
        pad_center = (r @ center[..., None]).squeeze(-1)+pos
        local_tangent_error = du[..., None]*u+dv[..., None]*v
        tangent_error = (r @ local_tangent_error[..., None]).squeeze(-1)
        lever = contact-object_pose[:, None, None, :3]
        point_velocity = object_velocity[:, None, None, :3]+torch.linalg.cross(
            object_velocity[:, None, None, 3:].expand_as(lever), lever)
        recovery = (point_velocity if transport_velocity_w is None else torch.zeros_like(point_velocity))+self.kp*tangent_error
        recovery -= (recovery*axis).sum(-1, keepdim=True)*axis
        recovery *= (.003/recovery.norm(dim=-1, keepdim=True).clamp_min(1e-30)).clamp_max(1.)
        if transport_velocity_w is not None:
            tangent_transport=transport-(transport*axis).sum(-1,keepdim=True)*axis
            opposing=(recovery*tangent_transport).sum(-1,keepdim=True).clamp_max(0.)
            recovery-=opposing/tangent_transport.square().sum(-1,keepdim=True).clamp_min(1e-30)*tangent_transport

        relative_velocity=velocity-transport
        vn = (relative_velocity*axis).sum(-1)
        suppressed = active & (ellipse >= 1.) & (vn > 0)
        target_vn = torch.where(suppressed, 0., vn)
        old_tangent = relative_velocity-vn[..., None]*axis
        tangent = old_tangent+weight[..., None]*(recovery-old_tangent)
        adjusted = tangent+target_vn[..., None]*axis
        # Remove finite-precision projection leakage without changing tangent.
        adjusted += (target_vn-(adjusted*axis).sum(-1))[..., None]*axis
        adjusted += transport
        applied = active & ((weight > 0) | suppressed)
        adjusted = torch.where(applied[..., None], adjusted, velocity)

        def geometry_only(value):
            return torch.where(geometry_valid[..., None], value, 0.)
        diag = dict(geometry_valid=geometry_valid.clone(), fresh=fresh.clone(),
            cached=(geometry_valid & ~fresh), age_s=self._age.clone(), bad_env=bad.clone(),
            active=active, applied=applied, weight=weight,
            ellipse=torch.where(geometry_valid, ellipse, 0.),
            contact_local_m=geometry_only(self._local), contact_point_w=geometry_only(contact),
            pad_center_w=geometry_only(pad_center), tangent_error_w=geometry_only(tangent_error),
            recovery_tangent_velocity=geometry_only(recovery), positive_normal_suppressed=suppressed)
        return adjusted, diag
