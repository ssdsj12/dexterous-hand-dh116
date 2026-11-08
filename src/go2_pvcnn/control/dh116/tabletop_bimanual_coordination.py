"""Opt-in simulation interventions on the existing physical 24D action path.

The shared Actor is untouched. The caller supplies measured world coordinates,
effective opposed contacts and attributed upward support. These privileged
signals belong to experiments and do not become visual Student observations.
The unchanged native actuator, collision and table guards execute afterwards.
"""
from dataclasses import dataclass
import math

import torch

from .tabletop_surface_contract import ACTION_CONTRACT
from .tabletop_policy_control import CONTRACT as POLICY_CONTROL


CONTRACT = dict(
    version='dh116-bimanual-coordination-experiment-v1',
    modes=['Z0', 'Z1', 'Z2', 'Z3'], default='Z0',
    action='existing_physical24_world_translation_intrinsic_XYZ_rotation',
    controller_output_source='privileged_simulation_intervention',
    measurement_source='native_preceding_physical_sample',
    support_allocation='longitudinal_vertical_static_moment_projection_on_actual_grasp_line',
    support_scope='diagnostic_vertical_equilibrium_not_full_contact_wrench_feasibility',
    z3_pickup_admission='native_opposed_pads_above_0.05N_and_positive_attributed_upward_support_each_hand',
    z3_floating_stability='full_COM_allocated_support_at_native_30mm_hold_height',
    success_evaluation='unchanged_native_30mm_5s_0.5mm_bilateral_opposed_contact',
    hardware_validated=False,
)


@dataclass
class CoordinationState:
    """One preceding valid native sample per environment.

    Palm and grasp points are measured world coordinates. ``palm_offset_m`` is
    the actual carriage displacement in the native SDK target frame, supplied
    after the root/palm transform has been audited. It must exclude the dynamic
    task offset. ``support_n`` is signed, attributed object-side upward support,
    not total normal-load magnitude. Episode IDs are unique monotonically
    assigned integers per row, with the full episode_uid retained by the runner.
    """

    palm_world_m: torch.Tensor
    palm_offset_m: torch.Tensor
    clearance_m: torch.Tensor
    grasp_points_w: torch.Tensor
    object_com_w: torch.Tensor
    object_lift_m: torch.Tensor
    opposed: torch.Tensor
    contact_valid: torch.Tensor
    support_n: torch.Tensor
    object_weight_n: torch.Tensor
    joint_tracking_error_rad: torch.Tensor
    mimic_error_rad: torch.Tensor
    slip_m: torch.Tensor
    slip_valid: torch.Tensor
    slip_applicable: torch.Tensor
    live: torch.Tensor
    episode_id: torch.Tensor


def coordination_state_from_native(world, task, env):
    """Read the current native simulation without filling missing contacts.

    The DH116 source asset uses base_link as the wrist/palm reference. Actual
    body-link coordinates are used, with the existing target-frame conversion
    for queued targets. Grasp points are measured effective-contact centroids.
    A hand without contact uses the object center as an explicitly unopposed
    placeholder; it cannot satisfy the load-allocation/contact ascent gate.
    """
    from .tabletop_palm_mount import measured_offset
    from .tabletop_surface_force_sharing import surface_targets
    from .tabletop_surface_exploration import support_components
    from .tabletop_tensor_world import rotated_points

    m = task.m
    reference = world.roots
    n = world.n
    loads = m['effective_pad_loads_n'].to(reference)
    vectors = m['effective_pad_vectors_w'].to(reference)
    valid_pad = m['effective_pad_valid']
    contact_valid = (torch.isfinite(loads).all(-1) & (loads >= 0).all(-1)
                     & torch.isfinite(vectors).all((-1, -2)))
    clean_loads = torch.where(contact_valid[:, :, None], loads, 0.)
    clean_vectors = torch.where(contact_valid[:, :, None, None], vectors, 0.)
    paired = surface_targets(clean_loads, clean_vectors, valid=valid_pad)['paired']
    points = m['effective_pad_points_w'].to(reference)
    point_valid = torch.isfinite(points).all(-1)
    bearing = valid_pad & point_valid & (loads > .05)
    weights = torch.where(bearing, clean_loads, 0.)
    center = world.wp.to(reference)
    com = getattr(world.wire.data, 'root_com_pos_w', None)
    if com is None:
        # Older SDKs expose a local COM transform on the PhysX view. Apply the
        # measured object body-link transform instead of assuming COM=origin.
        local_com = world.wire.root_physx_view.get_coms().to(reference).reshape(n, 7)[:, :3]
        com = rotated_points(torch.cat((center, world.wq.to(reference)), -1), local_com)
    else:
        com = com.to(reference)
    grasp = (torch.where(point_valid[..., None], points, 0.) * weights[..., None]).sum(-2)
    grasp /= weights.sum(-1).clamp_min(1e-12)[..., None]
    grasp = torch.where((weights.sum(-1) > 0)[..., None], grasp, center[:, None, :])
    contact_valid &= (~valid_pad | point_valid).all(-1)
    # Same object-side attribution as the native surface diagnostic. Positive
    # friction is credited only to accepted contacts; downward force stays signed.
    whole_support = world.vertical_components.sum(-1)
    _, support = support_components(m, whole_support)
    measured = measured_offset(reference, world.root_targets, world.palms)
    episode_id = env.diagnostic_episode_ids
    live = ~task.done
    telemetry = getattr(env, 'evaluation_telemetry', None)
    if telemetry is None:
        raise ValueError('native evaluation telemetry required for physical lift reference')
    raw_valid = (torch.isfinite(reference).all((1, 2)) & torch.isfinite(center).all(-1)
                 & torch.isfinite(com).all(-1)
                 & contact_valid.all(-1))
    # _last intentionally preserves terminal evidence through no-op/reset. Read
    # the current physical anchor instead of borrowing a previous episode's slip.
    slip, slip_valid, _, applicable, _ = telemetry._slip(
        m, raw_valid, (paired & contact_valid).all(-1), contact_valid.all(-1),
        live, telemetry.capture_anchor(task))
    bottom = m['bottom'].to(reference)
    return CoordinationState(
        palm_world_m=reference[:, :, :3].clone(), palm_offset_m=measured,
        clearance_m=m['hand_clearance'].to(reference), grasp_points_w=grasp,
        object_com_w=com, object_lift_m=bottom - telemetry.reference_bottom_m.to(reference),
        opposed=paired, contact_valid=contact_valid, support_n=support.to(reference),
        object_weight_n=world.object_weight_n.to(reference),
        joint_tracking_error_rad=(task.command - m['q']).reshape(n, 2, 6).abs().amax(-1).to(reference),
        mimic_error_rad=m['mimic_errors'].reshape(n, 2, 5).abs().amax(-1).to(reference),
        slip_m=slip, slip_valid=slip_valid, slip_applicable=applicable,
        live=live, episode_id=episode_id.to(device=reference.device, dtype=torch.long),
    )


def static_support_allocation(grasp_points_w, object_com_w, object_weight_n):
    """Vertical reactions balancing the longitudinal COM moment projection.

    A COM outside the horizontal grasp segment requires a negative reaction.
    Such a sample is marked infeasible. Transverse COM offset and contact
    wrench moments remain outside this projection's feasibility claim.
    """
    if (grasp_points_w.ndim != 3 or grasp_points_w.shape[1:] != (2, 3)
            or object_com_w.shape != (len(grasp_points_w), 3)
            or object_weight_n.shape != (len(grasp_points_w),)):
        raise ValueError('matching world grasp points, COM and object weight required')
    line = grasp_points_w[:, 1, :2] - grasp_points_w[:, 0, :2]
    distance2 = line.square().sum(-1)
    along = ((object_com_w[:, :2] - grasp_points_w[:, 0, :2]) * line).sum(-1)
    fraction = along / distance2.clamp_min(1e-16)
    finite = (torch.isfinite(grasp_points_w).all((1, 2))
              & torch.isfinite(object_com_w).all(-1) & torch.isfinite(object_weight_n))
    valid = finite & (object_weight_n > 0) & (distance2 > 1e-16)
    valid &= (fraction >= 0) & (fraction <= 1)
    required = torch.stack((1 - fraction, fraction), -1) * object_weight_n[:, None]
    return torch.where(valid[:, None], required, float('nan')), valid


class BimanualCoordinator:
    """Batched bounded Z interventions with episode-local stability history.

    ``project`` belongs immediately before the native physical control call,
    once per actual substep. The runner MUST apply ``cancel_upward_target`` to
    its queued carriage Z target using ``hold_offset_m`` before the unchanged
    native safety projection. A zero velocity alone retains a stale target.

    Z2 only restricts the right Z request. Z3 changes left/right translation Z
    and reserves the original shared Cartesian speed budget. Fingers, rotations
    and the physical limits are never changed here.
    """

    def __init__(self, n, *, mode='Z0', resume_dwell_s=.2,
                 prediction_horizon_s=1/60, target_lift_m=.03,
                 objective_weights=(1., 1., 1., 1., 1.)):
        if type(n) is not int or n < 1 or mode not in CONTRACT['modes']:
            raise ValueError('positive environment count and Z0-Z3 mode required')
        if (not math.isfinite(resume_dwell_s) or resume_dwell_s < 0
                or not math.isfinite(prediction_horizon_s) or prediction_horizon_s <= 0
                or not math.isfinite(target_lift_m) or target_lift_m < .03):
            raise ValueError('finite nonnegative dwell, positive horizon and at least 30mm goal required')
        if len(objective_weights) != 5 or any(not math.isfinite(w) or w <= 0 for w in objective_weights):
            raise ValueError('five finite positive experiment objective weights required')
        self.n, self.mode = n, mode
        self.resume_dwell_s = resume_dwell_s
        self.horizon = prediction_horizon_s
        self.target_lift_m = target_lift_m
        self.weights = objective_weights
        self._episode = self._stable_s = self._grasp_offset_z = self._anchored = None

    def reset(self, ids):
        """Discard selected histories before the next physical episode."""
        if self._episode is not None:
            self._episode[ids] = -1
            self._stable_s[ids] = 0
            self._grasp_offset_z[ids] = 0
            self._anchored[ids] = False

    def _validate_state(self, state, action):
        if not isinstance(state, CoordinationState):
            raise ValueError('measured CoordinationState required for Z2/Z3')
        shapes = dict(palm_world_m=(2, 3), palm_offset_m=(2, 3), clearance_m=(2,),
            grasp_points_w=(2, 3), object_com_w=(3,), object_lift_m=(),
            support_n=(2,), object_weight_n=(), joint_tracking_error_rad=(2,),
            mimic_error_rad=(2,), slip_m=())
        finite = torch.ones(self.n, dtype=torch.bool, device=action.device)
        for name, tail in shapes.items():
            value = getattr(state, name)
            if (not isinstance(value, torch.Tensor) or value.shape != (self.n,) + tail
                    or value.device != action.device or value.dtype != action.dtype):
                raise ValueError(f'{name} requires matching device/dtype and shape {(self.n,) + tail}')
            # Unknown slip before an anchor is expected and remains NA.
            if name != 'slip_m':
                finite &= torch.isfinite(value).reshape(self.n, -1).all(-1)
        for name, tail in dict(opposed=(2,), contact_valid=(2,), slip_valid=(),
                               slip_applicable=(), live=()).items():
            value = getattr(state, name)
            if (not isinstance(value, torch.Tensor) or value.shape != (self.n,) + tail
                    or value.device != action.device or value.dtype != torch.bool):
                raise ValueError(f'{name} requires matching boolean shape {(self.n,) + tail}')
        if (state.episode_id.shape != (self.n,) or state.episode_id.device != action.device
                or state.episode_id.dtype != torch.long or (state.episode_id < 0).any()):
            raise ValueError('one nonnegative int64 episode_id per environment required')
        return finite

    def _history(self, state, action, finite):
        if self._episode is None:
            self._episode = torch.full_like(state.episode_id, -1)
            self._stable_s = action.new_zeros(self.n)
            self._grasp_offset_z = action.new_zeros(self.n, 2)
            self._anchored = torch.zeros_like(state.live)
        if self._stable_s.device != action.device or self._stable_s.dtype != action.dtype:
            raise ValueError('coordinator device/dtype cannot change between physical samples')
        changed = self._episode != state.episode_id
        if (state.episode_id < self._episode).any():
            raise ValueError('episode identity cannot move backwards')
        self._stable_s[changed] = 0
        self._anchored[changed] = False
        relative = state.palm_world_m[:, :, 2] - state.grasp_points_w[:, :, 2]
        # Invalid first samples cannot establish a valid geometric reference.
        capture = ~self._anchored & finite & state.live & (state.opposed & state.contact_valid).all(-1)
        self._grasp_offset_z[capture] = relative[capture]
        self._anchored[capture] = True
        self._episode[finite & state.live] = state.episode_id[finite & state.live]
        relative_valid = self._anchored[:, None] & state.opposed & state.contact_valid
        return torch.where(relative_valid, relative - self._grasp_offset_z, float('nan')), changed

    @torch.no_grad()
    def project(self, physical_action24, state=None, *, dt_s=None):
        """Return a fresh issued action plus per-environment audit tensors."""
        raw = physical_action24
        if (not isinstance(raw, torch.Tensor) or raw.shape != (self.n, 24)
                or not raw.is_floating_point() or not torch.isfinite(raw).all()
                or (raw.abs() > 1 + 1e-12).any()):
            raise ValueError('finite normalized physical [N,24] action in [-1,1] required')
        output = raw.clone()
        info = dict(CONTRACT, mode=self.mode)
        if self.mode == 'Z0':
            return output, info
        if self.mode == 'Z1':
            output[:, 17] = 0
            return output, info
        if dt_s is None or not math.isfinite(dt_s) or dt_s <= 0:
            raise ValueError('positive native physical dt_s required')
        measured = self._validate_state(state, raw)
        relative_error, changed = self._history(state, raw, measured)
        required, allocation_valid = static_support_allocation(
            state.grasp_points_w, state.object_com_w, state.object_weight_n)
        tracking = ((state.joint_tracking_error_rad.abs() < .03).all(-1)
                    & (state.mimic_error_rad.abs() < POLICY_CONTROL['severe_coupling_rad']).all(-1))
        slip_ok = (~state.slip_applicable | (state.slip_valid & torch.isfinite(state.slip_m)
                                            & (state.slip_m <= .0005) & (state.slip_m >= 0)))
        bilateral = measured & state.live & tracking & slip_ok
        bilateral &= (state.opposed & state.contact_valid).all(-1)
        floating_support = bilateral & allocation_valid & (state.support_n >= required).all(-1)
        # Native opposition already requires measured bearing pads above 0.05 N.
        # Pickup needs positive accepted upward support in both hands, while the
        # original fixture may still carry part of the cable's weight. Requiring
        # full weight before the first upward motion would create a deadlock.
        pickup_support = bilateral & (state.support_n > 0).all(-1)
        at_hold_height = state.object_lift_m >= .03
        support = (pickup_support & (~at_hold_height | floating_support)
                   if self.mode == 'Z3' else floating_support)
        self._stable_s = torch.where(support, self._stable_s + dt_s, 0.)
        ready = support & (self._stable_s >= self.resume_dwell_s - 1e-12)
        rate = ACTION_CONTRACT['translation_rate_m_s']
        floor = POLICY_CONTROL['palm_descent_clearance_m']
        lower = ((floor - state.clearance_m) / (rate * dt_s)).clamp(-1, 0)
        relative_lower = (-relative_error / (rate * dt_s)).clamp(-1, 0)
        relative_known = self._anchored & state.opposed[:, 1] & state.contact_valid[:, 1]
        relative_lower[:, 1] = torch.where(relative_known, relative_lower[:, 1], -1.)
        table_limited = raw[:, 17] < lower[:, 1]
        relative_limited = raw[:, 17] < relative_lower[:, 1]
        if self.mode == 'Z2':
            low = torch.maximum(lower[:, 1], relative_lower[:, 1])
            z = torch.maximum(output[:, 17], low)
            z = torch.where((z > 0) & ~ready, 0., z)
            output[:, 17] = torch.where(measured & state.live, z, 0.)
            cancel = torch.zeros_like(state.opposed)
            cancel[:, 1] = ~ready & state.live
            objective = {}
        else:
            z, objective = self._coordinated_z(state, relative_error)
            z = torch.maximum(z, lower)
            z = torch.where((z > 0) & ~ready[:, None], 0., z)
            z = torch.where(measured[:, None], z, 0.)
            # Native XYZ velocity is vector-norm limited. Reserve X/Y budget;
            # avoid changing their values while selecting an admissible Z.
            xy = raw[:, 12:18].reshape(self.n, 2, 3)[:, :, :2]
            budget = (1 - xy.square().sum(-1)).clamp_min(0).sqrt()
            z = torch.maximum(torch.minimum(z, budget), -budget)
            output[:, [14, 17]] = z
            cancel = (~ready & state.live)[:, None].expand(self.n, 2).clone()
        output[~state.live] = 0
        info.update(measurement_valid=measured, episode_changed=changed,
            support_ready=support, resume_ready=ready, stable_s=self._stable_s.clone(),
            pickup_support_ready=pickup_support, floating_support_ready=floating_support,
            at_native_hold_height=at_hold_height,
            required_support_n=required, load_allocation_valid=allocation_valid,
            support_share_residual_n=state.support_n - required,
            relative_grasp_error_m=relative_error, table_limited=table_limited,
            relative_grasp_valid=torch.isfinite(relative_error),
            right_relative_height_limited=relative_limited,
            cancel_upward_target=cancel, hold_offset_m=state.palm_offset_m.clone(),
            hold_offset_valid=torch.isfinite(state.palm_offset_m).all(-1),
            requested_z_action=raw[:, [14, 17]].clone(), issued_z_action=output[:, [14, 17]].clone(),
            objective=objective,
            objective_scope='kinematic_candidate_before_hard_ascent_clearance_and_velocity_gates',
            objective_weights=dict(zip(('lift', 'contact', 'slip', 'joint', 'sync'), self.weights)),
            resume_dwell_s=self.resume_dwell_s, prediction_horizon_s=self.horizon,
            target_lift_m=self.target_lift_m,
            stable_duration_scope='continuous_measured_support_within_same_physical_episode')
        return output, info

    def _coordinated_z(self, state, relative_error):
        """Solve a two-variable convex Z objective before hard projection.

        Lift tracks the physical object gap. Contact tracks the episode's palm
        to grasp-point offset. Slip and joint terms penalize motion according
        to measured drift/tracking. Sync tracks unequal measured offsets, not
        identical commanded XYZ. This kinematic objective does not predict
        contact forces or certify the native success criterion.
        """
        lift_weight, contact_weight, slip_weight, joint_weight, sync_weight = self.weights
        reach = ACTION_CONTRACT['translation_rate_m_s'] * self.horizon
        clean_relative = torch.nan_to_num(relative_error)
        desired = torch.nan_to_num((self.target_lift_m - state.object_lift_m) / reach).clamp(0, 1)
        contact_desired = (-clean_relative / reach).clamp(-1, 1)
        slip_ratio = torch.where(state.slip_applicable & state.slip_valid,
            torch.nan_to_num(state.slip_m / .0005, nan=1.).clamp(0, 1), 0.)
        joint_ratio = torch.nan_to_num(state.joint_tracking_error_rad / .03, nan=1.).abs().clamp(0, 1)
        sync_offset = ((clean_relative[:, 0] - clean_relative[:, 1]) / reach).clamp(-2, 2)
        # J_lift=(mean(z)-desired)^2; J_contact=mean((z-c)^2).
        shared = lift_weight * .25
        cross = shared - sync_weight
        diagonal = (shared + contact_weight * .5 + slip_weight * slip_ratio[:, None]
                    + joint_weight * joint_ratio + sync_weight)
        rhs = (lift_weight * desired[:, None] * .5
               + contact_weight * contact_desired * .5)
        rhs[:, 0] -= sync_weight * sync_offset
        rhs[:, 1] += sync_weight * sync_offset
        determinant = diagonal[:, 0] * diagonal[:, 1] - cross * cross
        left = (rhs[:, 0] * diagonal[:, 1] - rhs[:, 1] * cross) / determinant
        right = (rhs[:, 1] * diagonal[:, 0] - rhs[:, 0] * cross) / determinant
        z = torch.stack((left, right), -1).clamp(-1, 1)
        objective = dict(lift=(z.mean(-1) - desired).square(),
            contact=(z - contact_desired).square().mean(-1),
            slip=slip_ratio * z.square().sum(-1), joint=(joint_ratio * z.square()).sum(-1),
            sync=(z[:, 0] - z[:, 1] + sync_offset).square())
        objective['total'] = sum(w * objective[name] for w, name in
            zip(self.weights, ('lift', 'contact', 'slip', 'joint', 'sync')))
        return z, objective
