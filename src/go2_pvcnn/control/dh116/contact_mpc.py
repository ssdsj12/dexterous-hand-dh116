"""Parallel shooting MPC for the twelve mounted DH116 actuator coordinates.

The prediction model is a locally linearized pad Jacobian with first-order
joint tracking. CUDA Cosserat/Isaac supplies subsequent physical observations;
this optimizer does not claim to roll out the full deformable simulation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .linear_coupling import active_limits


@dataclass(frozen=True)
class ContactMPCConfig:
    horizon: int = 8
    candidates: int = 128
    elites: int = 16
    iterations: int = 4
    control_dt: float = .1
    joint_speed_limit: float = .3
    tracking_scale_m: float = .001
    movement_weight: float = .002
    tangential_weight: float = .02
    minimum_clearance_m: float = .0016
    clearance_weight: float = .1
    actuator_response: float = .5
    seed: int = 0

    def __post_init__(self):
        if (self.horizon < 1 or not 1 <= self.elites <= self.candidates or
                self.candidates < 3 or self.iterations < 1):
            raise ValueError("invalid MPC horizon/candidate counts")
        if (not all(math.isfinite(v) and v > 0 for v in
                    (self.control_dt, self.joint_speed_limit, self.tracking_scale_m))
                or not 0 < self.actuator_response <= 1 or self.movement_weight < 0
                or not 0 < self.tangential_weight <= 1
                or not math.isfinite(self.minimum_clearance_m) or self.minimum_clearance_m<=0
                or not math.isfinite(self.clearance_weight) or self.clearance_weight<0):
            raise ValueError("invalid MPC time, speed or model settings")


class ParallelContactMPC:
    """Optimize candidate hand trajectories in parallel; return the first target."""

    def __init__(self, config=None, *, device="cpu"):
        self.config = config or ContactMPCConfig()
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA contact MPC requires CUDA")
        lo, hi = active_limits()
        lo64 = torch.as_tensor(lo, dtype=torch.float64, device=self.device)
        hi64 = torch.as_tensor(hi, dtype=torch.float64, device=self.device)
        lo32, hi32 = lo64.float(), hi64.float()
        # Position targets are float32 in PhysX, while the coupling check is
        # float64. Round bounds inward so a boundary command cannot cross its
        # authoritative limit merely through representation conversion.
        self.lower = torch.where(lo32.double() < lo64,
            torch.nextafter(lo32, torch.full_like(lo32, float('inf'))), lo32).repeat(2)
        self.upper = torch.where(hi32.double() > hi64,
            torch.nextafter(hi32, torch.full_like(hi32, -float('inf'))), hi32).repeat(2)
        self.generator = torch.Generator(device=self.device).manual_seed(self.config.seed)
        self._warm_start = None
        self._tracking_bias=torch.zeros(12,dtype=torch.float32,device=self.device)
        self._last_predicted_joints=None

    def reset(self):
        """Clear warm-start and tracking state before a new episode."""
        self._warm_start=None
        self._tracking_bias.zero_()
        self._last_predicted_joints=None

    def restore_tracking_bias(self,value):
        bias=torch.as_tensor(value,dtype=torch.float32,device=self.device)
        if (bias.shape!=(12,) or not bool(torch.isfinite(bias).all())
                or bool((bias.abs()>math.pi/180+1e-8).any())):
            raise ValueError('bounded 12D hand tracking bias required')
        self._tracking_bias=bias.clone()
        self._last_predicted_joints=None

    @torch.no_grad()
    def plan_contact(self, *, active_joints, previous_action, pad_positions,
                     pad_jacobians, target_points, actuator_load_offset=None,
                     clearances=None, clearance_jacobians=None,held_coordinates=None,
                     reference_active_joints=None, reference_tolerance_rad=None):
        """Use world-frame pad points/Jacobians, excluding every Panda column."""
        values = []
        for name, value, shape in (
            ("active_joints", active_joints, (2, 6)),
            ("previous_action", previous_action, (12,)),
            ("pad_positions", pad_positions, (2, 3, 3)),
            ("pad_jacobians", pad_jacobians, (2, 3, 3, 6)),
            ("target_points", target_points, (2, 3, 3)),
        ):
            tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device)
            if tensor.shape != shape or not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"{name} must be finite with shape {shape}")
            values.append(tensor)
        q, previous, points, jacobians, targets = values
        q = q.reshape(12)
        held=(torch.zeros(12,dtype=torch.bool,device=self.device) if held_coordinates is None else
              torch.as_tensor(held_coordinates,device=self.device))
        if held.shape!=(12,) or held.dtype!=torch.bool:
            raise ValueError('held_coordinates requires 12 boolean hand-only coordinates')
        load_offset=(torch.zeros_like(q) if actuator_load_offset is None else
            torch.as_tensor(actuator_load_offset,dtype=q.dtype,device=self.device))
        if load_offset.shape!=(12,) or not bool(torch.isfinite(load_offset).all()):
            raise ValueError('actuator_load_offset must be a finite 12D hand-only vector')
        if (clearances is None)!=(clearance_jacobians is None):
            raise ValueError('clearances and clearance_jacobians must be supplied together')
        distances=(torch.empty(0,dtype=q.dtype,device=self.device) if clearances is None else
            torch.as_tensor(clearances,dtype=q.dtype,device=self.device))
        clearance_matrix=(torch.empty((0,12),dtype=q.dtype,device=self.device) if clearance_jacobians is None else
            torch.as_tensor(clearance_jacobians,dtype=q.dtype,device=self.device))
        if (distances.ndim!=1 or clearance_matrix.shape!=(len(distances),12)
                or not bool(torch.isfinite(distances).all()) or not bool(torch.isfinite(clearance_matrix).all())):
            raise ValueError('clearance observations require finite (C,) distances and (C,12) hand Jacobians')
        previous = previous.clamp(self.lower, self.upper)
        matrix = torch.zeros((18, 12), dtype=q.dtype, device=self.device)
        matrix[:9, :6] = jacobians[0].reshape(9, 6)
        matrix[9:, 6:] = jacobians[1].reshape(9, 6)
        affected = matrix.abs().sum(0)+clearance_matrix.abs().sum(0) > 1e-12
        optimized=affected & ~held
        cfg = self.config
        analytic_load_offset=load_offset.clone()
        innovation=torch.zeros_like(q)
        if actuator_load_offset is not None:
            if self._last_predicted_joints is not None:
                innovation=self._last_predicted_joints-q
                # Causal constant-disturbance observer: only the already
                # executed previous command and its measured endpoint enter.
                # Correct the reduced predictor, never measured forces or
                # controller qualification. The simulation correction bound
                # matches the existing one-degree hand tracking tolerance.
                self._tracking_bias=(self._tracking_bias+.25*innovation/cfg.actuator_response).clamp(
                    -math.pi/180,math.pi/180)
            self._tracking_bias*=affected
        else:
            self._tracking_bias.zero_()
            self._last_predicted_joints=None
        load_offset=load_offset+self._tracking_bias
        prediction_margin=.001 if actuator_load_offset is not None else 0.
        command_lower=torch.maximum(self.lower,self.lower+load_offset+prediction_margin*affected)
        command_upper=torch.minimum(self.upper,self.upper+load_offset-prediction_margin*affected)
        pose_lock = None
        if reference_active_joints is not None or reference_tolerance_rad is not None:
            if reference_active_joints is None or reference_tolerance_rad is None:
                raise ValueError('pose lock requires reference joints and tolerance')
            reference = torch.as_tensor(reference_active_joints, dtype=q.dtype, device=self.device).reshape(-1)
            if (reference.shape != (12,) or not bool(torch.isfinite(reference).all())
                    or not math.isfinite(float(reference_tolerance_rad))
                    or not 0. < float(reference_tolerance_rad) <= 1.):
                raise ValueError('pose lock requires a finite 12D reference and tolerance in (0,1]')
            tolerance = torch.as_tensor(float(reference_tolerance_rad), dtype=q.dtype, device=self.device)
            local_lower = torch.maximum(command_lower, reference - tolerance)
            local_upper = torch.minimum(command_upper, reference + tolerance)
            if bool((local_lower >= local_upper).any()):
                raise ValueError('pose lock neighborhood does not intersect active limits')
            if bool(((q < local_lower - 1e-6) | (q > local_upper + 1e-6)).any()):
                raise ValueError('measured hand state is outside candidate pose neighborhood')
            if bool(((previous < local_lower - 1e-6) | (previous > local_upper + 1e-6)).any()):
                raise ValueError('previous hand command is outside candidate pose neighborhood')
            command_lower, command_upper = local_lower, local_upper
            pose_lock = dict(reference_active_joints=reference.cpu().tolist(),
                             reference_tolerance_rad=float(reference_tolerance_rad),
                             lower=local_lower.cpu().tolist(), upper=local_upper.cpu().tolist())
        # A rolling closest surface point is not a fixed material marker.
        # Prioritize radial closure without suppressing necessary tangent motion.
        normal = (targets - points).reshape(6, 3)
        normal_length=torch.linalg.vector_norm(normal, dim=1, keepdim=True)
        normal = normal / normal_length.clamp_min(1e-12)
        tangent_scale = math.sqrt(cfg.tangential_weight)
        point_metric = (tangent_scale * torch.eye(3, dtype=q.dtype, device=self.device)
                        + (1. - tangent_scale) * normal[:, :, None] * normal[:, None, :])
        point_metric=torch.where((normal_length<1e-9)[:,:,None],
            torch.eye(3,dtype=q.dtype,device=self.device),point_metric)
        metric = torch.block_diag(*point_metric.unbind(0))
        tracking_matrix = metric @ matrix
        max_step = cfg.control_dt * cfg.joint_speed_limit
        weights = torch.ones(cfg.horizon, dtype=q.dtype, device=self.device)
        weights[-1] = 4.
        # Solve the unconstrained linear MPC exactly as the directed shooting
        # seed. CEM then refines the bounded nonlinear clipping, rather than
        # relying on random exploration to discover the first closing motion.
        indices = torch.arange(cfg.horizon, device=self.device)
        lag = indices[:, None] - indices[None, :]
        response = torch.where(lag >= 0, 1. - (1. - cfg.actuator_response) **
                               (lag + 1).clamp_min(0), 0.)
        decay = (1. - cfg.actuator_response) ** (indices + 1)
        hold_state = decay[:, None] * q + (1. - decay[:, None]) * (previous-load_offset)
        hold_residual = ((points.reshape(18) + (hold_state - q) @ matrix.T -
                         targets.reshape(18)) @ metric.T) / cfg.tracking_scale_m
        seed_matrix=tracking_matrix
        if len(distances):
            hold_clearances=distances+(hold_state-q)@clearance_matrix.T
            active=(hold_clearances<cfg.minimum_clearance_m).any(0)
            scale=math.sqrt(cfg.clearance_weight*18/len(distances))*active.to(q.dtype)
            seed_matrix=torch.cat((seed_matrix,clearance_matrix*scale[:,None]),dim=0)
            hold_residual=torch.cat((hold_residual,
                (hold_clearances-cfg.minimum_clearance_m)*scale/cfg.tracking_scale_m),dim=1)
        seed_rows=seed_matrix.shape[0]
        prediction = (response[:, None, :, None] * seed_matrix[None, :, None, :]).reshape(
            cfg.horizon * seed_rows, cfg.horizon * 12) / cfg.tracking_scale_m
        row_weights = weights.repeat_interleave(seed_rows) / (18 * cfg.horizon)
        weighted = prediction.T * row_weights
        regularizer = cfg.movement_weight / (12 * cfg.horizon * max_step**2)
        normal_matrix = weighted @ prediction + (regularizer + 1e-8) * torch.eye(
            cfg.horizon * 12, dtype=q.dtype, device=self.device)
        nominal = torch.linalg.solve(normal_matrix, -weighted @ hold_residual.reshape(-1))
        # Project the complete desired command path onto the rate bound. Simply
        # clipping each unconstrained increment discards the initial maneuver
        # rather than spreading it across subsequent prediction steps.
        desired_commands=(previous+nominal.reshape(cfg.horizon,12).cumsum(0)).clamp(command_lower,command_upper)
        command=previous.clone();projected=[]
        for desired in desired_commands:
            delta=(desired-command)*optimized
            scale=(max_step/delta.abs().amax().clamp_min(1e-12)).clamp_max(1.)
            increment=delta*scale
            projected.append(increment);command=command+increment
        nominal=torch.stack(projected)
        mean = nominal.clone() if self._warm_start is None else self._warm_start.clone()
        deviation = torch.full_like(mean, max_step * .5)
        zero = torch.zeros_like(mean)

        def rollout(controls):
            commands = (previous + controls.cumsum(1)).clamp(command_lower, command_upper)
            state = q.expand(len(controls), 12)
            predicted = [];applied_commands=[]
            last_command=previous.expand(len(controls),12)
            for index in range(cfg.horizon):
                desired=torch.where(held,previous,commands[:,index].clamp(command_lower,command_upper))
                last_command=last_command+(desired-last_command).clamp(-max_step,max_step)
                applied_commands.append(last_command)
                state = state + cfg.actuator_response * (last_command - load_offset - state)
                predicted.append(state)
            commands=torch.stack(applied_commands,dim=1)
            states = torch.stack(predicted, dim=1)
            predicted_points = points.reshape(18) + torch.einsum("ij,chj->chi", matrix, states - q)
            predicted_clearances=distances+torch.einsum('ij,chj->chi',clearance_matrix,states-q)
            residual = ((predicted_points - targets.reshape(18)) @ metric.T) / cfg.tracking_scale_m
            costs = (residual.square().mean(2) * weights).mean(1)
            if len(distances):
                violation=torch.relu(cfg.minimum_clearance_m-predicted_clearances)/cfg.tracking_scale_m
                costs+=cfg.clearance_weight*(violation.square().mean(2)*weights).mean(1)
            costs += cfg.movement_weight * (controls / max_step).square().mean((1, 2))
            return costs, commands, predicted_points, predicted_clearances

        hold_cost = rollout(zero[None])[0][0]
        best_controls = zero.clone()
        best_cost = hold_cost.clone()
        for _ in range(cfg.iterations):
            controls = mean + deviation * torch.randn((cfg.candidates, cfg.horizon, 12),
                generator=self.generator, dtype=q.dtype, device=self.device)
            controls = controls.clamp(-max_step, max_step) * optimized
            controls[0] = best_controls
            controls[1] = zero
            controls[2] = nominal
            costs, _, _, _ = rollout(controls)
            elite_ids = costs.topk(cfg.elites, largest=False).indices
            elite = controls[elite_ids]
            chosen = elite[0]
            chosen_cost = costs[elite_ids[0]]
            improve = chosen_cost < best_cost
            best_controls = torch.where(improve, chosen, best_controls)
            best_cost = torch.minimum(best_cost, chosen_cost)
            mean = elite.mean(0)
            deviation = elite.std(0, unbiased=False).clamp_min(max_step * .02)
        costs, trajectory, predicted_points, predicted_clearances = rollout(best_controls[None])
        self._warm_start = torch.cat((best_controls[1:], best_controls[-1:] * 0), dim=0)
        predicted_joints=q+cfg.actuator_response*(trajectory[0,0]-load_offset-q)
        self._last_predicted_joints=(predicted_joints.clone() if actuator_load_offset is not None else None)
        return trajectory[0, 0].reshape(2, 6), dict(
            trajectory=trajectory[0], predicted_pad_points=predicted_points[0].reshape(cfg.horizon, 2, 3, 3),
            optimized_cost=costs[0], hold_cost=hold_cost, action_dim=12,
            held_coordinates=held.clone(),
            model="linearized_pad_kinematics", candidates=cfg.candidates,
            actuator_load_offset_rad=load_offset,
            analytic_load_offset_rad=analytic_load_offset,tracking_bias_rad=self._tracking_bias.clone(),
            prediction_innovation_rad=innovation,predicted_active_joints=predicted_joints,
            prediction_joint_margin_rad=prediction_margin,
            actuator_response=cfg.actuator_response,
            predicted_clearances=predicted_clearances[0],clearance_count=len(distances),
            minimum_clearance_m=cfg.minimum_clearance_m,clearance_weight=cfg.clearance_weight,
            iterations=cfg.iterations, control_dt_s=cfg.control_dt,
            pose_lock=pose_lock)


@torch.no_grad()
def wire_surface_targets(pad_positions, wire_nodes, *, radius=.008,
                         target_loads=None, contact_stiffness=1e4, opposed_pinch=False,
                         penetration_m=None):
    """Select actual centerline surface points with a small force preload."""
    points = torch.as_tensor(pad_positions)
    nodes = torch.as_tensor(wire_nodes, dtype=points.dtype, device=points.device)
    if points.shape != (2, 3, 3) or nodes.ndim != 2 or nodes.shape[0] != 3 or nodes.shape[1] < 2:
        raise ValueError("expected six world pad positions and a (3,N) wire centerline")
    start = nodes[:, :-1].T
    segment = nodes[:, 1:].T - start
    offset = points.reshape(6, 1, 3) - start[None]
    fraction = (offset * segment[None]).sum(2) / segment.square().sum(1).clamp_min(1e-12)[None]
    closest = start[None] + fraction.clamp(0, 1)[..., None] * segment[None]
    distance2 = (points.reshape(6, 1, 3) - closest).square().sum(2)
    ids = distance2.argmin(1)
    centers = closest[torch.arange(6, device=points.device), ids]
    radial = points.reshape(6, 3) - centers
    length = torch.linalg.vector_norm(radial, dim=1, keepdim=True)
    fallback = torch.zeros_like(radial); fallback[:, 2] = 1.
    radial = torch.where(length > 1e-9, radial / length.clamp_min(1e-9), fallback)
    radii = torch.as_tensor(radius, dtype=points.dtype, device=points.device)
    if radii.ndim:
        radii = radii[ids]
    loads = torch.zeros(6, dtype=points.dtype, device=points.device) if target_loads is None else \
        torch.as_tensor(target_loads, dtype=points.dtype, device=points.device).reshape(6)
    depth = loads / contact_stiffness if penetration_m is None else torch.as_tensor(
        penetration_m, dtype=points.dtype, device=points.device).reshape(6)
    desired_radius = (radii - depth).clamp_min(0.)
    targets=(centers + radial * desired_radius[..., None]).reshape(2, 3, 3)
    if opposed_pinch:
        centers=centers.reshape(2,3,3)
        hand_center=centers[:,:2].mean(1)
        tangent=segment[ids.reshape(2,3)[:,0]]
        tangent=tangent/torch.linalg.vector_norm(tangent,dim=1,keepdim=True).clamp_min(1e-9)
        axis=points[:,0]-points[:,1]
        axis=axis-(axis*tangent).sum(1,keepdim=True)*tangent
        axis=axis/torch.linalg.vector_norm(axis,dim=1,keepdim=True).clamp_min(1e-9)
        radii=desired_radius.expand(6).reshape(2,3)
        targets[:,0]=hand_center+axis*radii[:,0,None]
        targets[:,1]=hand_center-axis*radii[:,1,None]
    return targets
