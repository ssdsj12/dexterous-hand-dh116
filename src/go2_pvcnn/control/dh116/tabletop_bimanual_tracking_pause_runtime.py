"""Explicit hybrid post-control / pre-transport common wrist pause adapter.

The original controller executes once. This helper can then shorten its finger
increments and cancel both queued wrist targets in the measured SDK chart.
The original advance, transport, collision, unloading and terminal guards still
execute. No measured roots, velocities, forces or Student observations change.
"""
from dataclasses import replace

import torch

from .tabletop_bimanual_tracking_guard import (CommonTrackingPause,
    common_pause_feedback_from_native, apply_common_pause_wrist_targets,read_common_pause_targets)


class CommonTrackingPauseRuntime:
    """Teacher/hybrid-only candidate, with separate logical/SDK/readback records.

    Direct order::

        allowed, info = runtime.control(world, task, raw, live, dt,
            original_guard_feedback=original_guard_feedback)
        measurement = world.advance(task, allowed, live)  # original guards
        record = runtime.after_advance(world, task, live)

    ``original_guard_feedback`` must supply the original controller's mandatory
    finger/wrist unloading masks and overload mask. They are never inferred from
    the sign of a joint action, an optimizer prediction, or a task phase.
    """
    def __init__(self,n,*,teacher_error_p999_rad,enabled=False,source='teacher_hybrid',
                 sink=None,max_pause_s=2.,safe_velocity_rad_s=.05):
        if enabled and source not in ('teacher_hybrid','hybrid'):
            raise ValueError('common pause runtime requires explicit teacher/hybrid provenance')
        if sink is not None and not callable(sink):raise ValueError('optional record sink must be callable')
        self.guard=CommonTrackingPause(n,teacher_error_p999_rad=teacher_error_p999_rad,enabled=enabled,
            max_pause_s=max_pause_s,safe_velocity_rad_s=safe_velocity_rad_s)
        self.enabled=enabled;self.source=source;self.sink=sink;self.pending=None;self.records=[]
        self.last_info=None;self._installed=None

    @torch.no_grad()
    def control(self,world,task,raw_action24,live,dt_s,*,original_guard_feedback=None,original_control=None):
        original=(self._installed[2] if self._installed is not None else task.control) if original_control is None else original_control
        if not self.enabled:
            return original(raw_action24,live,dt_s),dict(enabled=False,counts_as_student_success=False)
        if not callable(original_guard_feedback):
            raise ValueError('explicit original guard feedback callback required')
        if getattr(task,'tripod_recovery_runtime',None) is not None:
            raise ValueError('common pause is an independent experiment; cannot alter Tripod reservations')
        if self.pending is not None:raise RuntimeError('previous native submission record not settled')
        if (world.n!=self.guard.n or task.command.shape!=(world.n,12)
                or live.shape!=(world.n,) or live.dtype!=torch.bool or live.device!=task.command.device):
            raise ValueError('canonical native environment count, live and command buffers required')
        # Validate the action before allowing original control to mutate caches.
        self.guard._validator.project(raw_action24)
        before=task.command.detach().clone();readback_before=read_common_pause_targets(world)
        physics_step=int(world._physics_steps)
        allowed=original(raw_action24,live,dt_s)
        governed=task.command.detach().clone()
        feedback=common_pause_feedback_from_native(world,task)
        clock=readback_before['clock'];sim=world.w.sim
        if (int(world._physics_steps)!=clock['tensor_world_physics_step']
                or int(sim.current_time_step_index)!=clock['sdk_physics_step']
                or float(sim.current_time)!=clock['sdk_timestamp_s']):
            raise ValueError('original control changed the SDK clock before transport')
        feedback['pose_state']=replace(feedback['pose_state'],live=feedback['pose_state'].live&live)
        feedback['terminal'] |= ~live
        priority=original_guard_feedback(world,task,live)
        if not isinstance(priority,dict) or any(key not in priority for key in
                ('mandatory_release_mask','mandatory_wrist_release_mask','overload')):
            raise ValueError('original guard feedback requires explicit mandatory finger/wrist and overload masks')
        # Original free-wrist control converts requests to task.command dtype.
        # Preserve the raw request in evidence; pose holds use that same native
        # controller dtype/device instead of changing the Actor action contract.
        controller_action=raw_action24.to(task.command)
        _,info=self.guard.project(controller_action,**feedback,dt_s=dt_s,**priority)
        self.pending=dict(physics_step_before=physics_step,live=live.detach().clone(),source=self.source,
            raw_action24=raw_action24.detach().clone(),logical_command_before_control_rad=before,
            raw_action24_dtype=str(raw_action24.dtype),original_controller_action24=controller_action.detach().clone(),
            original_controller_action24_dtype=str(controller_action.dtype),
            original_governed_target_rad=governed,
            backend_target_readback_before_control_rad=readback_before['backend_active_rad'],
            isaaclab_command_cache_before_control_rad=readback_before['isaaclab_command_cache_active_rad'],
            backend_all_named_targets_before_control=readback_before['hands'],
            target_readback_clock_before_control=readback_before['clock'],target_source=readback_before['target_source'],dt_s=float(dt_s),
            original_controller_requested_target_rad=getattr(task,'joint_requested_target',governed).detach().clone(),
            measured_base_pose_w=world.roots.detach().clone(),current_common_offset_m=task.offset.detach().clone(),
            candidate_state=info['state'].detach().clone(),native_submission_confirmed=False,
            candidate_motion_qualified=False,original_physical_terminal=task.done.detach().clone(),
            original_mandatory_release_mask=priority['mandatory_release_mask'].detach().clone(),
            original_mandatory_wrist_release_mask=priority['mandatory_wrist_release_mask'].detach().clone(),
            command_cache_is_backend_readback=False,geometry_prediction_is_contact=False,
            counts_as_student_success=False,original_transport_guards_still_required=True)
        self.last_info=info
        unavailable=(info['pause_hand']&live[:,None]
            &(~info['cancel_translation']|~info['cancel_rotation']))
        if bool(unavailable.any()):
            # A zero projected action cannot cancel targets already made by
            # original control. Without a valid measured hold, reject before
            # advance instead of silently retaining its ordinary wrist goals.
            # Original mandatory target increments are left intact; the caller
            # must handle this explicit candidate refusal through its original
            # fail-safe path, never as a fabricated physical task terminal.
            self.pending.update(event_kind='candidate_refused_before_transport',
                refusal='ordinary_wrist_hold_unavailable',unavailable_ordinary_wrist_mask=unavailable.clone())
            if self.sink is not None:self.sink(self.pending)
            raise RuntimeError('common pause ordinary wrist hold unavailable; candidate not qualified before transport')
        # Preserve the original controller's corrections if an old command is
        # outside the existing joint/lead envelope. A pause cannot restore an
        # illegal prior target merely by shrinking a legal governed correction.
        q=task.m['q'];low=task.low.maximum(q-.1);high=task.high.minimum(q+.1)
        required_correction=(before<low)|(before>high)
        freeze=info['cancel_finger_closure']&live[:,None]&~required_correction
        task.command[freeze]=before[freeze]
        # Current task.offset may have changed in original control. The native
        # feedback builder above subtracts that *current* offset, and validates
        # actual base/seed quaternions in the original intrinsic XYZ chart.
        apply_common_pause_wrist_targets(world,task,info)
        info['applied_finger_freeze_mask']=freeze.clone()
        info['original_allowed_preserved']=True
        self.pending.update(logical_command_after_pause_rad=task.command.detach().clone(),candidate_motion_qualified=True,
            wrist_target_after_pause_offset_m=task.palm_targets.detach().clone(),
            wrist_target_after_pause_intrinsic_xyz_rad=task.rotation_targets.detach().clone())
        return allowed,info

    @torch.no_grad()
    def after_advance(self,world,task,live):
        """Read actual SDK target buffers only after original advance returned.

        Logical cache, float32 encoding and actual backend readback remain
        distinct fields. A mismatch is retained as unconfirmed evidence.
        """
        if not self.enabled:return None
        if self.pending is None:raise RuntimeError('no post-control pause record pending')
        row=self.pending
        if not row['candidate_motion_qualified']:raise RuntimeError('candidate not qualified for transport')
        if (int(world._physics_steps)!=row['physics_step_before']+1
                or not torch.equal(live,row['live'])):
            raise ValueError('same live scope and exactly one original physics advance required')
        readback=read_common_pause_targets(world);backend=readback['backend_active_rad']
        earlier=row['target_readback_clock_before_control'];later=readback['clock']
        if (later['sdk_physics_step']!=earlier['sdk_physics_step']+1
                or abs(later['sdk_timestamp_s']-earlier['sdk_timestamp_s']-row['dt_s'])>1e-9):
            raise ValueError('actual SDK target-readback clock did not advance exactly one original substep')
        logical=world.commands.reshape(world.n,12).detach().clone()
        expected=logical.float()
        matched=(bool(live.any()) and torch.equal(logical[live],row['logical_command_after_pause_rad'][live])
            and torch.equal(expected[live].detach().cpu().double(),backend[live].double()))
        row.update(physics_step_after=int(world._physics_steps),logical_command_submitted_rad=logical,
            sdk_expected_float32_rad=expected,backend_target_readback_after_advance_rad=backend,
            isaaclab_command_cache_after_advance_rad=readback['isaaclab_command_cache_active_rad'],
            backend_all_named_targets_after_advance=readback['hands'],target_readback_clock_after_advance=readback['clock'],
            original_transport_root_target_cache_w=world.root_targets.detach().clone(),
            native_submission_confirmed=bool(matched),
            submission_confirmation_scope='actual_backend_active_joint_target_readback;not_physical_tracking_or_contact')
        self.records.append(row);self.pending=None
        if self.sink is not None:self.sink(row)
        return row

    def uninstall(self):
        """Restore original methods; preserve any unconfirmed pending evidence."""
        if self._installed is None:return
        world,task,original_control,original_advance,wrapped_control,wrapped_advance,control_present,advance_present=self._installed
        if task.control is not wrapped_control:raise RuntimeError('another owner replaced installed control')
        if wrapped_advance is not None:
            if world.advance is not wrapped_advance:raise RuntimeError('another owner replaced installed advance')
        if control_present:task.control=original_control
        else:del task.control
        if wrapped_advance is not None:
            if advance_present:world.advance=original_advance
            else:del world.advance
        if getattr(task,'common_tracking_pause_runtime',None) is self:del task.common_tracking_pause_runtime
        self._installed=None


def install_common_tracking_pause(world,task,*,teacher_error_p999_rad,source,
                                 original_guard_feedback,sink=None,max_pause_s=2.):
    """Explicit independent hybrid opt-in for an existing env.step branch.

    Instance wrappers call original control/advance exactly once. TensorWorld
    source and all its transport/physics guards remain untouched. The advance
    wrapper only reads/records targets after the original method returns.
    Call ``uninstall`` in the branch runner's ``finally`` block.
    """
    if not callable(original_guard_feedback):raise ValueError('original guard feedback callback required')
    if getattr(task,'common_tracking_pause_runtime',None) is not None:
        raise ValueError('a common tracking pause runtime is already installed')
    if getattr(task,'tripod_recovery_runtime',None) is not None:
        raise ValueError('common pause requires an independent hybrid branch')
    runtime=CommonTrackingPauseRuntime(world.n,teacher_error_p999_rad=teacher_error_p999_rad,
        enabled=True,source=source,sink=sink,max_pause_s=max_pause_s)
    original_control=task.control;original_advance=getattr(world,'advance',None)
    control_present='control' in vars(task);advance_present='advance' in vars(world)
    def control(action,live,dt):
        allowed,_=runtime.control(world,task,action,live,dt,original_control=original_control,
            original_guard_feedback=original_guard_feedback)
        return allowed
    advance=None
    if original_advance is not None:
        def advance(active_task,allowed,live):
            if runtime.pending is None or not runtime.pending['candidate_motion_qualified']:
                raise RuntimeError('common pause candidate not qualified before original advance')
            result=original_advance(active_task,allowed,live)
            runtime.after_advance(world,active_task,live)
            return result
    task.control=control
    if advance is not None:world.advance=advance
    task.common_tracking_pause_runtime=runtime
    runtime._installed=(world,task,original_control,original_advance,control,advance,control_present,advance_present)
    return runtime
