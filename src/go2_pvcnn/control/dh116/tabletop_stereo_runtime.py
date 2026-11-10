"""Actual native stereo state protocol shared by collection and Student loops.

Only SensorBatch/StereoHistoryBatch is given to the Actor. The parallel 371D
state is training-only. Its supervision mask is deliberately conservative:
hardware/source-qualified labels do not become valid just because finite
simulation data exists.
"""
from pathlib import Path

import torch

from .tabletop_stereo_camera import StereoCamera
from .tabletop_stereo_contract import state_contract
from .tabletop_stereo_observation import SensorBatch, StereoHistory
from .tabletop_stereo_perception import WirePointExtractor
from .tabletop_stereo_teacher_collector import PreActionState, StepResult
from .tabletop_stereo_preprocessing import processing_contract


class NativeStereoEnvironment:
    evidence_scope = 'native_runtime_capture_unqualified'

    def __init__(self, bridge, camera_config, *, history=4, owner='student', checkpoint_sha256=None):
        self.bridge = bridge
        self.evidence_scope = getattr(bridge, 'evidence_scope', 'unqualified')
        self.n, self.device = bridge.n, bridge.device
        self.camera_config = Path(camera_config)
        self.preprocessing = processing_contract(camera_config)
        self.camera = StereoCamera.from_native(bridge.world.sim, bridge.device_world.origins,
            camera_config, device=self.device)
        self.history = StereoHistory(self.n, history, device=self.device)
        self.extractor = WirePointExtractor()
        self._cached = None
        self._pending_reset = torch.ones(self.n, device=self.device, dtype=torch.bool)
        self.bridge.set_execution_context(owner=owner, checkpoint_sha256=checkpoint_sha256)
        self.bridge.set_action_owners(['teacher' if owner == 'teacher' else 'student']*self.n)
        self._required_sources = {key for row in state_contract()['dimensions']
            for key in row['required_validity_keys']}
        self.last_camera_packet = None

    def set_action_owners(self, owners):
        self.bridge.set_action_owners(owners)

    @torch.no_grad()
    def snapshot(self):
        if self._cached is not None:
            return self._cached.clone()
        identities = self.bridge.pre_action_identities(self.camera.calibration.sha256)
        for identity in identities:
            identity['observation_preprocessor'] = self.preprocessing
            identity['observation_preprocessor_sha256'] = self.preprocessing['processing_sha256']
        pair = self.camera.sample(identity=identities)
        extraction = self.extractor.extract(pair['left_rgb'], pair['depth_m'], pair['valid'],
            pair['K_left'], pair['T_world_left'], reset_mask=self._pending_reset)
        h, w = pair['left_rgb'].shape[-2:]
        sensor = SensorBatch(left_rgb=pair['left_rgb'].float(), right_rgb=pair['right_rgb'].float(),
            depth_m=pair['depth_m'].float(), depth_valid=pair['valid'],
            depth_confidence=pair['confidence'].float(),
            points_world=extraction['points_world'].float(), point_valid=extraction['point_valid'],
            proprio=self.bridge.proprioception(), dt=torch.full((self.n,), 1/60, device=self.device),
            image_age=torch.zeros(self.n, device=self.device),
            image_valid=torch.ones(self.n, 2, device=self.device, dtype=torch.bool))
        history = self.history.update(sensor, reset_mask=self._pending_reset)
        self._pending_reset.zero_()
        critic = self.bridge.critic.clone()
        # These masks describe supervised Student labels, NOT whether the
        # privileged critic may use finite task state under native guards.
        label_valid = torch.zeros_like(critic, dtype=torch.bool)
        sources = {name: torch.zeros(self.n, device=self.device, dtype=torch.bool)
            for name in self._required_sources}
        for name in ('source_valid', 'identity_aligned', 'timestamp_valid'):
            if name in sources:
                sources[name].fill_(True)
        self._cached = PreActionState(sensor_history=history, privileged=critic,
            privileged_valid=label_valid, identities=pair['identity'],
            privileged_identities=[dict(row) for row in pair['identity']],
            extra_labels={'stereo_disparity_px': pair['disparity_px'].clone()},
            label_validity={'stereo_disparity_px': pair['valid'].clone()},
            state_sources=sources, force_qualification=None).validate()
        self.last_camera_packet = pair
        return self._cached.clone()

    def reset(self):
        self.bridge.reset()
        self.clear_sensor_history()
        return self.snapshot()

    def clear_sensor_history(self, *_):
        """Start a new sensor epoch after an externally audited physical reset."""
        self._cached = None
        self._pending_reset.fill_(True)
        self.history.reset(self._pending_reset)
        # ReferenceTime may identify simulation time rather than render calls.
        # A new reset UID is a new epoch even when physics time did not move.
        self.camera.last_frame = -1

    @torch.no_grad()
    def step(self, issued_actions):
        before = self.snapshot()
        start = int(self.bridge.device_world._physics_steps)
        start_time = float(self.bridge.world.sim.current_time)
        critic, reward, done, extra = self.bridge.step(issued_actions)
        stop = int(self.bridge.device_world._physics_steps)
        stop_time = float(self.bridge.world.sim.current_time)
        execution = [dict(physics_tick_start=start, physics_tick_end=stop,
            simulation_timestamp_start=start_time, simulation_timestamp_end=stop_time)
            for _ in range(self.n)]
        if stop <= start or stop_time <= start_time:
            raise RuntimeError('native step did not advance actual physics')
        self._cached = None
        self._pending_reset.copy_(done)
        after = self.snapshot()
        world = self.bridge.device_world
        sdk = [getattr(hand.data, 'joint_pos_target', None) for hand in world.hands]
        sdk = (torch.cat([target[:, ids] for target, ids in zip(sdk, world.w.ids)], -1).detach().clone()
            if all(target is not None for target in sdk) else None)
        valid = ~done
        command = self.bridge.task.command.detach().clone()
        palm = self.bridge.palm_poses()
        command = torch.where(valid[:, None], command, float('nan'))
        palm = torch.where(valid[:, None, None], palm, float('nan'))
        if sdk is not None:
            sdk = torch.where(valid[:, None], sdk, float('nan'))
        applied = dict(actual_sdk_joint_target_rad=sdk,
            actual_sdk_joint_target_valid=valid.clone() if sdk is not None else torch.zeros_like(valid),
            controller_joint_command_rad=command,
            actual_measured_palm_pose_wxyz=palm,
            post_step_readback_valid=valid.clone(),
            terminal_readback='NA_after_reset;use_preserved_pre_reset_native_physics_rows',
            source='readback_and_named_controller_targets;not_normalized_applied_action')
        info = dict(execution=execution, evidence_scope=self.evidence_scope,
            success=extra['success'].clone(), failure=extra['failure'].clone(),
            reward_terms={k: v.detach().float().clone() for k, v in extra['terms'].items()},
            first_episode=torch.tensor([row.get('trial_episode_index', row['reset_id']+1) == 1
                for row in before.identities],
                device=self.device, dtype=torch.bool),
            recovered=extra.get('recovered', torch.zeros_like(done)).clone(),
            restored=torch.zeros_like(done), native_evidence_file=str(self.bridge.output / 'native_physics.jsonl'),
            qualification='runtime_capture_not_native_source_qualification')
        return StepResult(next_state=after, reward=reward, done=done, info=info,
            issued_actions=issued_actions.detach().clone(), applied_targets=applied)
