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
from .tabletop_stereo_perception import WirePointExtractor, TrackedWirePointExtractor
from .tabletop_stereo_geometry import world_to_palms
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
        if callable(getattr(bridge,'bind_camera',None)):bridge.bind_camera(self.camera)
        else:bridge.camera=self.camera
        self.history = StereoHistory(self.n, history, device=self.device)
        self.extractor = (WirePointExtractor() if self.preprocessing['perception']['algorithm']=='warm_colour_visible_simple_path_ZhangSuen_v1'
            else TrackedWirePointExtractor())
        self._cached = None
        self._pending_reset = torch.ones(self.n, device=self.device, dtype=torch.bool)
        self.bridge.set_execution_context(owner=owner, checkpoint_sha256=checkpoint_sha256)
        self.bridge.set_action_owners(['teacher' if owner == 'teacher' else 'student']*self.n)
        self._required_sources = {key for row in state_contract()['dimensions']
            for key in row['required_validity_keys']}
        self.last_camera_packet = None
        self.last_visible_geometry = None

    def set_action_owners(self, owners):
        self.bridge.set_action_owners(owners)

    def _encoder_label_validity(self,critic,pair,requested,proprio):
        """Training-only SDK encoder truth, checked against this exact pre-state.

        A finite 371D tensor is insufficient. Re-read the actual active joint
        indices, Task measurement/normalization and both surrounding identities.
        This confirms simulator labels only; it grants no hardware/force/vision
        qualification and cannot enable any slice beyond q/qd [0:24].
        """
        empty=torch.zeros_like(critic,dtype=torch.bool)
        try:
            bridge=self.bridge;world=bridge.device_world;task=bridge.task
            first=bridge.pre_action_identities(self.camera.calibration.sha256)
            low,high=task.low,task.high
            if any(not isinstance(value,torch.Tensor) or value.shape!=(12,)
                    or not value.is_floating_point() or value.device!=critic.device for value in (low,high)):
                return empty
            if len(world.hands)!=2 or len(world.w.ids)!=2:return empty
            positions=[];velocities=[]
            for hand,indices in zip(world.hands,world.w.ids):
                q,qd=hand.data.joint_pos,hand.data.joint_vel
                if (not isinstance(indices,(list,tuple)) or len(indices)!=6 or len(set(indices))!=6
                        or any(type(index) is not int for index in indices)):
                    return empty
                if any(not isinstance(value,torch.Tensor) or value.ndim!=2 or value.shape[0]!=self.n
                        or not value.is_floating_point() or value.device!=critic.device for value in (q,qd)):
                    return empty
                if q.shape!=qd.shape or min(indices)<0 or max(indices)>=q.shape[1]:return empty
                positions.append(q[:,indices].detach().double().clone())
                velocities.append(qd[:,indices].detach().double().clone())
            q=torch.cat(positions,-1);qd=torch.cat(velocities,-1)
            measured_q,measured_qd=task.m['q'],task.m['qd']
            if any(not isinstance(value,torch.Tensor) or value.shape!=(self.n,12)
                    or value.device!=critic.device or not value.is_floating_point() for value in (measured_q,measured_qd)):
                return empty
            last=bridge.pre_action_identities(self.camera.calibration.sha256)
            rows=pair['identity']
            if any(not isinstance(value,list) or len(value)!=self.n for value in (requested,first,last,rows)):
                return empty
            keys=('run_uuid','env_id','episode_uid','reset_id','policy_tick','physics_tick_start',
                  'physics_tick_end','simulation_timestamp','calibration_sha256')
            aligned=torch.tensor([all(isinstance(value[i],dict) for value in (requested,first,last,rows))
                and first[i].get('env_id')==i and all(key in first[i] and all(value[i].get(key)==first[i][key]
                    for value in (requested,last,rows)) for key in keys)
                for i in range(self.n)],device=critic.device,dtype=torch.bool)
            bounds=torch.isfinite(low)&torch.isfinite(high)&(high>low)
            # Invalid denominators stay masked; never substitute them as valid
            # zeros or clip an outlying measured joint into its nominal range.
            denominator=torch.where(bounds,high-low,torch.ones_like(high)).double()
            normalized_q=(2*(q-low.double())/denominator-1).to(critic.dtype)
            normalized_qd=(qd/.3).to(critic.dtype)
            close=lambda a,b:torch.isfinite(a)&torch.isfinite(b)&torch.isclose(a.double(),b.double(),atol=1e-6,rtol=1e-5)
            valid_q=(close(q,measured_q)&close(q,proprio[:,:12])&close(normalized_q,critic[:,:12]))
            valid_qd=(close(qd,measured_qd)&close(qd,proprio[:,12:24])&close(normalized_qd,critic[:,12:24]))
            empty[:,:24]=torch.cat((valid_q,valid_qd),-1)&bounds.repeat(2)[None]&aligned[:,None]
        except (AttributeError,KeyError,TypeError,IndexError,ValueError):
            return torch.zeros_like(critic,dtype=torch.bool)
        return empty

    @torch.no_grad()
    def snapshot(self):
        if self._cached is not None:
            return self._cached.clone()
        identities = self.bridge.pre_action_identities(self.camera.calibration.sha256)
        for identity in identities:
            identity['observation_preprocessor'] = self.preprocessing
            identity['observation_preprocessor_sha256'] = self.preprocessing['processing_sha256']
        pair = self.camera.sample(identity=identities)
        extraction_kwargs=dict(reset_mask=self._pending_reset)
        if isinstance(self.extractor,TrackedWirePointExtractor):
            extraction_kwargs.update(timestamp=[row['simulation_timestamp'] for row in pair['identity']],
                point_confidence=pair['confidence'])
        extraction = self.extractor.extract(pair['left_rgb'], pair['depth_m'], pair['valid'],
            pair['K_left'], pair['T_world_left'], **extraction_kwargs)
        if 'timestamp_anomaly' in extraction:self._pending_reset|=extraction['timestamp_anomaly']
        h, w = pair['left_rgb'].shape[-2:]
        sensor = SensorBatch(left_rgb=pair['left_rgb'].float(), right_rgb=pair['right_rgb'].float(),
            depth_m=pair['depth_m'].float(), depth_valid=pair['valid'],
            depth_confidence=pair['confidence'].float(),
            points_world=extraction['points_world'].float(), point_valid=extraction['point_valid'],
            point_confidence=extraction.get('confidence',extraction['point_valid'].float()).float(),
            proprio=self.bridge.proprioception(), dt=torch.full((self.n,), 1/60, device=self.device),
            image_age=torch.zeros(self.n, device=self.device),
            image_valid=torch.ones(self.n, 2, device=self.device, dtype=torch.bool))
        # These SAME measured-FK local coordinates are also typed Actor inputs;
        # they are observed geometry, not simulator nodes or qualified labels.
        palms=sensor.proprio[:,24:38].reshape(self.n,2,7)
        local_points=world_to_palms(sensor.points_world,palms)
        local_valid=sensor.point_valid[:,None,:].expand(-1,2,-1).clone()&torch.isfinite(local_points).all(-1)
        geometry=dict(visible_wire_points_world_m=sensor.points_world.clone(),
            visible_wire_points_palms_m=torch.where(local_valid[...,None],local_points,float('nan')),
            visible_wire_points_palms_valid=local_valid,
            visible_wire_geometry_source='derived_RGB_stereo_world_points_and_same_sample_proprio_FK',
            visible_wire_geometry_is_ground_truth=False)
        history = self.history.update(sensor, reset_mask=self._pending_reset)
        self._pending_reset.zero_()
        critic = self.bridge.critic.clone()
        # These masks describe supervised Student labels, NOT whether the
        # privileged critic may use finite task state under native guards.
        label_valid = self._encoder_label_validity(critic,pair,identities,sensor.proprio)
        sources = {name: torch.zeros_like(critic, dtype=torch.bool)
            for name in self._required_sources}
        for name in ('source_valid', 'identity_aligned', 'timestamp_valid','proprio_source_confirmed'):
            if name in sources:
                sources[name].copy_(label_valid)
        self._cached = PreActionState(sensor_history=history, privileged=critic,
            privileged_valid=label_valid, identities=pair['identity'],
            privileged_identities=[dict(row) for row in pair['identity']],
            extra_labels={'stereo_disparity_px': pair['disparity_px'].clone(),
                'training_label_source_scope':'simulation_SDK_encoder_readback_only',
                'training_label_evidence_scope':self.evidence_scope,'hardware_feedback_qualified':False,**geometry},
            label_validity={'stereo_disparity_px': pair['valid'].clone()},
            state_sources=sources, force_qualification=None).validate()
        self.last_visible_geometry={key:(value.clone() if isinstance(value,torch.Tensor) else value) for key,value in geometry.items()}
        self.last_visible_geometry['wire_tracking']={key:(value.clone() if isinstance(value,torch.Tensor) else value)
            for key,value in extraction.items() if key not in ('points_world','point_valid','xyz','valid')}
        pair['wire_tracking']=self.last_visible_geometry['wire_tracking']
        pair.update({key:(value.clone() if isinstance(value,torch.Tensor) else value) for key,value in geometry.items()})
        self.last_camera_packet = pair
        return self._cached.clone()

    def reset(self):
        self.bridge.reset()
        self.clear_sensor_history()
        return self.snapshot()

    def clear_sensor_history(self, *_):
        """Start a new sensor epoch after an externally audited physical reset."""
        self._cached = None
        self.last_visible_geometry = None
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
